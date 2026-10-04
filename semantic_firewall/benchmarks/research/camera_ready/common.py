"""Shared code for the camera-ready evaluation.

Rules this harness enforces (they answer the reviewers' protocol concerns):
  * Parameters are tuned on neuralchemy `validation`; final numbers come from `test` once.
  * Every run gets its own empty semantic cache directory. "Warm" means the cache is
    seeded from the *train* split only (train/validation/test share no group_id and no
    exact text), so test prompts are never in the cache before they are evaluated.
  * Every per-sample prediction is written to predictions.jsonl, so every number in the
    paper (metrics, McNemar tests, cost, latency) is recomputable from raw records.
  * LLM failures are counted and reported, never silently turned into "benign".
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_ROOT = PROJECT_ROOT / "semantic_firewall" / "benchmarks" / "research" / "results_camera_ready"
RUNS_DIR = OUT_ROOT / "runs"
WORK_DIR = OUT_ROOT / "_work"

ALL_AGENTS = [
    "Context Flooding Detector", "PII Detector", "Secrets Detector", "Abuse Detector",
    "Injection Detector", "Unsafe Content Detector", "Threat Intel Detector", "Custom Rules Detector",
]
DETERMINISTIC_AGENTS = [a for a in ALL_AGENTS if a not in {"Injection Detector", "Unsafe Content Detector"}]

# Environment applied to every run unless a config overrides it.
BASE_ENV = {
    "SEMANTIC_FIREWALL_CACHE_TTL_SEC": "0",          # no exact-match result cache: every prompt is analysed
    "SEMANTIC_FIREWALL_AUDIT_ENABLED": "0",          # keep SQLite I/O out of latency measurements
    "SEMANTIC_FIREWALL_DISABLE_LLM_DETECTORS": "0",
    "SEMANTIC_FIREWALL_LLM_GATE_ENABLED": "1",
    "SEMANTIC_FIREWALL_LLM_GATE_THRESHOLD": "1.0",
    "SEMANTIC_FIREWALL_EARLY_EXIT_ON_BLOCK": "1",
    "SEMANTIC_FIREWALL_DISABLED_AGENTS": "",
    "SEMANTIC_FIREWALL_SEMANTIC_CACHE_ENABLED": "1",
    "SEMANTIC_FIREWALL_CACHE_SIM_THRESHOLD": "0.90",
    "SEMANTIC_FIREWALL_CACHE_WRITEBACK": "1",
    "SEMANTIC_FIREWALL_CACHE_WRITEBACK_REQUIRE_LLM": "1",
    "SEMANTIC_FIREWALL_CACHE_WRITEBACK_MIN_CONFIDENCE": "0.85",
    "SEMANTIC_FIREWALL_LLM_AGENT_REGEX_ENABLED": "1",
    "SEMANTIC_FIREWALL_ENSEMBLE_ENABLED": "1",
    "SEMANTIC_FIREWALL_ALLOWLIST_MODE": "skip_llm",
}


@dataclass(frozen=True)
class SystemConfig:
    name: str
    description: str
    env: dict
    uses_llm: bool = True


def _cfg(name, description, uses_llm=True, **env):
    return SystemConfig(name, description, {f"SEMANTIC_FIREWALL_{k}": str(v) for k, v in env.items()}, uses_llm)


# Ablation / system variants. Names map to the paper's Table 4 rows.
CONFIGS = {c.name: c for c in [
    _cfg("full", "Full pipeline: regex + semantic cache + 8 detectors + keyword-gated LLM, early exit on BLOCK."),
    _cfg("full_all_unresolved", "Full pipeline, but every prompt not blocked earlier goes to the LLM (Algorithm 1 as written).",
         LLM_GATE_ENABLED=0),
    _cfg("no_cache", "Full pipeline without semantic memory.", SEMANTIC_CACHE_ENABLED=0),
    _cfg("no_early_exit", "Full pipeline; the LLM also runs when the deterministic stage already blocked.",
         EARLY_EXIT_ON_BLOCK=0),
    _cfg("fast_layers", "Regex + semantic cache + deterministic detectors; no LLM calls.", uses_llm=False,
         DISABLE_LLM_DETECTORS=1),
    _cfg("deterministic_only", "Regex pre-screen + deterministic detectors; no cache, no LLM.", uses_llm=False,
         DISABLE_LLM_DETECTORS=1, SEMANTIC_CACHE_ENABLED=0),
    _cfg("regex_only", "Injection/unsafe regex pre-screen + threat-intel signatures only.", uses_llm=False,
         DISABLE_LLM_DETECTORS=1, SEMANTIC_CACHE_ENABLED=0,
         DISABLED_AGENTS="Context Flooding Detector,PII Detector,Secrets Detector,Abuse Detector,Custom Rules Detector"),
    _cfg("cache_only", "Semantic cache only (misses are allowed).", uses_llm=False,
         DISABLE_LLM_DETECTORS=1, DISABLED_AGENTS=",".join(ALL_AGENTS)),
    _cfg("llm_agents_only", "Reviewer 2 baseline: the two LLM agents in parallel on every prompt; no regex, "
         "no deterministic detectors, no cache.", LLM_GATE_ENABLED=0, LLM_AGENT_REGEX_ENABLED=0,
         SEMANTIC_CACHE_ENABLED=0, DISABLED_AGENTS=",".join(DETERMINISTIC_AGENTS)),
    _cfg("injection_llm_only", "Single LLM injection agent on every prompt (closest to an 'LLM-only gate').",
         LLM_GATE_ENABLED=0, LLM_AGENT_REGEX_ENABLED=0, SEMANTIC_CACHE_ENABLED=0,
         DISABLED_AGENTS=",".join(DETERMINISTIC_AGENTS + ["Unsafe Content Detector"])),
]}
for _agent in ALL_AGENTS:
    CONFIGS[f"loo:{_agent}"] = _cfg(f"loo:{_agent}", f"Full pipeline without the {_agent} (leave-one-out).",
                                    DISABLED_AGENTS=_agent)


# ── Datasets ─────────────────────────────────────────────────────────────────

def _load(name, *args, **kwargs):
    from datasets import load_dataset
    return load_dataset(name, *args, **kwargs)


def load_samples(dataset: str, split: str = "test", max_samples: int = 0, seed: int = 0) -> list[dict]:
    """Return [{"id", "text", "label"}] with label 1 = attack/unsafe, 0 = benign."""
    rows: list[dict] = []
    if dataset == "neuralchemy":
        for r in _load("neuralchemy/Prompt-injection-dataset", "core")[split]:
            rows.append({"id": r["group_id"], "text": r["text"], "label": int(r["label"]),
                         "category": r.get("category")})
    elif dataset == "deepset":
        for r_split in (["train", "test"] if split == "all" else [split]):
            for i, r in enumerate(_load("deepset/prompt-injections")[r_split]):
                rows.append({"id": f"deepset-{r_split}-{i}", "text": r["text"], "label": int(r["label"])})
    elif dataset == "beavertails":
        for i, r in enumerate(_load("PKU-Alignment/BeaverTails", split="30k_test")):
            rows.append({"id": f"bt-{i}", "text": r["prompt"], "label": 0 if r["is_safe"] else 1})
    elif dataset == "toxicchat":
        for r in _load("lmsys/toxic-chat", "toxicchat0124")[split]:
            label = 1 if (r.get("toxicity") == 1 or r.get("jailbreaking") == 1) else 0
            rows.append({"id": r["conv_id"], "text": r["user_input"], "label": label})
    elif dataset == "no_robots":
        for i, r in enumerate(_load("HuggingFaceH4/no_robots")[split]):
            rows.append({"id": f"nr-{i}", "text": r["prompt"], "label": 0})
    elif dataset == "alpaca":
        for i, r in enumerate(_load("tatsu-lab/alpaca")["train"]):
            text = (r["instruction"] + " " + (r.get("input") or "")).strip()
            rows.append({"id": f"alpaca-{i}", "text": text, "label": 0})
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    rows = [r for r in rows if r["text"] and r["text"].strip()]
    if max_samples and len(rows) > max_samples:
        rng = random.Random(seed)
        rows = rng.sample(rows, max_samples)
    return rows


def train_attack_texts(dataset: str = "neuralchemy") -> list[str]:
    """Attack prompts from the train split; the only source allowed for a warm cache."""
    return [r["text"] for r in load_samples(dataset, "train") if r["label"] == 1]


# ── Firewall construction ────────────────────────────────────────────────────

def apply_env(config: SystemConfig, overrides: Optional[dict] = None, cache_dir: Optional[Path] = None) -> dict:
    env = dict(BASE_ENV)
    env.update(config.env)
    if overrides:
        env.update({k: str(v) for k, v in overrides.items()})
    if cache_dir is not None:
        env["SEMANTIC_FIREWALL_SEMANTIC_CACHE_PATH"] = str(cache_dir)
    os.environ.update(env)
    return env


def build_firewall(config: SystemConfig, run_name: str, cache_protocol: str = "cold",
                   overrides: Optional[dict] = None, quiet: bool = True):
    """Fresh orchestrator with its own empty cache directory.

    cache_protocol:
      cold         empty cache, online write-back on (cache learns during the run)
      warm_frozen  cache seeded with train-split attacks, write-back off
      warm_online  seeded with train-split attacks, write-back on
    """
    cache_dir = WORK_DIR / "caches" / run_name.replace(":", "_").replace(" ", "_")
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    overrides = dict(overrides or {})
    if cache_protocol == "warm_frozen":
        overrides.setdefault("SEMANTIC_FIREWALL_CACHE_WRITEBACK", "0")
    env = apply_env(config, overrides, cache_dir)
    from semantic_firewall.core.orchestrator.orchestrator import SemanticFirewallOrchestrator

    with quiet_stdout(quiet):
        fw = SemanticFirewallOrchestrator(db_path=str(WORK_DIR / "audit.db"))
        fw.semantic_cache.reset()
        seeded = 0
        if cache_protocol in {"warm_frozen", "warm_online"} and fw.semantic_cache.enabled:
            seeded = fw.semantic_cache.seed_threats(train_attack_texts(), origin="train_seed")
    return fw, env, seeded


@contextlib.contextmanager
def quiet_stdout(enabled: bool = True):
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ── Running and recording ────────────────────────────────────────────────────

def _llm_failed(decision) -> bool:
    for result in getattr(decision, "agent_results", []) or []:
        meta = getattr(result, "meta", {}) or {}
        if meta.get("llm_skipped_reason") in {"llm_error", "missing_api_key"}:
            return True
        if meta.get("llm_called") and meta.get("llm_parse_status") not in (None, "ok", "not_attempted"):
            return True
        if getattr(result, "threat_type", "") == "SYSTEM_UNAVAILABLE":
            return True
    return False


def decision_record(sample: dict, decision, latency_ms: float) -> dict:
    return {
        "id": sample["id"],
        "label": int(sample["label"]),
        "pred": 1 if decision.action in {"BLOCK", "FLAG", "REDACT"} else 0,
        "action": decision.action,
        "resolution_stage": getattr(decision, "resolution_stage", "unknown"),
        "llm_called": bool(getattr(decision, "llm_called", False)),
        "llm_failed": _llm_failed(decision),
        "latency_ms": round(latency_ms, 3),
        "stage_latency_ms": {k: round(v, 3) for k, v in (getattr(decision, "stage_latency_ms", {}) or {}).items()},
        "semantic_similarity": getattr(decision, "semantic_similarity", None),
        "triggered_agents": list(getattr(decision, "triggered_agents", []) or []),
        "degraded": bool(getattr(decision, "degraded", False)),
        "text_chars": len(sample["text"]),
    }


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run_samples(predict: Callable[[dict], dict], samples: list[dict], out_path: Path,
                workers: int = 1, progress_every: int = 50) -> list[dict]:
    """Run predict() over samples, appending to out_path; already-recorded ids are skipped (resume)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(out_path)}
    todo = [s for s in samples if s["id"] not in done]
    if done:
        print(f"  resuming: {len(done)} already recorded, {len(todo)} to go")
    started = time.time()
    with out_path.open("a", encoding="utf-8") as fh:
        if workers <= 1:
            for i, sample in enumerate(todo, 1):
                fh.write(json.dumps(predict(sample)) + "\n")
                fh.flush()
                if i % progress_every == 0:
                    print(f"  {i}/{len(todo)} ({time.time() - started:.0f}s)")
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(predict, s) for s in todo]
                for i, future in enumerate(as_completed(futures), 1):
                    fh.write(json.dumps(future.result()) + "\n")
                    fh.flush()
                    if i % progress_every == 0:
                        print(f"  {i}/{len(todo)} ({time.time() - started:.0f}s)")
    ids = {s["id"] for s in samples}
    return [r for r in read_jsonl(out_path) if r["id"] in ids]


# ── Metrics ──────────────────────────────────────────────────────────────────

def confusion(records: Iterable[dict]) -> dict:
    tp = fp = tn = fn = 0
    for r in records:
        if r["label"] == 1 and r["pred"] == 1:
            tp += 1
        elif r["label"] == 0 and r["pred"] == 1:
            fp += 1
        elif r["label"] == 0:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def rates(c: dict) -> dict:
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    div = lambda a, b: a / b if b else 0.0
    p, r = div(tp, tp + fp), div(tp, tp + fn)
    return {
        "precision": p, "recall": r, "f1": div(2 * p * r, p + r),
        "accuracy": div(tp + tn, tp + tn + fp + fn), "fpr": div(fp, fp + tn), "fnr": div(fn, fn + tp),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def bootstrap_ci(records: list[dict], metric: str, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample = [records[rng.randrange(len(records))] for _ in records]
        vals.append(rates(confusion(sample))[metric])
    return percentile(vals, 2.5), percentile(vals, 97.5)


def summarize(records: list[dict], with_ci: bool = True) -> dict:
    c = confusion(records)
    out = {"n": len(records), "n_attack": sum(r["label"] for r in records), **c, **rates(c)}
    if with_ci and records:
        for metric in ("f1", "fpr", "recall"):
            lo, hi = bootstrap_ci(records, metric)
            out[f"{metric}_ci95"] = [lo, hi]
    lat = [r["latency_ms"] for r in records if "latency_ms" in r]
    if lat:
        out.update({"latency_mean_ms": sum(lat) / len(lat), "latency_p50_ms": percentile(lat, 50),
                    "latency_p95_ms": percentile(lat, 95), "latency_p99_ms": percentile(lat, 99)})
    if records and "resolution_stage" in records[0]:
        stages: dict = {}
        for r in records:
            stages[r["resolution_stage"]] = stages.get(r["resolution_stage"], 0) + 1
        out["resolution_stage_counts"] = stages
        out["llm_call_rate"] = sum(r.get("llm_called", False) for r in records) / len(records)
    out["n_llm_failed"] = sum(r.get("llm_failed", False) for r in records)
    out["n_errors"] = sum(r.get("error") is not None for r in records if "error" in r)
    return out


# ── Similarity to a train-seeded threat cache ────────────────────────────────

def nearest_cache_similarity(texts: list[str], seed_texts: Optional[list[str]] = None,
                             name: str = "similarity_index") -> list[float]:
    """Cosine similarity of each text to its nearest train-split attack (the warm cache)."""
    from semantic_firewall.core.orchestrator.semantic_cache import SemanticCache

    cache_dir = WORK_DIR / "caches" / name
    with quiet_stdout():
        cache = SemanticCache(db_path=str(cache_dir), writeback=False)
        if cache.count() == 0:
            cache.seed_threats(seed_texts if seed_texts is not None else train_attack_texts(), origin="train_seed")
    sims: list[float] = []
    vectors = cache.embed_many(texts)
    for start in range(0, len(vectors), 256):
        res = cache.collection.query(query_embeddings=vectors[start:start + 256], n_results=1, include=["distances"])
        sims.extend(1.0 - float(d[0]) for d in res["distances"])
    return sims


# ── Provenance ───────────────────────────────────────────────────────────────

def code_fingerprint() -> str:
    """SHA-256 over the detector/orchestrator sources, stored with every result."""
    h = hashlib.sha256()
    for path in sorted((PROJECT_ROOT / "semantic_firewall" / "core").rglob("*.py")):
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def run_metadata(extra: Optional[dict] = None) -> dict:
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "code_fingerprint": code_fingerprint(),
    }
    meta.update(extra or {})
    return meta


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
