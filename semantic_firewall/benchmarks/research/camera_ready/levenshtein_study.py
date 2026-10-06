"""Is Levenshtein keyword repair (core/agents/fuzzy_keywords.py) useful?

Compares the deterministic keyword layers with repair OFF vs ON:
  * injection regex prescreen (InjectionDetectorAgent._regex_prescreen)
  * LLM-escalation keyword check (orchestrator._LLM_ESCALATION_PATTERN)
on
  1. natural data: neuralchemy validation/test, deepset test, no_robots test (benign);
     every changed prediction is counted (0 changes = reported results unaffected);
  2. obfuscated attacks: test attacks that the regex catches when clean, with one
     random edit (adjacent swap / deletion / substitution / leetspeak) in every
     regex keyword, 3 seeds.
No LLM calls. Output: results_camera_ready/levenshtein_study.json

    python -m semantic_firewall.benchmarks.research.camera_ready.levenshtein_study
"""
from __future__ import annotations

import json
import random
import re
import statistics
import time
from pathlib import Path

from semantic_firewall.benchmarks.research.camera_ready.common import load_samples
from semantic_firewall.core.agents.fuzzy_keywords import KeywordRepairer, keywords_from_patterns
from semantic_firewall.core.agents.injection_detector import InjectionDetectorAgent
from semantic_firewall.core.orchestrator.orchestrator import _LLM_ESCALATION_PATTERN

OUT = Path(__file__).resolve().parents[1] / "results_camera_ready" / "levenshtein_study.json"
SEEDS = (0, 1, 2)
LEET = {"o": "0", "i": "1", "e": "3", "a": "4", "s": "5", "t": "7"}

agent = InjectionDetectorAgent()
escalation_repairer = KeywordRepairer(keywords_from_patterns([_LLM_ESCALATION_PATTERN.pattern]))
KEYWORDS = agent.keyword_repairer.keywords


def regex_flag(text: str, fuzzy: bool) -> bool:
    agent.fuzzy_enabled = fuzzy
    return bool(agent._regex_prescreen(text)[0])


def escalates(text: str, fuzzy: bool) -> bool:
    if _LLM_ESCALATION_PATTERN.search(text):
        return True
    return fuzzy and bool(_LLM_ESCALATION_PATTERN.search(escalation_repairer.repair(text)))


def perturb(text: str, rng: random.Random) -> str:
    """One random edit inside every regex keyword occurring in the text."""
    def edit(m: re.Match) -> str:
        w = m.group(0)
        if w.lower() not in KEYWORDS:
            return w
        i = rng.randrange(1, len(w) - 1)
        op = rng.choice(["swap", "delete", "substitute", "leet"])
        if op == "swap":
            return w[:i] + w[i + 1] + w[i] + w[i + 2:]
        if op == "delete":
            return w[:i] + w[i + 1:]
        if op == "substitute":
            return w[:i] + rng.choice("abcdefghijklmnopqrstuvwxyz") + w[i + 1:]
        js = [j for j, ch in enumerate(w) if ch.lower() in LEET]
        if not js:
            return w[:i] + w[i + 1:]
        j = rng.choice(js)
        return w[:j] + LEET[w[j].lower()] + w[j + 1:]
    return re.sub(r"[A-Za-z]{5,}", edit, text)


def natural(name: str, rows: list[dict]) -> dict:
    res = {"dataset": name, "n_attack": sum(r["label"] for r in rows),
           "n_benign": sum(1 - r["label"] for r in rows)}
    for layer, fn in (("regex", regex_flag), ("escalation", escalates)):
        off = [fn(r["text"], False) for r in rows]
        on = [fn(r["text"], True) for r in rows]
        res[layer] = {
            "attacks_flagged_off": sum(o for o, r in zip(off, rows) if r["label"]),
            "attacks_flagged_on": sum(o for o, r in zip(on, rows) if r["label"]),
            "benign_flagged_off": sum(o for o, r in zip(off, rows) if not r["label"]),
            "benign_flagged_on": sum(o for o, r in zip(on, rows) if not r["label"]),
            "predictions_changed": sum(a != b for a, b in zip(off, on)),
            "changed_examples": [r["text"][:160] for a, b, r in zip(off, on, rows) if a != b][:10],
        }
    return res


def latency_ms(rows: list[dict], fuzzy: bool) -> float:
    t0 = time.perf_counter()
    for r in rows:
        regex_flag(r["text"], fuzzy)
        escalates(r["text"], fuzzy)
    return (time.perf_counter() - t0) / len(rows) * 1000


def main() -> None:
    data = {
        "neuralchemy-validation": load_samples("neuralchemy", "validation"),
        "neuralchemy-test": load_samples("neuralchemy", "test"),
        "deepset-test": load_samples("deepset", "test"),
        "no_robots-test (benign)": load_samples("no_robots", "test"),
    }
    out = {"keywords_injection": len(KEYWORDS),
           "keywords_escalation": sorted(escalation_repairer.keywords),
           "natural": [natural(k, v) for k, v in data.items()]}

    test_attacks = [r["text"] for r in data["neuralchemy-test"] if r["label"] == 1]
    caught = [t for t in test_attacks if regex_flag(t, False)]
    obf = []
    for seed in SEEDS:
        rng = random.Random(seed)
        pert = [perturb(t, rng) for t in caught]
        rng = random.Random(seed)
        pert_all = [perturb(t, rng) for t in test_attacks]
        obf.append({
            "seed": seed,
            "n_regex_caught_clean": len(caught),
            "regex_caught_off": sum(regex_flag(t, False) for t in pert),
            "regex_caught_on": sum(regex_flag(t, True) for t in pert),
            "n_test_attacks": len(test_attacks),
            "escalated_clean": sum(escalates(t, False) for t in test_attacks),
            "escalated_off": sum(escalates(t, False) for t in pert_all),
            "escalated_on": sum(escalates(t, True) for t in pert_all),
        })
    out["obfuscated"] = obf
    out["obfuscated_summary"] = {
        "regex_recall_off_pct": round(100 * statistics.mean(o["regex_caught_off"] for o in obf) / len(caught), 1),
        "regex_recall_on_pct": round(100 * statistics.mean(o["regex_caught_on"] for o in obf) / len(caught), 1),
    }
    bench = data["neuralchemy-test"] + data["no_robots-test (benign)"]
    out["latency_ms_per_prompt"] = {"off": round(latency_ms(bench, False), 3), "on": round(latency_ms(bench, True), 3)}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "keywords_escalation"}, indent=1)[:6000])


if __name__ == "__main__":
    main()
