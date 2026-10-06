"""Build the OOD-500 evaluation set (Table 3b): 250 attacks + 250 benign prompts from public sources that
neuralchemy/Prompt-injection-dataset was NOT built from (it uses HackAPrompt, WildGuard, HarmBench and its own data).

Attacks   Lakera/gandalf_ignore_instructions (MIT)                       100  direct prompt injections
          TrustAIRLab/in-the-wild-jailbreak-prompts, 2023-12-25 (MIT)     100  in-the-wild jailbreaks
          reshabhs/SPML_Chatbot_Prompt_Injection, injection = 1 (MIT)      50  injections against chatbots
Benign    OpenAssistant/oasst2, first English user turns (Apache-2.0)      100  ordinary requests
          reshabhs/SPML_Chatbot_Prompt_Injection, injection = 0 (MIT)      75  ordinary chatbot messages
          Paul/XSTest, label = safe (CC-BY-4.0)                            75  hard negatives ("kill a Python process")

Filters: 20-4,000 characters; mostly-ASCII text (English); exact and near duplicates removed (cosine >= 0.95
within the set); OOD: any prompt with cosine >= 0.90 to a neuralchemy (core) train/validation/test prompt is
removed. Sampling is seeded; counts of removed items are written to the metadata.

Output: data/datasets/ood_curated_500.jsonl (+ ood_curated_500.meta.json)
  python build_ood_set.py
"""

import json
import random
import re

from common import PROJECT_ROOT, _load, quiet_stdout, write_json

SEED = 0
OOD_MAX_SIM = 0.90
DUP_SIM = 0.95
OUT = PROJECT_ROOT / "data" / "datasets" / "ood_curated_500.jsonl"
PLAN = [  # (source key, label, count)
    ("gandalf", 1, 100), ("in_the_wild_jailbreak", 1, 100), ("spml_injection", 1, 50),
    ("oasst2", 0, 100), ("spml_benign", 0, 75), ("xstest_safe", 0, 75),
]


def _clean(t):
    return re.sub(r"\s+", " ", str(t or "")).strip()


def _english(t):
    return sum(c.isascii() for c in t) / max(len(t), 1) >= 0.97


def candidates():
    c = {}
    g = _load("Lakera/gandalf_ignore_instructions")
    c["gandalf"] = [r["text"] for sp in ("train", "validation", "test") for r in g[sp]]
    w = _load("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25")["train"]
    c["in_the_wild_jailbreak"] = [r["prompt"] for r in w if str(r["jailbreak"]).lower() == "true"]
    s = _load("reshabhs/SPML_Chatbot_Prompt_Injection")["train"]
    c["spml_injection"] = [r["User Prompt"] for r in s if str(r["Prompt injection"]) == "1"]
    c["spml_benign"] = [r["User Prompt"] for r in s if str(r["Prompt injection"]) == "0"]
    o = _load("OpenAssistant/oasst2")
    c["oasst2"] = [r["text"] for sp in ("train", "validation") for r in o[sp]
                   if r["role"] == "prompter" and r["parent_id"] in (None, "None") and r["lang"] == "en"
                   and str(r["deleted"]).lower() != "true"]
    x = _load("Paul/XSTest")["train"]
    c["xstest_safe"] = [r["prompt"] for r in x if r["label"] == "safe"]
    return {k: [t for t in map(_clean, v) if 20 <= len(t) <= 4000 and _english(t)] for k, v in c.items()}


def main():
    from common import load_samples
    from semantic_firewall.core.orchestrator.semantic_cache import shared_embedder
    import numpy as np

    rng = random.Random(SEED)
    pools = candidates()
    with quiet_stdout():
        embed = shared_embedder()

    def vecs(texts):
        v = np.asarray(embed(texts), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    neural = [r["text"] for sp in ("train", "validation", "test") for r in load_samples("neuralchemy", sp)]
    neural_v = vecs(neural)

    rows, removed, chosen_v = [], {}, np.zeros((0, neural_v.shape[1]), dtype=np.float32)
    for key, label, count in PLAN:
        pool = sorted(set(pools[key]))
        rng.shuffle(pool)
        stats = {"pool": len(pool), "ood_removed": 0, "dup_removed": 0}
        picked = []
        for start in range(0, len(pool), 256):
            batch = pool[start:start + 256]
            bv = vecs(batch)
            near_neural = (bv @ neural_v.T).max(axis=1)
            for t, v, sim in zip(batch, bv, near_neural):
                if len(picked) == count:
                    break
                if sim >= OOD_MAX_SIM:
                    stats["ood_removed"] += 1
                    continue
                if chosen_v.shape[0] and float((chosen_v @ v).max()) >= DUP_SIM:
                    stats["dup_removed"] += 1
                    continue
                picked.append((t, float(sim)))
                chosen_v = np.vstack([chosen_v, v])
            if len(picked) == count:
                break
        if len(picked) < count:
            raise SystemExit(f"{key}: only {len(picked)} of {count} prompts left after filtering")
        stats["selected"] = count
        removed[key] = stats
        rows += [{"id": f"ood-{key}-{i:03d}", "text": t, "label": label, "source": key,
                  "max_sim_to_neuralchemy": round(sim, 4)} for i, (t, sim) in enumerate(picked)]

    rng.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(OUT.with_suffix(".meta.json"), {
        "seed": SEED, "ood_max_similarity_to_neuralchemy": OOD_MAX_SIM, "near_duplicate_similarity": DUP_SIM,
        "plan": [{"source": k, "label": l, "count": n} for k, l, n in PLAN], "filtering": removed,
        "n": len(rows), "attacks": sum(r["label"] for r in rows), "benign": sum(1 - r["label"] for r in rows),
    })
    print(json.dumps(removed, indent=1))
    print(f"wrote {len(rows)} prompts to {OUT}")


if __name__ == "__main__":
    main()
