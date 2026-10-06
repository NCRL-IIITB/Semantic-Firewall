"""Build every number of the result tables from raw per-sample records (no hand-typed values).

Full system (warm, frozen cache seeded with neuralchemy train attacks) at threshold tau is computed exactly from
the no-cache run of the same pipeline:  pred = 1 if sim(prompt, nearest train attack) >= tau else no-cache pred.
A cache hit also means the LLM is not called for that prompt.

  python final_tables.py --tau 0.65
Output: results_camera_ready/final_tables.json (+ printed tables)
"""

import argparse
import json

from common import OUT_ROOT, RUNS_DIR, load_samples, nearest_cache_similarity, read_jsonl, run_metadata, summarize, \
    write_json

SYSTEM_CONFIG = "full_all_unresolved"
DATASETS = {  # key: (dataset, split, max_samples)
    "neuralchemy-test": ("neuralchemy", "test", 0),
    "ood500": ("ood500", "all", 0),
    "deepset": ("deepset", "all", 0),
    "beavertails": ("beavertails", "test", 1000),
}
BASELINES = {
    "neuralchemy-test": ["baseline-llm-openai__gpt-4o", "baseline-llm-google__gemini-2.5-flash",
                         "baseline-llm-anthropic__claude-haiku-4.5", "baseline-llama_guard-meta-llama__llama-guard-4-12b",
                         "baseline-hf-protectai__deberta-v3-base-prompt-injection-v2", "baseline-hf-leolee99__PIGuard"],
    "deepset": ["baseline-hf-protectai__deberta-v3-base-prompt-injection-v2"],
    "beavertails": ["baseline-llama_guard-meta-llama__llama-guard-4-12b"],
}


def run_dir(dataset, split, name):
    return RUNS_DIR / f"{dataset}-{split}-{name}"


def system_at_tau(key, tau):
    dataset, split, max_samples = DATASETS[key]
    path = run_dir(dataset, split, f"{SYSTEM_CONFIG}-none") / "predictions.jsonl"
    if not path.exists():
        return None
    samples = load_samples(dataset, split, max_samples=max_samples)
    recs = {r["id"]: r for r in read_jsonl(path)}
    missing = [s["id"] for s in samples if s["id"] not in recs]
    if missing:
        return {"incomplete": f"{len(recs)}/{len(samples)}"}
    sims = nearest_cache_similarity([s["text"] for s in samples])
    out_nocache, out_full, hits, llm = [], [], 0, 0
    for s, sim in zip(samples, sims):
        r = recs[s["id"]]
        out_nocache.append(r)
        hit = sim >= tau
        hits += hit
        llm += (not hit) and r.get("llm_called", False)
        out_full.append({**r, "pred": 1 if hit else r["pred"]})
    full = summarize(out_full)
    full.update({"tau": tau, "cache_hit_rate": hits / len(samples), "llm_call_rate": llm / len(samples),
                 "llm_bypass_rate": 1 - llm / len(samples)})
    return {"full_warm": full, "no_cache": summarize(out_nocache)}


def baseline(key, name):
    dataset, split, _ = DATASETS[key]
    path = run_dir(dataset, split, name) / "predictions.jsonl"
    return summarize(read_jsonl(path)) if path.exists() else None


def fmt(m):
    if not m or "incomplete" in m:
        return f"(incomplete {m.get('incomplete') if m else ''})"
    ci = m.get("f1_ci95")
    pm = f" +/-{(ci[1] - ci[0]) * 50:.2f}" if ci else ""
    return (f"P {m['precision'] * 100:6.2f}  R {m['recall'] * 100:6.2f}  F1 {m['f1'] * 100:6.2f}{pm}  "
            f"FPR {m['fpr'] * 100:5.2f}%  n={m['n']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tau", type=float, required=True)
    args = parser.parse_args()

    tables = {}
    for key in DATASETS:
        sysres = system_at_tau(key, args.tau)
        tables[key] = {"system": sysres, "baselines": {b: baseline(key, b) for b in BASELINES.get(key, [])}}
        print(f"\n=== {key}")
        if sysres and "full_warm" in sysres:
            f = sysres["full_warm"]
            print(f"  Semantic Firewall (warm, tau={args.tau}): {fmt(f)} | LLM skipped {f['llm_bypass_rate'] * 100:.1f}%"
                  f" (cache hits {f['cache_hit_rate'] * 100:.1f}%)")
            print(f"  Semantic Firewall (no cache)          : {fmt(sysres['no_cache'])}")
        else:
            print("  Semantic Firewall:", fmt(sysres))
        for b, m in tables[key]["baselines"].items():
            print(f"  {b[9:]:58}: {fmt(m)}")
    write_json(OUT_ROOT / "final_tables.json", {"tau": args.tau, "tables": tables, "meta": run_metadata()})


if __name__ == "__main__":
    main()
