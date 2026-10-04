"""Is the warm-cache gain just memorisation of near-duplicates? (Reviewer 3, item 1)

For every test prompt we compute the similarity to its nearest train-split attack (the
warm cache) and report, per similarity bucket, how many test prompts fall there and how
each system performs. We also report every system on the "novel" subset: test prompts
whose nearest train attack is below 0.80 similarity, where the cache cannot fire.

  python cache_overlap.py --runs neuralchemy-test-full-warm_frozen neuralchemy-test-no_cache-none
"""

import argparse

from common import OUT_ROOT, RUNS_DIR, confusion, load_samples, nearest_cache_similarity, rates, read_jsonl, \
    run_metadata, write_json

BUCKETS = [(0.0, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 0.95), (0.95, 1.01)]
NOVEL_BELOW = 0.80


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test")
    parser.add_argument("--runs", nargs="*", default=[])
    args = parser.parse_args()

    samples = load_samples("neuralchemy", args.split)
    sims = dict(zip([s["id"] for s in samples], nearest_cache_similarity([s["text"] for s in samples])))
    labels = {s["id"]: s["label"] for s in samples}

    buckets = []
    for lo, hi in BUCKETS:
        ids = [i for i, v in sims.items() if lo <= v < hi]
        buckets.append({"range": [lo, min(hi, 1.0)], "n": len(ids),
                        "n_attack": sum(labels[i] for i in ids), "n_benign": sum(1 - labels[i] for i in ids)})

    systems = {}
    for run in args.runs:
        preds = {r["id"]: r for r in read_jsonl(RUNS_DIR / run / "predictions.jsonl")}
        per_bucket = []
        for (lo, hi) in BUCKETS:
            recs = [preds[i] for i, v in sims.items() if lo <= v < hi and i in preds]
            per_bucket.append({"range": [lo, min(hi, 1.0)], "n": len(recs), **rates(confusion(recs))})
        novel = [preds[i] for i, v in sims.items() if v < NOVEL_BELOW and i in preds]
        systems[run] = {"per_bucket": per_bucket, "novel_subset": {"n": len(novel), **rates(confusion(novel))},
                        "all": rates(confusion(list(preds.values())))}

    result = {"split": args.split, "buckets": buckets, "novel_threshold": NOVEL_BELOW, "systems": systems,
              "meta": run_metadata()}
    write_json(OUT_ROOT / "cache_overlap" / f"overlap_{args.split}.json", result)
    for b in buckets:
        print(f"  sim {b['range'][0]:.2f}-{b['range'][1]:.2f}: n={b['n']:4d} attack={b['n_attack']:4d} benign={b['n_benign']:4d}")
    for run, s in systems.items():
        nv = s["novel_subset"]
        print(f"  {run}: all F1={s['all']['f1']:.4f}  novel(<{NOVEL_BELOW}) n={nv['n']} F1={nv['f1']:.4f} FPR={nv['fpr']:.4f}")


if __name__ == "__main__":
    main()
