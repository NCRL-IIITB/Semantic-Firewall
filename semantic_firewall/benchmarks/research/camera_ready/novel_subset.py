"""Warm-cache memorisation check.

Buckets the test prompts by the similarity of their nearest train-split attack (the frozen warm cache),
as recorded in the full-system run (`semantic_similarity`), and reports every system per bucket. The
bucket below tau is the "novel" subset on which the cache cannot fire.

  python novel_subset.py --reference neuralchemy-test-full_all_unresolved-warm_frozen-tau0.65 \
      --compare neuralchemy-test-full_all_unresolved-none neuralchemy-test-baseline-llm-openai__gpt-4o
"""

import argparse

from common import OUT_ROOT, RUNS_DIR, read_jsonl, run_metadata, write_json


def metrics(ids, preds, labels):
    tp = sum(preds[i] & labels[i] for i in ids)
    fp = sum(preds[i] & (1 - labels[i]) for i in ids)
    fn = sum((1 - preds[i]) & labels[i] for i in ids)
    tn = len(ids) - tp - fp - fn
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(ids), "attacks": tp + fn, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0, "fpr": fp / (fp + tn) if fp + tn else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--compare", nargs="*", default=[])
    parser.add_argument("--tau", type=float, default=0.65)
    args = parser.parse_args()

    ref = {r["id"]: r for r in read_jsonl(RUNS_DIR / args.reference / "predictions.jsonl")}
    labels = {i: r["label"] for i, r in ref.items()}
    sims = {i: r.get("semantic_similarity") or 0.0 for i, r in ref.items()}
    systems = {args.reference: {i: r["pred"] for i, r in ref.items()}}
    for name in args.compare:
        systems[name] = {r["id"]: r["pred"] for r in read_jsonl(RUNS_DIR / name / "predictions.jsonl")}
    buckets = [(0.0, args.tau), (args.tau, 0.80), (0.80, 0.90), (0.90, 1.01)]
    out = {"reference": args.reference, "tau": args.tau, "buckets": [], "meta": run_metadata()}
    for lo, hi in buckets:
        ids = [i for i in ref if lo <= sims[i] < hi]
        row = {"range": [lo, hi], "systems": {name: metrics(ids, preds, labels) for name, preds in systems.items()}}
        out["buckets"].append(row)
        print(f"[{lo:.2f}, {hi:.2f}) n={len(ids)}")
        for name, m in row["systems"].items():
            print(f"  {name:75} F1 {100*m['f1']:5.1f}  R {100*m['recall']:5.1f}  FPR {100*m['fpr']:5.1f}")
    write_json(OUT_ROOT / "significance" / "novel_subset.json", out)


if __name__ == "__main__":
    main()
