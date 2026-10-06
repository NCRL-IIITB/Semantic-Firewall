"""Warm-cache run at a fixed tau, derived exactly from a no-cache run (Table 2, Table 4 C8, Table 3).

With the cache seeded only with train-split attacks and frozen, a warm-cache prompt is blocked by the cache
when its nearest train attack has cosine similarity >= tau, and otherwise follows exactly the same path as in
the no-cache run. So
    pred_warm(tau) = 1 if sim(prompt, nearest train attack) >= tau else pred_no_cache
Each derived record is the no-cache record plus `semantic_similarity` and `cache_hit`.

  python derive_warm.py --source neuralchemy-test-full_all_unresolved-none --dataset neuralchemy --split test
"""

import argparse

from common import RUNS_DIR, load_samples, nearest_cache_similarity, read_jsonl, run_metadata, summarize, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="no-cache run name (ends in -none)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--tau", type=float, default=0.65)
    parser.add_argument("--out", default="", help="output run name (default: <source without -none>-warm_frozen-tau<tau>)")
    args = parser.parse_args()

    records = read_jsonl(RUNS_DIR / args.source / "predictions.jsonl")
    samples = {s["id"]: s for s in load_samples(args.dataset, args.split)}
    missing = [r["id"] for r in records if r["id"] not in samples]
    if missing:
        raise SystemExit(f"{len(missing)} record ids not in {args.dataset}/{args.split}")
    sims = nearest_cache_similarity([samples[r["id"]]["text"] for r in records])

    derived = []
    for r, sim in zip(records, sims):
        hit = sim >= args.tau
        d = dict(r)
        d["semantic_similarity"] = round(sim, 4)
        d["cache_hit"] = hit
        if hit:
            d["pred"] = 1
        derived.append(d)

    name = args.out or f"{args.source.removesuffix('-none')}-warm_frozen-tau{args.tau}"
    out_dir = RUNS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for d in derived:
            f.write(__import__("json").dumps(d, ensure_ascii=False) + "\n")
    summary = summarize(derived, with_ci=False)
    summary.update({"derived_from": args.source, "tau": args.tau,
                    "rule": "pred = 1 if sim(prompt, nearest train attack) >= tau else no-cache pred",
                    "meta": run_metadata()})
    write_json(out_dir / "summary.json", summary)
    print(f"{name}: F1 {summary['f1']:.4f}  P {summary['precision']:.4f}  R {summary['recall']:.4f}  "
          f"FPR {summary['fpr']:.4f}  cache hits {sum(d['cache_hit'] for d in derived)}/{len(derived)}")


if __name__ == "__main__":
    main()
