"""Select the semantic-cache threshold tau on the validation split (Reviewer 1, item 2).

With a frozen, train-seeded cache the decision for each prompt is
    pred(tau) = 1                      if sim(prompt, nearest train attack) >= tau
                downstream_pred        otherwise
where downstream_pred is the pipeline's prediction with the cache disabled. The sweep is
therefore exact and needs only one downstream run per split.

  # downstream predictions without the cache (LLM pipeline):
  python run_system.py --config no_cache --split validation --cache none
  python tune_threshold.py --split validation --downstream-run neuralchemy-validation-no_cache-none

  # offline variant with no API (deterministic detectors as the downstream stage):
  python run_system.py --config deterministic_only --split validation --cache none
  python tune_threshold.py --split validation --downstream-run neuralchemy-validation-deterministic_only-none

Then evaluate once on test with the selected tau (run_system.py --tau <tau*>).
"""

import argparse
import csv

from common import OUT_ROOT, RUNS_DIR, load_samples, nearest_cache_similarity, rates, confusion, read_jsonl, \
    run_metadata, write_json

TAU_GRID = [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.97, 0.99]


def sweep(samples, sims, downstream):
    rows = []
    base = [{"label": s["label"], "pred": downstream[s["id"]]} for s in samples]
    rows.append({"tau": "no cache", "hit_rate": 0.0, "hit_precision": None, **rates(confusion(base))})
    for tau in TAU_GRID:
        recs, hits, hit_attacks = [], 0, 0
        for s, sim in zip(samples, sims):
            hit = sim >= tau
            hits += hit
            hit_attacks += hit and s["label"] == 1
            recs.append({"label": s["label"], "pred": 1 if hit else downstream[s["id"]]})
        rows.append({"tau": tau, "hit_rate": hits / len(samples),
                     "hit_precision": hit_attacks / hits if hits else None, **rates(confusion(recs))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--downstream-run", required=True, help="run directory name under results_camera_ready/runs")
    args = parser.parse_args()

    samples = load_samples("neuralchemy", args.split)
    preds = {r["id"]: r["pred"] for r in read_jsonl(RUNS_DIR / args.downstream_run / "predictions.jsonl")}
    missing = [s["id"] for s in samples if s["id"] not in preds]
    if missing:
        raise SystemExit(f"{len(missing)} samples have no downstream prediction; finish that run first")

    sims = nearest_cache_similarity([s["text"] for s in samples])
    rows = sweep(samples, sims, preds)
    candidates = [r for r in rows if r["tau"] != "no cache"]
    best = max(candidates, key=lambda r: (round(r["f1"], 6), r["tau"]))  # ties -> larger (more conservative) tau

    out = OUT_ROOT / "threshold"
    write_json(out / f"tau_sweep_{args.split}_{args.downstream_run}.json", {
        "split": args.split, "downstream_run": args.downstream_run, "rows": rows,
        "selected_tau": best["tau"] if args.split == "validation" else None,
        "selection_rule": "argmax F1 on validation; ties broken towards the larger tau",
        "meta": run_metadata(),
    })
    with (out / f"tau_sweep_{args.split}_{args.downstream_run}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tau", "precision", "recall", "f1", "fpr", "cache_hit_rate", "hit_precision"])
        for r in rows:
            writer.writerow([r["tau"], f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}",
                             f"{r['fpr']:.4f}", f"{r['hit_rate']:.4f}",
                             "" if r["hit_precision"] is None else f"{r['hit_precision']:.4f}"])
    for r in rows:
        print(f"  tau={r['tau']!s:>8}  P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} "
              f"FPR={r['fpr']:.4f} hit={r['hit_rate']:.3f}")
    if args.split == "validation":
        print(f"selected tau = {best['tau']} (F1={best['f1']:.4f})")


if __name__ == "__main__":
    main()
