"""Select the cache threshold tau and the ensemble threshold theta on the validation split.

With a frozen, train-seeded cache, the decision for each prompt is
    pred(tau, theta) = 1                         if sim(prompt, nearest train attack) >= tau
                       downstream_pred(theta)    otherwise
where downstream_pred is the pipeline's prediction with the cache disabled. The downstream run records,
per prompt, the policy action before ensemble escalation and the ensemble score E (Eq. eq:ensemble). For the
balanced profile without sessions the final action is the stricter of the two, so
    downstream_pred(theta) = 1  if policy_action in {FLAG, REDACT, BLOCK} or E >= FLAG_RATIO * theta
with theta the BLOCK threshold and FLAG/REDACT scaled with it (default 1.8 / 2.6 / 3.5). The sweep is
therefore exact; it is checked by recomputing the recorded predictions at the default theta.
One downstream run per split is enough.

  python run_system.py --config no_cache --split validation --cache none
  python tune_threshold.py --split validation --downstream-run neuralchemy-validation-no_cache-none

  # OOD check (reported, never used for selection)
  python run_system.py --config no_cache --dataset <set> --split <split> --cache none
  python tune_threshold.py --dataset <set> --split <split> --downstream-run <that run>

Then evaluate once on test with the selected values: run_system.py --tau <tau*> --theta <theta*>.
"""

import argparse
import csv

from common import OUT_ROOT, RUNS_DIR, load_samples, nearest_cache_similarity, rates, confusion, read_jsonl, \
    run_metadata, write_json

TAU_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.97, 0.99]
THETA_GRID = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
DEFAULT_THETA = 3.5
FLAG_RATIO = 1.8 / 3.5
POSITIVE = {"FLAG", "REDACT", "BLOCK"}


def downstream_pred(rec, theta):
    if rec.get("policy_action") is None or rec.get("ensemble_risk") is None:
        return rec["pred"]  # resolved before the detector stage (or an old run without E): theta has no effect
    return int(rec["policy_action"] in POSITIVE or rec["ensemble_risk"] >= FLAG_RATIO * theta)


def check_exact(samples, records):
    """At the default theta the recomputed prediction must equal the recorded one for every prompt."""
    have_e = [records[s["id"]] for s in samples if records[s["id"]].get("ensemble_risk") is not None]
    bad = [r["id"] for r in have_e if downstream_pred(r, DEFAULT_THETA) != r["pred"]]
    if bad:
        raise SystemExit(f"theta recomputation does not reproduce {len(bad)} recorded predictions, e.g. {bad[:3]}")
    return len(have_e)


def evaluate(samples, sims, records, tau, theta):
    recs, hits, hit_attacks = [], 0, 0
    for s, sim in zip(samples, sims):
        hit = tau is not None and sim >= tau
        hits += hit
        hit_attacks += hit and s["label"] == 1
        recs.append({"label": s["label"], "pred": 1 if hit else downstream_pred(records[s["id"]], theta)})
    return {"tau": "no cache" if tau is None else tau, "theta": theta, "hit_rate": hits / len(samples),
            "hit_precision": hit_attacks / hits if hits else None, **rates(confusion(recs))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="neuralchemy", help="neuralchemy (tuning) or another set (OOD check)")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--downstream-run", required=True, help="run directory name under results_camera_ready/runs")
    args = parser.parse_args()

    samples = load_samples(args.dataset, args.split)
    records = {r["id"]: r for r in read_jsonl(RUNS_DIR / args.downstream_run / "predictions.jsonl")}
    missing = [s["id"] for s in samples if s["id"] not in records]
    if missing:
        raise SystemExit(f"{len(missing)} samples have no downstream prediction; finish that run first")
    n_exact = check_exact(samples, records)
    thetas = THETA_GRID if n_exact else [DEFAULT_THETA]

    sims = nearest_cache_similarity([s["text"] for s in samples])
    tau_rows = [evaluate(samples, sims, records, None, DEFAULT_THETA)] + \
               [evaluate(samples, sims, records, tau, DEFAULT_THETA) for tau in TAU_GRID]
    grid = [evaluate(samples, sims, records, tau, theta) for theta in thetas for tau in TAU_GRID]
    # ties -> larger (more conservative) tau, then the default theta, then larger theta
    best = max(grid, key=lambda r: (round(r["f1"], 6), r["tau"], r["theta"] == DEFAULT_THETA, r["theta"]))
    tuning = args.dataset == "neuralchemy" and args.split == "validation"

    out = OUT_ROOT / "threshold"
    stem = f"tau_sweep_{args.dataset}_{args.split}_{args.downstream_run}"
    write_json(out / f"{stem}.json", {
        "dataset": args.dataset, "split": args.split, "downstream_run": args.downstream_run,
        "theta_exact_check": {"records_with_E": n_exact, "reproduced_at_default_theta": True},
        "tau_rows_at_default_theta": tau_rows, "joint_grid": grid,
        "selected": {"tau": best["tau"], "theta": best["theta"], "f1": best["f1"]} if tuning else None,
        "selection_rule": "argmax F1 on neuralchemy validation; ties -> larger tau, then default theta",
        "meta": run_metadata(),
    })
    with (out / f"{stem}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tau", "theta", "precision", "recall", "f1", "fpr", "cache_hit_rate", "hit_precision"])
        for r in tau_rows + grid:
            writer.writerow([r["tau"], r["theta"], f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}",
                             f"{r['fpr']:.4f}", f"{r['hit_rate']:.4f}",
                             "" if r["hit_precision"] is None else f"{r['hit_precision']:.4f}"])
    print(f"theta check: {n_exact} records with E reproduced exactly at theta={DEFAULT_THETA}")
    for r in tau_rows:
        print(f"  tau={r['tau']!s:>8}  P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} "
              f"FPR={r['fpr']:.4f} hit={r['hit_rate']:.3f}")
    if len(thetas) > 1:
        print("  best F1 per theta:", {t: round(max(r["f1"] for r in grid if r["theta"] == t), 4) for t in thetas})
    label = "selected" if tuning else "(information only, not a tuning split) best"
    print(f"{label}: tau={best['tau']} theta={best['theta']} F1={best['f1']:.4f}")


if __name__ == "__main__":
    main()
