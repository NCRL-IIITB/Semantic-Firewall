"""Paired significance tests between systems run on the same samples (Reviewer 1, item 2).

McNemar's test on the per-sample correctness of the reference system vs each comparator
(exact binomial when there are fewer than 25 discordant pairs, otherwise chi-square with
continuity correction), Holm-Bonferroni correction across all comparisons, and a paired
bootstrap 95% CI for the F1 and FPR differences.

  python significance.py --reference neuralchemy-test-full-warm_frozen \
      --compare neuralchemy-test-baseline-llm-openai__gpt-4o-2024-08-06 neuralchemy-test-llm_agents_only-none
"""

import argparse
import math
import random

from common import OUT_ROOT, RUNS_DIR, confusion, percentile, rates, read_jsonl, run_metadata, write_json


def mcnemar(b: int, c: int) -> tuple[float, float, str]:
    """b = reference right & other wrong, c = reference wrong & other right."""
    n = b + c
    if n == 0:
        return 0.0, 1.0, "no discordant pairs"
    if n < 25:
        k = min(b, c)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
        return float("nan"), p, "exact binomial"
    chi2 = (abs(b - c) - 1) ** 2 / n
    p = math.erfc(math.sqrt(chi2 / 2))  # survival function of chi-square with 1 dof
    return chi2, p, "chi-square, continuity-corrected"


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def paired_bootstrap(ref, other, n=2000, seed=0):
    rng = random.Random(seed)
    d_f1, d_fpr = [], []
    idx = list(range(len(ref)))
    for _ in range(n):
        pick = [idx[rng.randrange(len(idx))] for _ in idx]
        a = rates(confusion([ref[i] for i in pick]))
        b = rates(confusion([other[i] for i in pick]))
        d_f1.append(a["f1"] - b["f1"])
        d_fpr.append(a["fpr"] - b["fpr"])
    return [percentile(d_f1, 2.5), percentile(d_f1, 97.5)], [percentile(d_fpr, 2.5), percentile(d_fpr, 97.5)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--compare", nargs="+", required=True)
    parser.add_argument("--name", default="significance")
    args = parser.parse_args()

    ref_all = {r["id"]: r for r in read_jsonl(RUNS_DIR / args.reference / "predictions.jsonl")}
    rows = []
    for run in args.compare:
        other_all = {r["id"]: r for r in read_jsonl(RUNS_DIR / run / "predictions.jsonl")}
        common_ids = sorted(set(ref_all) & set(other_all))
        ref = [ref_all[i] for i in common_ids]
        other = [other_all[i] for i in common_ids]
        b = sum((r["pred"] == r["label"]) and (o["pred"] != o["label"]) for r, o in zip(ref, other))
        c = sum((r["pred"] != r["label"]) and (o["pred"] == o["label"]) for r, o in zip(ref, other))
        chi2, p, test = mcnemar(b, c)
        ci_f1, ci_fpr = paired_bootstrap(ref, other)
        rows.append({"compare": run, "n_common": len(common_ids), "ref_only_correct": b, "other_only_correct": c,
                     "chi2": chi2, "p": p, "test": test, "ref": rates(confusion(ref)), "other": rates(confusion(other)),
                     "delta_f1_ci95": ci_f1, "delta_fpr_ci95": ci_fpr})
    for row, p_adj in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = p_adj

    write_json(OUT_ROOT / "significance" / f"{args.name}.json",
               {"reference": args.reference, "comparisons": rows, "meta": run_metadata()})
    for r in rows:
        print(f"  vs {r['compare']}: n={r['n_common']} b={r['ref_only_correct']} c={r['other_only_correct']} "
              f"p={r['p']:.4g} p_holm={r['p_holm']:.4g} dF1 CI={r['delta_f1_ci95'][0]:+.4f}..{r['delta_f1_ci95'][1]:+.4f}")


if __name__ == "__main__":
    main()
