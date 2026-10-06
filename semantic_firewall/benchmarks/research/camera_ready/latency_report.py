"""Latency report and Eq. (effective latency) from measured data.

Inputs
  --latency-run  sequential run (one request at a time) of the full system, e.g. on a low-latency LLM host;
                 the first --warmup records (model loading, connection set-up) are excluded.
  --paths-run    full-size run whose per-prompt resolution path gives the path probabilities P_k
                 (e.g. the derived full-system test run at the chosen tau).
Paths: cache (blocked by semantic memory), deterministic (resolved without the LLM after the detector mesh),
llm (the LLM stage ran). L_eff = sum_k P_k * L_k with L_k the mean latency of path k in the latency run.

  python latency_report.py --latency-run neuralchemy-test-latency-groq \
      --paths-run neuralchemy-test-full_all_unresolved-warm_frozen-tau0.65
"""

import argparse
import random
from collections import defaultdict

from common import OUT_ROOT, RUNS_DIR, percentile, read_jsonl, run_metadata, write_json


def path_of(r):
    if r.get("cache_hit") or r.get("resolution_stage") == "semantic_cache":
        return "cache"
    return "llm" if r.get("llm_called") else "deterministic"


def stats(values):
    if not values:
        return None
    return {"n": len(values), "mean": sum(values) / len(values), "p50": percentile(values, 50),
            "p95": percentile(values, 95), "p99": percentile(values, 99), "min": min(values), "max": max(values)}


def prevalence(lat, paths, L, prevalences=(0.01, 0.05, 0.10), draws=200_000, seed=0):
    """LLM-call rate and latency when a fraction pi of the traffic is attacks.
    Per-class path probabilities come from the paths run, per-path mean latency from the latency run;
    p95/p99 resample the class-conditional latency lists of the latency run."""
    rng = random.Random(seed)
    out = {}
    for pi in prevalences:
        P = defaultdict(float)
        for c, w in ((1, pi), (0, 1 - pi)):
            rs = [r for r in paths if r["label"] == c]
            for r in rs:
                P[path_of(r)] += w / len(rs)
        lists = {c: [r["latency_ms"] for r in lat if r["label"] == c] for c in (0, 1)}
        sample = [rng.choice(lists[1] if rng.random() < pi else lists[0]) for _ in range(draws)]
        out[str(pi)] = {"llm_call_rate": P["llm"], "path_probabilities": dict(P),
                        "effective_latency_ms": sum(P[k] * L[k] for k in P if k in L),
                        "p95_ms": percentile(sample, 95), "p99_ms": percentile(sample, 99)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--latency-run", required=True)
    parser.add_argument("--paths-run", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    lat = read_jsonl(RUNS_DIR / args.latency_run / "predictions.jsonl")[args.warmup:]
    paths = read_jsonl(RUNS_DIR / args.paths_run / "predictions.jsonl")

    by_path = defaultdict(list)
    stage = defaultdict(list)
    for r in lat:
        by_path[path_of(r)].append(r["latency_ms"])
        for k, v in (r.get("stage_latency_ms") or {}).items():
            stage[k].append(v)
    p_counts = defaultdict(int)
    for r in paths:
        p_counts[path_of(r)] += 1
    P = {k: v / len(paths) for k, v in p_counts.items()}
    L = {k: stats(v)["mean"] for k, v in by_path.items() if v}
    l_eff = sum(P[k] * L[k] for k in P if k in L)
    out = {
        "latency_run": args.latency_run, "paths_run": args.paths_run, "warmup_excluded": args.warmup,
        "overall_ms": stats([r["latency_ms"] for r in lat]),
        "by_path_ms": {k: stats(v) for k, v in by_path.items()},
        "by_stage_ms": {k: stats(v) for k, v in stage.items()},
        "path_probabilities": P, "path_mean_latency_ms": L, "effective_latency_ms": l_eff,
        "llm_hosts": sorted({h for r in lat for h in r.get("llm_hosts", [])}),
        "llm_failed": sum(r.get("llm_failed", False) for r in lat), "meta": run_metadata(),
        "prevalence": prevalence(lat, paths, L),
    }
    write_json(OUT_ROOT / "latency" / f"latency_{args.latency_run}.json", out)
    fmt = lambda s: (f"mean {s['mean']:8.1f}  p50 {s['p50']:8.1f}  p95 {s['p95']:8.1f}  p99 {s['p99']:8.1f}  (n={s['n']})"
                     if s else "-")
    print(f"overall         : {fmt(out['overall_ms'])}")
    for k, v in out["by_path_ms"].items():
        print(f"path {k:11}: {fmt(v)}")
    for k, v in out["by_stage_ms"].items():
        print(f"stage {k:10}: {fmt(v)}")
    for pi, v in out["prevalence"].items():
        print(f"attack rate {float(pi):4.0%}: LLM calls {v['llm_call_rate']:.3f}  L_eff {v['effective_latency_ms']:.1f} ms"
              f"  p95 {v['p95_ms']:.1f}  p99 {v['p99_ms']:.1f}")
    print("P_k:", {k: round(v, 4) for k, v in P.items()}, "| L_eff = %.1f ms" % l_eff, "| hosts:", out["llm_hosts"],
          "| llm_failed:", out["llm_failed"])


if __name__ == "__main__":
    main()
