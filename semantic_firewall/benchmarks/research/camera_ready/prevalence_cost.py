"""LLM-call rate, cost and tail latency at realistic attack prevalence (Reviewer 1, item 3).

Uses the per-sample records of a firewall run (resolution stage, llm_called, latency) split
by class, then re-weights to an attack prevalence pi:
    E[LLM calls] = pi * r_attack + (1 - pi) * r_benign
Tail latencies (p95/p99) are estimated by resampling the class-conditional latency lists.
Path probabilities and mean path latencies for Eq. (7) are reported from the same records.

  python prevalence_cost.py --run neuralchemy-test-full-warm_frozen --price-per-mtok 0.13 --calls-per-llm-stage 2
"""

import argparse
import random

from common import OUT_ROOT, RUNS_DIR, percentile, read_jsonl, run_metadata, write_json

PREVALENCES = [0.01, 0.05, 0.10, 0.25, 0.50]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--price-per-mtok", type=float, required=True,
                        help="USD per 1M input tokens of the gate model on your provider (state it in the paper)")
    parser.add_argument("--calls-per-llm-stage", type=int, default=2,
                        help="LLM requests per invocation of the LLM stage (injection + unsafe-content agents = 2)")
    parser.add_argument("--prompt-overhead-tokens", type=int, default=600,
                        help="system-prompt tokens per LLM request (measure with the provider's tokenizer)")
    parser.add_argument("--requests", type=int, default=1_000_000)
    args = parser.parse_args()

    recs = [r for r in read_jsonl(RUNS_DIR / args.run / "predictions.jsonl") if r.get("error") is None]
    by_class = {c: [r for r in recs if r["label"] == c] for c in (0, 1)}
    if not by_class[0] or not by_class[1]:
        raise SystemExit("need both benign and attack samples")

    def llm_rate(rs):
        return sum(r.get("llm_called", False) for r in rs) / len(rs)

    avg_tokens = sum(r.get("text_chars", 0) for r in recs) / len(recs) / 4 + args.prompt_overhead_tokens
    stages = sorted({r["resolution_stage"] for r in recs})
    path = {}
    for s in stages:
        rs = [r for r in recs if r["resolution_stage"] == s]
        path[s] = {"probability": len(rs) / len(recs), "mean_latency_ms": sum(r["latency_ms"] for r in rs) / len(rs)}

    rng = random.Random(0)
    rows = []
    for pi in PREVALENCES:
        rate = pi * llm_rate(by_class[1]) + (1 - pi) * llm_rate(by_class[0])
        mean_lat = pi * (sum(r["latency_ms"] for r in by_class[1]) / len(by_class[1])) + \
            (1 - pi) * (sum(r["latency_ms"] for r in by_class[0]) / len(by_class[0]))
        draws = [rng.choice(by_class[1] if rng.random() < pi else by_class[0])["latency_ms"] for _ in range(100_000)]
        cost = args.requests * rate * args.calls_per_llm_stage * avg_tokens / 1e6 * args.price_per_mtok
        always = args.requests * args.calls_per_llm_stage * avg_tokens / 1e6 * args.price_per_mtok
        rows.append({"prevalence": pi, "llm_call_rate": rate, "mean_latency_ms": mean_lat,
                     "p95_latency_ms": percentile(draws, 95), "p99_latency_ms": percentile(draws, 99),
                     "gate_cost_usd": cost, "always_llm_cost_usd": always,
                     "saving_vs_always_llm": 1 - cost / always if always else 0.0})

    result = {"run": args.run, "llm_call_rate_attack": llm_rate(by_class[1]), "llm_call_rate_benign": llm_rate(by_class[0]),
              "eq7_paths": path, "assumptions": vars(args) | {"avg_tokens_per_request": avg_tokens},
              "rows": rows, "meta": run_metadata()}
    write_json(OUT_ROOT / "cost" / f"prevalence_{args.run}.json", result)
    print(f"  LLM call rate: attacks={result['llm_call_rate_attack']:.3f} benign={result['llm_call_rate_benign']:.3f}")
    for s, v in path.items():
        print(f"  path {s:15s} P={v['probability']:.3f} L={v['mean_latency_ms']:.1f} ms")
    for r in rows:
        print(f"  pi={r['prevalence']:.2f} llm_rate={r['llm_call_rate']:.3f} mean={r['mean_latency_ms']:.0f}ms "
              f"p95={r['p95_latency_ms']:.0f} p99={r['p99_latency_ms']:.0f} saving={r['saving_vs_always_llm']:.1%}")


if __name__ == "__main__":
    main()
