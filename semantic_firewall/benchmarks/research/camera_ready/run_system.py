"""Evaluate one Semantic Firewall configuration on one dataset split.

Examples
  # main result (tau chosen on validation, see tune_threshold.py)
  python run_system.py --config full --dataset neuralchemy --split test --cache warm_frozen --tau 0.90
  # ablation rows
  python run_system.py --config no_cache --split test
  python run_system.py --config fast_layers --split test --cache warm_frozen
  # Reviewer 2 baseline: parallel LLM agents, no deterministic part
  python run_system.py --config llm_agents_only --split test --cache none
  # per-detector leave-one-out
  python run_system.py --config "loo:PII Detector" --split test

Outputs results_camera_ready/runs/<run-name>/{predictions.jsonl, summary.json}.
Runs are resumable: re-running the same command continues where it stopped.
"""

import argparse
import time

from common import CONFIGS, RUNS_DIR, build_firewall, decision_record, load_samples, run_metadata, run_samples, \
    summarize, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--dataset", default="neuralchemy")
    parser.add_argument("--split", default="test")
    parser.add_argument("--cache", default="warm_frozen", choices=["cold", "warm_frozen", "warm_online", "none"],
                        help="semantic-cache protocol (ignored when the config disables the cache)")
    parser.add_argument("--tau", type=float, default=None, help="cache similarity threshold (default: settings value)")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0,
                        help="parallel requests; default 1 (clean latency, ordered write-back) except 4 for "
                             "LLM configs with a frozen/disabled cache, where requests are network-bound")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    config = CONFIGS[args.config]
    overrides = {}
    if args.cache == "none":
        overrides["SEMANTIC_FIREWALL_SEMANTIC_CACHE_ENABLED"] = "0"
    if args.tau is not None:
        overrides["SEMANTIC_FIREWALL_CACHE_SIM_THRESHOLD"] = str(args.tau)
    run_name = args.run_name or f"{args.dataset}-{args.split}-{args.config}-{args.cache}".replace(":", "_").replace(" ", "_")
    out_dir = RUNS_DIR / run_name

    samples = load_samples(args.dataset, args.split, args.max_samples)
    print(f"[{run_name}] {len(samples)} samples, config={config.name}: {config.description}")
    fw, env, seeded = build_firewall(config, run_name, cache_protocol=args.cache, overrides=overrides)
    print(f"  cache enabled={fw.semantic_cache.enabled} seeded={seeded} tau={fw.semantic_cache.similarity_threshold}")

    def predict(sample):
        started = time.perf_counter()
        try:
            from common import quiet_stdout
            with quiet_stdout():
                decision = fw.analyze(sample["text"])
            record = decision_record(sample, decision, (time.perf_counter() - started) * 1000)
            record["error"] = None
        except Exception as exc:  # recorded, never silently counted as benign
            record = {"id": sample["id"], "label": sample["label"], "pred": 0, "action": "ERROR",
                      "latency_ms": (time.perf_counter() - started) * 1000, "error": str(exc)[:300]}
        return record

    online = args.cache in {"cold", "warm_online"} and fw.semantic_cache.enabled
    workers = args.workers or (4 if (config.uses_llm and not online) else 1)
    records = run_samples(predict, samples, out_dir / "predictions.jsonl", workers=workers)

    summary = summarize(records)
    summary["config"] = {"name": config.name, "description": config.description, "uses_llm": config.uses_llm}
    summary["protocol"] = {"dataset": args.dataset, "split": args.split, "cache": args.cache, "seeded_entries": seeded,
                           "cache_size_after": fw.semantic_cache.count(), "cache_stats": fw.semantic_cache.stats,
                           "workers": workers}
    summary["env"] = {k: v for k, v in env.items() if k.startswith("SEMANTIC_FIREWALL_")}
    summary["meta"] = run_metadata()
    write_json(out_dir / "summary.json", summary)
    print(f"  F1={summary['f1']:.4f} P={summary['precision']:.4f} R={summary['recall']:.4f} "
          f"FPR={summary['fpr']:.4f} LLM-call-rate={summary.get('llm_call_rate', 0):.3f} "
          f"p50={summary.get('latency_p50_ms', 0):.1f}ms llm_failed={summary['n_llm_failed']} errors={summary['n_errors']}")


if __name__ == "__main__":
    main()
