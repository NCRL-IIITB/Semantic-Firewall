"""
Experiment 19: Gate Swap Comparison (Architecture-Agnostic Proof)
=================================================================
Keeps the ENTIRE Semantic Firewall architecture (Regex, Semantic Cache,
Heuristic Agents) but SWAPS the final LLM Gate to different commercial
models via OpenRouter.

This proves that the architecture itself adds value regardless of the
underlying LLM. If "Firewall + GPT-4o-mini" beats standalone GPT-4o-mini,
the architecture is the innovation.

Models tested as the LLM Gate:
  1. openai/gpt-4o-mini
  2. anthropic/claude-3.5-haiku
  3. google/gemini-2.0-flash-lite
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from semantic_firewall.sdk import Firewall
from semantic_firewall.benchmarks.research.experiments.helpers import (
    RESULTS_ROOT, TABLES_DIR,
    compute_metrics, format_pct, write_csv_rows, write_json,
    normalize_binary_label,
    print_header, print_section,
)

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

# The LLM gate models to test via OpenRouter
GATE_MODELS = [
    ("openai/gpt-4o-mini", "Firewall + GPT-4o-mini Gate"),
    ("anthropic/claude-3.5-haiku", "Firewall + Claude 3.5 Haiku Gate"),
    ("google/gemini-2.0-flash-lite", "Firewall + Gemini Flash Lite Gate"),
]


def fresh_firewall_with_gate(gate_model: str):
    """Create a fresh Firewall instance with a specific LLM Gate model."""
    from semantic_firewall.core.orchestrator.semantic_cache import SemanticCache
    import threading
    SemanticCache._instance = None
    SemanticCache._lock = threading.Lock()

    # Set the LLM gate model via environment variable
    os.environ["SEMANTIC_FIREWALL_DISABLE_LLM_DETECTORS"] = "0"
    os.environ["SEMANTIC_FIREWALL_LLM_GATE_ENABLED"] = "1"
    os.environ["SEMANTIC_FIREWALL_LLM_GATE_THRESHOLD"] = "1.0"
    os.environ["SEMANTIC_FIREWALL_SIMILAR_CACHE_ENABLED"] = "1"
    os.environ["SEMANTIC_FIREWALL_ENSEMBLE_ENABLED"] = "1"
    # Override the model used for the LLM gate (read by DetectorLLMClient).
    os.environ["SEMANTIC_FIREWALL_LLM_MODEL"] = gate_model

    fw = Firewall()
    # Clear cache for fair comparison
    if fw._local and hasattr(fw._local, '_cache'):
        fw._local._cache.clear()
    return fw


def evaluate_single(fw, text, true_label):
    """Evaluate a single sample through the full firewall pipeline."""
    try:
        start_t = time.perf_counter()
        decision = fw.analyze(text)
        lat = (time.perf_counter() - start_t) * 1000

        if "Detector unavailable" in str(decision.reason):
            return true_label, 0, -1

        predicted = 1 if decision.action in {"BLOCK", "FLAG", "REDACT"} else 0
        return true_label, predicted, lat
    except Exception as e:
        err = str(e)
        if '429' in err or 'rate' in err.lower() or 'unavailable' in err.lower() or 'timeout' in err.lower():
            return true_label, 0, -1
        return true_label, 0, -1


def run_gate_benchmark(gate_model: str, label: str, test_set: list, workers: int = 5):
    """Run the full Semantic Firewall with a specific LLM gate on a dataset."""
    print_section(f"Evaluating: {label} ({gate_model})")

    fw = fresh_firewall_with_gate(gate_model)

    tp = fp = tn = fn = 0
    latencies = []
    llm_invoked = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(evaluate_single, fw, ex["text"], normalize_binary_label(ex["label"])): ex
            for ex in test_set
        }
        for i, future in enumerate(as_completed(futures)):
            t_label, p_label, lat = future.result()

            if lat == -1:
                continue

            latencies.append(lat)
            if t_label == 1 and p_label == 1: tp += 1
            elif t_label == 0 and p_label == 1: fp += 1
            elif t_label == 0 and p_label == 0: tn += 1
            elif t_label == 1 and p_label == 0: fn += 1

            if (i + 1) % 50 == 0:
                curr = compute_metrics(tp, fp, tn, fn)
                print(f"    [{label}] {i+1}/{len(test_set)} | F1: {format_pct(curr['f1'])}")

    metrics = compute_metrics(tp, fp, tn, fn)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    eval_count = tp + fp + tn + fn

    print(f"\n  [{label}] FINAL -> F1={format_pct(metrics['f1'])}, P={format_pct(metrics['precision'])}, R={format_pct(metrics['recall'])}")
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn} | Evaluated: {eval_count}/{len(test_set)}")
    print(f"  Avg Latency: {avg_latency:.2f}ms")

    return {
        "label": label,
        "gate_model": gate_model,
        "total": eval_count,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "metrics": metrics,
        "avg_latency_ms": round(avg_latency, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Gate swap comparison experiment")
    parser.add_argument("--max-samples", type=int, default=2000,
                        help="Number of samples (default 2000, matches ablation study)")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    print_header("EXPERIMENT 19: GATE SWAP COMPARISON (ARCHITECTURE-AGNOSTIC PROOF)")

    if load_dataset is None:
        raise RuntimeError("datasets package is required")

    print("  Loading neuralchemy/Prompt-injection-dataset ...")
    dataset = load_dataset("neuralchemy/Prompt-injection-dataset", "core")
    test_set = list(dataset["test"])[:args.max_samples]
    print(f"  Using {len(test_set)} test samples.\n")

    all_results = {}

    for gate_model, label in GATE_MODELS:
        result = run_gate_benchmark(gate_model, label, test_set, workers=args.workers)
        all_results[label] = result
        # Brief pause between models to avoid rate limit cascades
        print("  Pausing 10s before next model...")
        time.sleep(10)

    # Save JSON
    os.makedirs(RESULTS_ROOT / "gate_swap", exist_ok=True)
    write_json(RESULTS_ROOT / "gate_swap" / "gate_swap_comparison.json", all_results)

    # Save CSV
    header = [
        "configuration", "gate_model", "sample_count",
        "accuracy", "precision", "recall", "f1",
        "fpr", "fnr", "avg_latency_ms"
    ]
    rows = []
    for label, data in all_results.items():
        m = data["metrics"]
        rows.append([
            label,
            data["gate_model"],
            data["total"],
            round(m["accuracy"], 4),
            round(m["precision"], 4),
            round(m["recall"], 4),
            round(m["f1"], 4),
            round(m["false_positive_rate"], 4),
            round(m["false_negative_rate"], 4),
            data["avg_latency_ms"],
        ])
    write_csv_rows(TABLES_DIR / "gate_swap_comparison.csv", header, rows)

    # Print summary comparison
    print_header("GATE SWAP COMPARISON SUMMARY")
    print(f"  {'Configuration':<40} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("  " + "-" * 68)
    for label, data in all_results.items():
        m = data["metrics"]
        print(f"  {label:<40} {format_pct(m['f1']):>8} {format_pct(m['precision']):>10} {format_pct(m['recall']):>8}")

    print_header("EXPERIMENT 19 COMPLETE")


if __name__ == "__main__":
    main()
