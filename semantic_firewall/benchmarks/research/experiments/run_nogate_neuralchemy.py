"""
Quick experiment: neuralchemy full_system with LLM Gate DISABLED (threshold=0.0)
Shows true max recall of the LLM detector on injection/jailbreak attacks.
Compares to the gated run (F1=24.5%, R=14.3%) to quantify the gate's cost/benefit.

Uses max 40 samples to stay within free-tier API budget.
"""
import os, sys, time, argparse, json
from pathlib import Path

from helpers import (
    PROJECT_ROOT, RESULTS_ROOT, TABLES_DIR,
    compute_metrics, format_pct, write_csv_rows, write_json,
    experiment_metadata, print_header, print_section,
)

sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Force LLM gate OFF — every prompt goes to LLM
os.environ["SEMANTIC_FIREWALL_DISABLE_LLM_DETECTORS"] = "0"
os.environ["SEMANTIC_FIREWALL_LLM_GATE_ENABLED"] = "0"
os.environ["SEMANTIC_FIREWALL_LLM_GATE_THRESHOLD"] = "0.0"

from datasets import load_dataset
from semantic_firewall.sdk import Firewall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument("--sleep", type=float, default=4.0)
    args = parser.parse_args()

    print_header("NEURALCHEMY — FULL SYSTEM, NO LLM GATE (true max recall)")

    fw = Firewall()
    dataset = load_dataset("neuralchemy/Prompt-injection-dataset", "core")
    test_set = dataset["test"].select(range(min(args.max_samples, len(dataset["test"]))))
    print(f"  Using {len(test_set)} samples from neuralchemy test set\n")

    tp = fp = tn = fn = 0
    results = []

    for i, example in enumerate(test_set):
        text = example["text"]
        true_label = int(example["label"])

        start = time.perf_counter()
        try:
            decision = fw.analyze(text)
            elapsed = (time.perf_counter() - start) * 1000
            predicted = 1 if decision.action in {"BLOCK", "FLAG", "REDACT"} else 0
        except Exception as e:
            elapsed = 0
            predicted = 0
            print(f"  [ERROR] sample {i}: {e}")

        if true_label == 1 and predicted == 1: tp += 1
        elif true_label == 0 and predicted == 1: fp += 1
        elif true_label == 0 and predicted == 0: tn += 1
        else: fn += 1

        results.append({
            "id": i, "true_label": true_label, "predicted": predicted,
            "action": decision.action if 'decision' in dir() else "ERROR",
            "latency_ms": round(elapsed, 2),
            "text": text[:120],
        })

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(test_set)} — TP={tp}, FP={fp}, FN={fn}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    metrics = compute_metrics(tp, fp, tn, fn)

    print_section("Results — No LLM Gate")
    print(f"  Samples tested : {len(test_set)}")
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  Precision      : {format_pct(metrics['precision'])}")
    print(f"  Recall         : {format_pct(metrics['recall'])}")
    print(f"  F1             : {format_pct(metrics['f1'])}")
    print(f"  FPR            : {format_pct(metrics['false_positive_rate'])}")


    out_dir = RESULTS_ROOT / "baselines"
    write_json(out_dir / "baseline_nogate_neuralchemy.json", {
        "meta": experiment_metadata(),
        "config": "full_system_no_llm_gate",
        "dataset": "neuralchemy/Prompt-injection-dataset",
        "num_samples": len(test_set),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "metrics": metrics,
        "results": results,
    })

    header = ["config", "dataset", "samples", "precision", "recall", "f1", "fpr"]
    rows = [
        ["no_gate_0.0", "neuralchemy", len(test_set),
         round(metrics["precision"], 4), round(metrics["recall"], 4),
         round(metrics["f1"], 4), round(metrics["false_positive_rate"], 4)],
    ]
    write_csv_rows(TABLES_DIR / "gate_vs_nogate_neuralchemy.csv", header, rows)
    print_header("COMPLETE")


if __name__ == "__main__":
    main()
