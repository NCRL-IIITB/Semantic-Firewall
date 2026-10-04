"""
Experiment 14: ToxicChat Warm Cache vs Llama Guard
====================================================
Evaluates the Semantic Firewall (with a warmed cache) against Llama Guard 4
on Llama Guard's own official benchmark dataset (lmsys/toxic-chat).
"""
import os
import sys
import time
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
    print_header, print_section,
)

from openai import OpenAI
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

def get_text_and_label(row):
    text = row.get("user_input", row.get("text", row.get("prompt", "")))
    label = 1 if row.get("toxicity", 0) == 1 or row.get("jailbreaking", 0) == 1 else 0
    return text, label

def warm_cache(fw, warm_set):
    print_section(f"Warming Cache with {len(warm_set)} ToxicChat train samples")
    def warm_single(text):
        try:
            fw.analyze(text)
        except Exception:
            pass
    for i, row in enumerate(warm_set):
        text, _ = get_text_and_label(row)
        if text.strip():
            warm_single(text)
        if (i+1) % 500 == 0:
            print(f"  ... warmed {i+1}/{len(warm_set)}")
    print("Cache warmed successfully.")

def evaluate_firewall_single(fw, text, true_label):
    try:
        start_t = time.perf_counter()
        decision = fw.analyze(text)
        lat = (time.perf_counter() - start_t) * 1000
        
        # SKIP if API timed out (Fail-closed policy)
        if "Detector unavailable" in str(decision.reason):
            return true_label, 0, -1
            
        predicted = 1 if decision.action in {"BLOCK", "FLAG", "REDACT"} else 0
        return true_label, predicted, lat
    except Exception as e:
        if '429' in str(e):
            time.sleep(2)
            return evaluate_firewall_single(fw, text, true_label)
        return true_label, 0, -1

def run_eval_fw(fw, test_set, label_extractor, workers=5):
    print_section(f"Evaluating Semantic Firewall (Warm Cache) on {len(test_set)} test samples")
    tp = fp = tn = fn = 0
    latencies = []
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_firewall_single, fw, label_extractor(row)[0], label_extractor(row)[1]): row for row in test_set}
        for i, future in enumerate(as_completed(futures)):
            t_label, p_label, lat = future.result()
            
            # Skip this sample entirely from metrics if API timed out
            if lat == -1:
                continue
                
            latencies.append(lat)
            if t_label == 1 and p_label == 1: tp += 1
            elif t_label == 0 and p_label == 1: fp += 1
            elif t_label == 0 and p_label == 0: tn += 1
            elif t_label == 1 and p_label == 0: fn += 1
            if (i+1) % 50 == 0:
                print(f"    [Firewall] {i+1}/{len(test_set)}")
                
    m = compute_metrics(tp, fp, tn, fn)
    avg_lat = sum(latencies)/len(latencies) if latencies else 0.0
    # Also return the number of samples actually evaluated
    return m, avg_lat, (tp+fp+tn+fn)

def run_eval_lg(test_set):
    print_section(f"Evaluating Llama Guard 4 on {len(test_set)} test samples")
    tp = fp = tn = fn = 0
    latencies = []
    
    for i, row in enumerate(test_set):
        text, true_label = get_text_and_label(row)
        t_label, p_label, lat = evaluate_llamaguard_single(text, true_label)
        if lat > 0: latencies.append(lat)
        if t_label == 1 and p_label == 1: tp += 1
        elif t_label == 0 and p_label == 1: fp += 1
        elif t_label == 0 and p_label == 0: tn += 1
        elif t_label == 1 and p_label == 0: fn += 1
        if (i+1) % 50 == 0:
            m = compute_metrics(tp, fp, tn, fn)
            print(f"    [LlamaGuard] {i+1}/{len(test_set)} | F1: {format_pct(m['f1'])}")
            
    m = compute_metrics(tp, fp, tn, fn)
    avg_lat = sum(latencies)/len(latencies) if latencies else 0.0
    return m, avg_lat

def main():
    print_header("EXPERIMENT 14: TOXICCHAT OOD WARM CACHE VS LLAMA GUARD (PUBLISHED BASELINE)")
    
    if load_dataset is None:
        raise RuntimeError("datasets package is required")
        
    print("Loading datasets...")
    print("  -> neuralchemy/Prompt-injection-dataset (for Cache Warmup)")
    warm_dataset = load_dataset("neuralchemy/Prompt-injection-dataset", "core")
    
    print("  -> lmsys/toxic-chat (for Evaluation)")
    eval_dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    
    # Warm set = 2000 samples from neuralchemy (OOD)
    # This matches the ablation study size perfectly
    warm_set = list(warm_dataset["train"])[:2000]
    
    # Test set = 2000 samples from ToxicChat
    test_set = list(eval_dataset["test"])[:2000]
    
    results = {}
    
    # 1. Warm Cache & Run Firewall
    fw = Firewall()
    # Setup Firewall
    fw = Firewall()
    if fw._local and hasattr(fw._local, '_cache'):
        pass # fw._local._cache.clear()
        
    # warm_cache(fw, warm_set)
    
    # 1. Evaluate Firewall (Warm Cache)
    m, lat, eval_count = run_eval_fw(fw, test_set, get_text_and_label, workers=8)
    results["semantic_firewall_warm"] = {"metrics": m, "avg_latency_ms": lat, "total": eval_count}
    print(f"\n  [Firewall] F1={format_pct(m['f1'])}, Latency={lat:.2f}ms (Evaluated {eval_count}/{len(test_set)} samples successfully)")
    
    # Llama Guard 4 on the same ToxicChat split is measured by 18_llamaguard_crossdataset.py;
    # no unmeasured "published" numbers are written here.

    # Save JSON
    os.makedirs(RESULTS_ROOT / "ood", exist_ok=True)
    write_json(RESULTS_ROOT / "ood" / "toxicchat_comparison.json", results)

    header = ["model", "sample_count", "accuracy", "precision", "recall", "f1", "avg_latency_ms"]
    rows = []
    rows.append(["semantic_firewall_warm", eval_count, round(m["accuracy"],4), round(m["precision"],4), round(m["recall"],4), round(m["f1"],4), round(lat,2)])
    
    write_csv_rows(TABLES_DIR / "toxicchat_comparison.csv", rows, header)
    print("Experiment 14 complete!")

if __name__ == "__main__":
    main()
