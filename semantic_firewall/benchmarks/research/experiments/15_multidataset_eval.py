"""
Experiment 15: Multiple Dataset Benchmarking
====================================================
Evaluates the Semantic Firewall (with a warmed OOD cache) on
multiple HuggingFace datasets used by Llama Guard:
1. allenai/xstest (for False Positives / Over-refusal)
2. PKU-Alignment/BeaverTails (for Safety Classification)

Uses published baselines for Llama Guard to save API credits.
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

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

def get_xstest_label(row):
    text = row.get("prompt", "")
    label_text = row.get("label", "")
    # XSTest labels: safe, unsafe, etc. If it is 'safe', true label = 0
    return text, 1 if "unsafe" in label_text.lower() else 0

def get_beavertails_label(row):
    text = row.get("prompt", "")
    # BeaverTails label is a boolean 'is_safe'
    is_safe = row.get("is_safe", True)
    return text, 0 if is_safe else 1

def warm_cache(fw, warm_set):
    print_section(f"Warming Cache with {len(warm_set)} Neuralchemy OOD samples")
    def warm_single(text):
        try:
            fw.analyze(text)
        except Exception:
            pass
    for i, row in enumerate(warm_set):
        text = row.get("text", "")
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
    tp = fp = tn = fn = 0
    latencies = []
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_firewall_single, fw, label_extractor(row)[0], label_extractor(row)[1]): row for row in test_set}
        for i, future in enumerate(as_completed(futures)):
            t_label, p_label, lat = future.result()
            if lat == -1: continue
            
            latencies.append(lat)
            if t_label == 1 and p_label == 1: tp += 1
            elif t_label == 0 and p_label == 1: fp += 1
            elif t_label == 0 and p_label == 0: tn += 1
            elif t_label == 1 and p_label == 0: fn += 1
            if (i+1) % 50 == 0:
                print(f"    [Firewall] {i+1}/{len(test_set)}")
                
    m = compute_metrics(tp, fp, tn, fn)
    m['tp'] = tp
    m['fp'] = fp
    m['tn'] = tn
    m['fn'] = fn
    avg_lat = sum(latencies)/len(latencies) if latencies else 0.0
    return m, avg_lat, (tp+fp+tn+fn)

def main():
    print_header("EXPERIMENT 15: MULTIPLE DATASET BENCHMARKING (OOD WARM CACHE)")
    
    if load_dataset is None:
        raise RuntimeError("datasets package is required")
        
    print("Loading datasets...")
    print("  -> neuralchemy/Prompt-injection-dataset (for Cache Warmup)")
    warm_dataset = load_dataset("neuralchemy/Prompt-injection-dataset", "core")
    warm_set = list(warm_dataset["train"])[:2000]
    
    print("  -> allenai/xstest (for Over-refusal Evaluation)")
    try:
        xstest_data = list(load_dataset("allenai/xstest", "test")["test"])[:1000]
    except Exception:
        print("  [Warning] Could not load allenai/xstest (likely gated/removed). Skipping.")
        xstest_data = []
        
    print("  -> PKU-Alignment/BeaverTails (for Safety Classification Evaluation)")
    try:
        beaver_data = list(load_dataset("PKU-Alignment/BeaverTails", split="30k_test"))[:1000]
    except Exception:
        # BeaverTails fallback
        beaver_data = list(load_dataset("PKU-Alignment/BeaverTails", split="330k_test"))[:1000]
    
    results = {}
    
    # Setup Firewall
    fw = Firewall()
    if fw._local and hasattr(fw._local, '_cache'):
        pass # fw._local._cache.clear() # USER REQUESTED TO REUSE WARM CACHE
        
    # warm_cache(fw, warm_set) # USER REQUESTED TO REUSE WARM CACHE
    
    # 1. Evaluate on XSTest
    if xstest_data:
        print_section(f"Evaluating Semantic Firewall on {len(xstest_data)} XSTest samples")
        m_xs, lat_xs, eval_xs = run_eval_fw(fw, xstest_data, get_xstest_label, workers=8)
        results["xstest_firewall"] = {"metrics": m_xs, "avg_latency_ms": lat_xs, "total": eval_xs}
        print(f"\n  [XSTest Firewall] FPR (Over-refusal) = {format_pct(m_xs['fp'] / max((m_xs['fp'] + m_xs['tn']), 1))} | F1={format_pct(m_xs['f1'])}, Latency={lat_xs:.2f}ms")
    else:
        m_xs = {'accuracy':0, 'precision':0, 'recall':0, 'f1':0, 'fp':0, 'tn':0}
        lat_xs = 0.0
        eval_xs = 0
    
    # 2. Evaluate on BeaverTails
    print_section(f"Evaluating Semantic Firewall on {len(beaver_data)} BeaverTails samples")
    m_bv, lat_bv, eval_bv = run_eval_fw(fw, beaver_data, get_beavertails_label, workers=8)
    results["beavertails_firewall"] = {"metrics": m_bv, "avg_latency_ms": lat_bv, "total": eval_bv}
    print(f"\n  [BeaverTails Firewall] F1={format_pct(m_bv['f1'])}, Latency={lat_bv:.2f}ms")
    
    # Save Results
    os.makedirs(RESULTS_ROOT / "ood", exist_ok=True)
    write_json(RESULTS_ROOT / "ood" / "multidataset_eval.json", results)
    
    header = ["dataset", "model", "sample_count", "accuracy", "precision", "recall", "f1", "avg_latency_ms", "fpr"]
    rows = []
    
    # Only measured rows are written. Llama Guard numbers come from 18_llamaguard_crossdataset.py.
    fpr_xs = m_xs['fp'] / max((m_xs['fp'] + m_xs['tn']), 1)
    if eval_xs:
        rows.append(["xstest", "semantic_firewall_warm", eval_xs, round(m_xs["accuracy"], 4), round(m_xs["precision"], 4), round(m_xs["recall"], 4), round(m_xs["f1"], 4), round(lat_xs, 2), round(fpr_xs, 4)])

    fpr_bv = m_bv['fp'] / max((m_bv['fp'] + m_bv['tn']), 1)
    rows.append(["beavertails", "semantic_firewall_warm", eval_bv, round(m_bv["accuracy"], 4), round(m_bv["precision"], 4), round(m_bv["recall"], 4), round(m_bv["f1"], 4), round(lat_bv, 2), round(fpr_bv, 4)])
        
    write_csv_rows(TABLES_DIR / "multidataset_comparison.csv", rows, header)
    print("Experiment 15 complete!")

if __name__ == "__main__":
    main()
