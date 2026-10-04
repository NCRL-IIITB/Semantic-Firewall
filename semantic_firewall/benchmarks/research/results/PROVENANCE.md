# Legacy results (submission version): what each file actually measured

These raw outputs come from the runs behind the submitted paper (May–June 2026). Edited tables, figures
built from estimated values, and scripts that wrote hardcoded numbers were removed in branch
`camera-ready-fixes`. The camera-ready numbers are produced by `../camera_ready/` and written to
`../results_camera_ready/`.

| File | What it actually measured | Caveats |
|---|---|---|
| `ablation/chunked_ablation.json` | Configs on neuralchemy **train[:2000]** (1,654 attacks / 346 benign) | Config settings were mislabelled. Only "All Fast", "Full System" and "Config 9" store confusion matrices; the first and its summary metrics disagree. Recorded latencies are ~24 s/request. The persistent cache was shared across configs (test-set leakage). |
| `ablation/ablation_*.json` | Older ablation, first 100–500 test prompts | "no_semantic_cache" toggled only the exact-match cache, not the vector cache. |
| `baselines/baseline_*.json` | Regex / LLM / full system on the first 500 test prompts | |
| `baselines/cross_dataset_generalization.json` | neuralchemy **test rows 500–942** (442 prompts) | This is not deepset/prompt-injections. |
| `benign/benign_degradation.json` | tatsu-lab/alpaca, 2,000 prompts: **FPR 7.8%** (156/2,000) | Not no_robots. |
| `latency/clean_latency.json` | 10 prompts × 8 configs, sequential: ~2 s median for every config | Most of the overhead was the vector cache rebuilding its ONNX session on every query (fixed). |
| `llm_gate/llm_gate_analysis.json` | Full test split (942). Without gate: P 90.8 / R 91.3 / F1 91.1 / FPR 13.1% | "llm_invoked" counts LLM-agent *detections*, not calls. |
| `multi_turn/multi_turn_evasion.json` | **10 hand-written sessions × 20** (the dataset load fell back to synthetic sessions) | Regex-only and full system both scored 90%. |
| `ood/warmed_cache_ood.json` | deepset/prompt-injections train (546): F1 6.6% | Both configs gave identical outputs. |
| `ood/multidataset_eval.json` | BeaverTails 30k_test, 898 of the first 1,000: F1 74.8, FPR 67.8% | |
| `ood/llamaguard_crossdataset.json` | Llama Guard 4 on ToxicChat test and BeaverTails (first 2,000 each) | Different BeaverTails rows from the firewall run. |
| `ood/toxicchat_comparison.json` | Firewall on ToxicChat test (1,733 evaluated): F1 49.6 | The "llama_guard_published" entry is not a measurement. |
| `pii/pii_scaled_summary.json` | All 209,261 rows, 12 entity types: **P 53.8 / R 75.3 / F1 62.7** | Reproduced exactly by `camera_ready/pii_eval.py` (legacy protocol). |
| `red_team/*` | 56 LLM-generated attacks; 48 blocked (85.7%) | |
| `security_models/security_model_comparison.json` | Llama Guard 4 / GPT-OSS-Safeguard on the 942 test prompts | GPT-OSS-Safeguard was run without a policy prompt. |
| `threshold/threshold_sensitivity.json` | Sweep of the *keyword LLM-gate* threshold (0.80–0.99) on train[:500] | Not a sweep of the cache threshold τ. |
| `unsafe/unsafe_scaled_summary.json` | lmsys/toxic-chat train (384 toxic / 1,616 benign): P 69.0 / R 79.0 / F1 73.7 | |
| `SemanticFirewall_ZeroDay_Dataset.csv` | 212 red-team prompts, all labelled as attacks | Includes generator refusals ("I can't fulfill that request"). |
