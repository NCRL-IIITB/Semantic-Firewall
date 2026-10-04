# Legacy results (pre camera-ready)

Raw outputs from the earlier experiment scripts (May–June 2026). They are superseded by the
camera-ready evaluation in `../camera_ready/`, which writes to `../results_camera_ready/` and uses
the official neuralchemy train/validation/test splits. This table records what each file contains.

| File | Contents |
|---|---|
| `ablation/chunked_ablation.json` | Pipeline configurations on neuralchemy train[:2000] (1,654 attacks / 346 benign). Superseded by `camera_ready/run_system.py`, where the configurations are redefined. |
| `ablation/ablation_*.json` | Early ablation on the first 100–500 neuralchemy test prompts. |
| `baselines/baseline_*.json` | Regex / LLM / full system on the first 500 neuralchemy test prompts. |
| `baselines/cross_dataset_generalization.json` | Full system on neuralchemy test rows 500–942 (442 prompts). |
| `benign/benign_degradation.json` | tatsu-lab/alpaca, 2,000 benign prompts: FPR 7.8% (156/2,000). |
| `latency/clean_latency.json` | 10 prompts × 8 configurations, sequential; recorded before the embedding-cache speed-up. |
| `llm_gate/llm_gate_analysis.json` | Full neuralchemy test split (942), with and without the keyword gate. |
| `multi_turn/multi_turn_evasion.json` | Multi-turn smoke test with 10 hand-written sessions (×20). |
| `ood/warmed_cache_ood.json` | deepset/prompt-injections train split (546 prompts). |
| `ood/multidataset_eval.json` | BeaverTails 30k_test, 898 of the first 1,000 prompts. |
| `ood/llamaguard_crossdataset.json` | Llama Guard 4 on ToxicChat test and BeaverTails 30k_test (first 2,000 each). |
| `ood/toxicchat_comparison.json` | Firewall on ToxicChat test (1,733 prompts evaluated). The `llama_guard_published` entry is a reference value and was not produced by this code. |
| `pii/pii_scaled_summary.json` | PII detector on all 209,261 ai4privacy rows, 12 entity types (reproduced by `camera_ready/pii_eval.py`). |
| `red_team/*` | 56 LLM-generated attack prompts and the firewall's decisions. |
| `security_models/security_model_comparison.json` | Llama Guard 4 and GPT-OSS-Safeguard on the 942 neuralchemy test prompts. |
| `threshold/threshold_sensitivity.json` | Sweep of the keyword LLM-gate threshold on train[:500] (the cache threshold τ is tuned by `camera_ready/tune_threshold.py`). |
| `unsafe/unsafe_scaled_summary.json` | lmsys/toxic-chat train sample (384 toxic / 1,616 benign). |
| `SemanticFirewall_ZeroDay_Dataset.csv` | 212 LLM-generated red-team prompts. |
