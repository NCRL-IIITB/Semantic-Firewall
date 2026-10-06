# Research evaluation (ICISS 2026 camera-ready)

`camera_ready/` holds every script behind the paper's numbers, and `results_camera_ready/` holds the raw
per-prompt outputs they wrote. Every table value can be recomputed from these records; see
[`results_camera_ready/README.md`](results_camera_ready/README.md) for which run backs which table.

## Protocol

- **Data:** neuralchemy/Prompt-injection-dataset (core) with its official splits: train 4,391 / validation 941 /
  test 942. The splits share no `group_id` and no identical text.
- **Cache:** "warm" means the semantic cache is seeded with **train-split attacks only** and then frozen, so
  nothing from validation or test enters it. "Cold" starts empty and writes back during the run.
- **Tuning:** the cache threshold τ and the ensemble threshold θ are chosen on validation (τ = 0.65, θ = 3.5);
  the test split is evaluated once.
- **LLM gate:** `meta-llama/llama-3.3-70b-instruct` via OpenRouter, temperature 0, JSON mode. Accuracy runs use
  the hosts DeepInfra, Parasail and AkashML. The latency runs use Groq.
- **Failures:** a failed LLM call is fail-closed (FLAG) and counted in `llm_failed`; it is never treated as benign.
- **Provenance:** each `summary.json` stores a fingerprint of the detector/orchestrator code and the run metadata.

## Scripts

| Script | Purpose | Paper |
|---|---|---|
| `run_system.py` | One firewall configuration on one dataset split (`--config`, `--cache`, `--tau`, `--env`). Resumable. | Tables 2–4 |
| `derive_warm.py` | Warm-cache run at τ, derived exactly from a no-cache run plus nearest-train-attack similarity | Tables 2, 4 (C8) |
| `tune_threshold.py` | Exact τ × θ sweep from a no-cache run (any dataset) | Table 3 |
| `run_baselines.py` | Zero-shot LLMs (OpenRouter), Llama Guard 4, local classifiers (DeBERTa-v3, PIGuard) | Table 2 |
| `significance.py` | Paired McNemar tests with Holm–Bonferroni correction | Sec. 5.1, 5.3 |
| `final_tables.py` | Rebuilds the table numbers from raw records | Tables 2, 3 |
| `latency_report.py` | p50/p95/p99, per-path and per-stage latency, Eq. 8 L_eff, 1–10% attack prevalence | Sec. 5.1, Cost |
| `novel_subset.py` | Results by similarity to the nearest cached train attack (memorisation check) | Sec. 5.2 |
| `pii_eval.py` | PII detector on ai4privacy (209,261 rows; `--part dev/heldout`, `--profile ai4privacy`) | Table 2c |
| `build_ood_set.py` | Builds OOD-500 (`data/datasets/ood_curated_500.jsonl`) | Table 3b |
| `multiturn_eval.py` | Multi-turn sessions: SafeMTData attacks + benign no_robots chats (held-out 100+100, dev 50+50) | Table 2c (pending) |
| `build_multiturn_examples.py` | Worked examples for the Session Judge (never from the held-out part) | — |
| `levenshtein_study.py` | Levenshtein keyword-repair study on natural and obfuscated attacks | Sec. 3.3 |
| `common.py` | Shared loaders, configurations, firewall builder, metrics | — |

Not yet used in the paper: `cache_overlap.py`, `cache_robustness.py` (error propagation, cache poisoning,
allowlist abuse, cache growth), `adaptive_attack.py` (adaptive attackers), `prevalence_cost.py` (token cost
model), and the `llm_agents_only` / `loo:<detector>` configurations of `run_system.py`.

## Reproducing the main results

```bash
cd semantic_firewall/benchmarks/research/camera_ready
export HF_DATASETS_OFFLINE=0            # first run downloads the datasets
# 1. validation, no cache -> choose tau and theta
python run_system.py --config full_all_unresolved --split validation --cache none
python tune_threshold.py --dataset neuralchemy --split validation \
    --downstream-run neuralchemy-validation-full_all_unresolved-none
# 2. test, no cache, then the warm-cache result at the selected tau
python run_system.py --config full_all_unresolved --split test --cache none
python derive_warm.py --source neuralchemy-test-full_all_unresolved-none --dataset neuralchemy --split test --tau 0.65
# 3. baselines and significance
python run_baselines.py --kind llm --model openai/gpt-4o
python run_baselines.py --kind hf --model protectai/deberta-v3-base-prompt-injection-v2
python significance.py --reference neuralchemy-test-full_all_unresolved-warm_frozen-tau0.65 \
    --compare neuralchemy-test-baseline-llm-openai__gpt-4o
# 4. latency (sequential, first 10 prompts are warm-up)
python run_system.py --config full_all_unresolved --split test --cache warm_frozen --tau 0.65 \
    --max-samples 210 --workers 1 --run-name neuralchemy-test-latency-groq \
    --env SEMANTIC_FIREWALL_OPENROUTER_PROVIDERS=Groq
python latency_report.py --latency-run neuralchemy-test-latency-groq \
    --paths-run neuralchemy-test-full_all_unresolved-warm_frozen-tau0.65
```

Ablation rows use `--config` `injection_llm_only` (C1), `regex_only` (C2), `cache_only` (C3), `detectors_only`
(C4), `regex_cache` (C5), `deterministic_only` (C6), `fast_layers` (C7) and `full_all_unresolved` with
`--cache cold` (C9). Cross-dataset runs use `--dataset deepset --split all`, `--dataset beavertails`,
`--dataset ood500 --split all` and `--dataset no_robots`.

API-dependent runs need `OPENROUTER_API_KEY` in the repository's `.env` (git-ignored). The test suite writes
into `config/*.json`; restore them with `git checkout -- config/` after running it.
