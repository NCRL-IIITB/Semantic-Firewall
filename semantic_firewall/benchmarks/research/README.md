# Research experiments

## Camera-ready evaluation (use this)

`camera_ready/` holds the evaluation used for the camera-ready paper. Every script writes per-sample
predictions, so each reported number can be recomputed from raw records under `results_camera_ready/`.

Protocol:
- **Data:** neuralchemy/Prompt-injection-dataset (core) with its official splits (train 4,391 / validation 941 / test 942). The splits share no `group_id` and no identical text.
- **Tuning:** thresholds are chosen on validation, and test is evaluated once.
- **Cache:** every run starts from its own empty semantic cache. "Warm" means seeded with train-split attacks only, then frozen.
- **Failures:** LLM failures are counted and reported, never silently treated as benign.

| Script | Purpose |
|---|---|
| `run_system.py` | Firewall configurations and ablations (`--config`, see `common.CONFIGS`), including leave-one-out per detector |
| `run_baselines.py` | Zero-shot LLMs, Llama Guard 4, local classifiers (Prompt Guard 2, ProtectAI DeBERTa v2, PIGuard) on the same samples |
| `tune_threshold.py` | Cache threshold τ sweep on validation (exact, from a frozen cache) |
| `cache_overlap.py` | Near-duplicate analysis: results by similarity to the nearest cached train attack |
| `significance.py` | Paired McNemar tests with Holm–Bonferroni correction, bootstrap CIs |
| `prevalence_cost.py` | LLM-call rate, cost and p95/p99 latency at 1–50% attack prevalence; Eq. (7) path probabilities |
| `cache_robustness.py` | Error propagation, targeted cache poisoning, allowlist abuse, cache growth/eviction (offline) |
| `adaptive_attack.py` | Adaptive black-box attackers: rule-based mutations (offline) and a PAIR-style small open-weights attacker |
| `pii_eval.py` | PII detector on all 209,261 ai4privacy rows and all 12 entity types (offline) |

Typical order:

```bash
cd semantic_firewall/benchmarks/research/camera_ready
python run_system.py --config no_cache --split validation --cache none
python tune_threshold.py --split validation --downstream-run neuralchemy-validation-no_cache-none
python run_system.py --config full --split test --cache warm_frozen --tau <selected tau>
python run_system.py --config llm_agents_only --split test --cache none
python run_baselines.py --kind hf --model protectai/deberta-v3-base-prompt-injection-v2
python significance.py --reference neuralchemy-test-full-warm_frozen --compare <baseline runs...>
python prevalence_cost.py --run neuralchemy-test-full-warm_frozen --price-per-mtok <price>
python cache_robustness.py all
python adaptive_attack.py --attacker rules --config full --goals 100 --budget 30
```

API-dependent runs need `OPENROUTER_API_KEY` in `.env`. The gate model defaults to `meta-llama/llama-3.3-70b-instruct`.

## Legacy experiments

`experiments/` and `results/` are the scripts and raw outputs behind the submitted version.
`results/PROVENANCE.md` records what each raw file actually measured and its caveats; read it before reusing any legacy number.
