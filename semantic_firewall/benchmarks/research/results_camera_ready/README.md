# Raw results (camera-ready)

Every run folder in `runs/` contains:
- `predictions.jsonl`: one record per prompt, with id, label, prediction, action, resolution stage, LLM called/failed, LLM host, latency and per-stage latency, ensemble risk, triggered detectors, and cache similarity/hit for warm runs;
- `summary.json`: the metrics, plus a code fingerprint and run metadata.

Prompt texts are not stored. Records refer to the public datasets by id.

Run names follow `<dataset>-<split>-<config>-<cache>`. Runs ending in `-warm_frozen-tau0.65` are derived exactly
from the matching `-none` run with `camera_ready/derive_warm.py`.

## Runs reported in the paper

| Paper | Run(s) |
|---|---|
| Table 2a, ours (warm) / Table 4 C8 | `neuralchemy-test-full_all_unresolved-warm_frozen-tau0.65` (from `…-none`) |
| Table 2a, ours (cold) / Table 4 C9 | `neuralchemy-test-full_all_unresolved-cold`: **partial (344/942), resumable; not yet reported** |
| Table 2a, baselines | `neuralchemy-test-baseline-llm-openai__gpt-4o`, `…-google__gemini-2.5-flash`, `…-anthropic__claude-haiku-4.5`, `…-hf-protectai__deberta-v3-base-prompt-injection-v2`, `…-hf-leolee99__PIGuard` (provenance: `runs/BASELINES_PROVENANCE.md`) |
| Table 2b BeaverTails | `beavertails-test-full_all_unresolved-warm_frozen-tau0.65`, `beavertails-test-baseline-llama_guard-meta-llama__llama-guard-4-12b` |
| Table 2b deepset | `deepset-all-full_all_unresolved-warm_frozen-tau0.65` (Rebuff row is the published number of Palit & Woods) |
| Table 2c PII | `pii/pii_eval_ai4privacy.json` (all rows; `pii_eval_dev*`/`pii_eval_heldout*` are the dev/held-out halves, `pii_eval.json` the default profile) |
| Table 2c no_robots | `no_robots-test-full_all_unresolved-warm_frozen-tau0.65` (from `…-none`; 14 blocked = 2.80%, 37 flagged/redacted = 7.4%) |
| Table 2c multi-turn | pending (`multiturn/split.json` fixes the held-out 100+100 and dev 50+50 sessions) |
| Table 3a | `threshold/tau_sweep_neuralchemy_validation_neuralchemy-validation-full_all_unresolved-none.*` from `neuralchemy-validation-full_all_unresolved-none` |
| Table 3b | `threshold/tau_sweep_ood500_all_ood500-all-full_all_unresolved-none.*` from `ood500-all-full_all_unresolved-none` |
| Table 4 C1–C7 | `neuralchemy-test-injection_llm_only-none`, `…-regex_only-none`, `…-cache_only-warm_frozen`, `…-detectors_only-none`, `…-regex_cache-warm_frozen`, `…-deterministic_only-none`, `…-fast_layers-warm_frozen` |
| Table 4 latency | `neuralchemy-test-latency-<config>` (210 prompts, one at a time, first 10 are warm-up) |
| Latency, Eq. 8, prevalence | `latency/latency_neuralchemy-test-latency-groq.json` from `neuralchemy-test-latency-groq` |
| McNemar | `significance/significance.json` (vs baselines), `significance/ablation.json` (vs C1, C7) |
| Warm-cache check | `significance/novel_subset.json` |
| Levenshtein study | `levenshtein_study.json` |
| All table numbers | `final_tables.json` (`camera_ready/final_tables.py`) |

## Other runs kept as logs (not reported)

| Run(s) | Why it is not in the paper |
|---|---|
| `*-no_cache-none`, `neuralchemy-validation-deterministic_only-none*`, `deepset-all-deterministic_only-none`, matching `threshold/` sweeps | Earlier design in which a keyword gate decided whether to call the LLM (recall 14% on deepset, 6% on BeaverTails); replaced by `full_all_unresolved` |
| `neuralchemy-test-baseline-llm-openai__gpt-4o-mini`, `…-gpt-oss-120b`, `deepset-all-baseline-*` | Extra baselines run but not reported (author decision) |
| `neuralchemy-test-baseline-llama_guard-meta-llama__llama-guard-4-12b` | Included in the McNemar tests; not a Table 2a row |
| `beavertails-test-full_all_unresolved-none`, `deepset-all-full_all_unresolved-none`, `ood500-all-full_all_unresolved-none`, `neuralchemy-test-full_all_unresolved-none` | No-cache sources of the derived warm runs |
| `multiturn/multiturn-dev-*` | Development runs of the multi-turn check (dev part only) |

`_work/` (git-ignored) holds the per-run vector stores and can be deleted.
