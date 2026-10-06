# Baseline runs reused from the 2026-10-04 session

The `*-baseline-*` runs in this folder were produced on 2026-10-04 by `camera_ready/run_baselines.py`
(zero-shot prompt in that script; Llama Guard 4 via its own first-line safe/unsafe output; local HF classifiers)
in a working copy that was later deleted, and were copied back on 2026-10-06 from that copy.

Checks done before reuse (2026-10-06):
- every run's sample ids equal the current loader's ids (neuralchemy core test, 942; deepset train+test, 662;
  BeaverTails 30k_test seeded sample of 1,000);
- `n_llm_failed` = 0 in every summary.json;
- code fingerprints recorded in each summary.json (`c14f6a2af17921ab` for the neuralchemy LLM runs).

Model identifiers are the OpenRouter IDs in the folder names; `openai/gpt-4o` is OpenRouter's alias for the
current GPT-4o snapshot at query time (2026-10-04), not a dated snapshot.
