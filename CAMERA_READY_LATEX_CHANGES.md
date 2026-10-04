# Camera-ready changes: Multi-agent Semantic Firewall (ICISS 2026)

Point-wise list of what to change in the LaTeX source, section by section.

**Tags:**
- **[R1] / [R2] / [R3]:** the reviewer who asked for the change.
- **[Audit]:** not raised by a reviewer, but the paper text disagrees with the code or the raw results in the repo, so it has to change too.
- **[NEW RESULT]:** the number has to come from a new run. Leave it as a TODO until then; don't reuse the old value.

Scripts referred to below live in `semantic_firewall/benchmarks/research/camera_ready/`.

---

## 0. Before you start

- [ ] **Check you're editing the submitted version.** Reviewer 1 quotes an abstract that pairs *F1 83.93%* with *recall 87.59%*, and a Table 3b labelled *N = 500*. Neither appears in `Prompt_injection_Springer_Format.pdf`. Check the EasyChair abstract field too.
- [ ] **Delete numbers that have no source in the repo.** Remove each of these unless a new run reproduces it:
  - the 66.88% and 41.93% fast-path rates
  - L_eff ≈ 444 ms
  - the $500 → $166 cost figures
  - 731.64 ms, 427.72 ms and 459.85 ms
  - McNemar χ² = 4.28 (p = 0.038) and χ² = 135.2
  - the ±0.42 in the abstract
  - all of Table 3a and Table 3b
  - the "deepset" F1 of 83.93% and the Rebuff row
  - Synthetic Safety 82.50 / 84.30 / 83.38
  - FPR 4.20% "on no_robots"
  - the PII figures 85.90 / 96.70 / 91.00 on "209K"
  - multi-turn 90.0%
  - every value in Table 4

---

## 1. Title, abstract, keywords

- [ ] **"self-healing semantic cache" → "self-updating semantic threat cache".** Apply this everywhere: abstract, contributions, §3.2, Fig. 1 and the conclusion. [R2, R3]
- [ ] **"achieves 92.64% F1 (±0.42) and 99.82% recall under cold-start evaluation"** → the F1, recall and FPR on the neuralchemy test split (942 prompts), with 95% bootstrap CIs written as "(95% CI a–b)". [NEW RESULT] [R1]
- [ ] **"bypassing LLM inference for 66.88% of requests on the evaluated traffic distribution"** → "invokes the LLM for x% of benign and y% of attack prompts; at a realistic 5% attack rate, z% of requests reach the LLM". [NEW RESULT: `prevalence_cost.py`] [R1]
- [ ] **"a deterministic Llama-3.3-70B gate ... with schema-constrained outputs"** → "a Llama-3.3-70B-Instruct gate (temperature 0) prompted to return a JSON verdict". Only say "constrained" if you re-run with `SEMANTIC_FIREWALL_LLM_JSON_MODE=1`. [R1, Audit]
- [ ] **"Cross-dataset experiments on BeaverTails, DeepSet, AI4Privacy, walledai ... evaluate generalization"** → "We additionally report results on related tasks: harmful-content classification (BeaverTails), PII detection (ai4privacy) and multi-turn jailbreaks." [R1]
- [ ] **"outperforms proprietary LLM guardrails in detection accuracy at a third of the cost ... proving to be a practical, production-ready defense"** → a moderated claim that matches the new Table 2. Drop "production-ready" and "a third of the cost". [R1]

## 2. Introduction

- [ ] **"our own profiling of a comparable LLM-only gate measures 731.64 ms"** → the measured p50 and p95 of the `llm_agents_only` run. [NEW RESULT]
- [ ] **Contribution 2** → "A self-updating semantic threat cache with an explicit write-back policy (LLM-confirmed detections only, confidence floor, size cap with eviction), and an evaluation of its poisoning and error-propagation risks." [R2, R3]
- [ ] **Add contribution 4** → "A security evaluation: threat model, adaptive black-box attackers, cache-poisoning and allowlist-abuse experiments." [R1, R2]
- [ ] **"evaluated on five public benchmarks against six representative baselines"** → state the correct counts after the new runs (currently five baselines are named, not six). [R1]

## 3. Related work (§2)

- [ ] **Add training-based and architectural defenses** [R1]:
  - Wallace et al., *The Instruction Hierarchy*, arXiv:2404.13208 (2024)
  - Chen, Piet, Sitawarin, Wagner, *StruQ*, USENIX Security 2025
  - Chen et al., *SecAlign*, ACM CCS 2025
  - Debenedetti et al., *Defeating Prompt Injections by Design* (CaMeL), arXiv:2503.18813 (2025)
- [ ] **Add purpose-built injection classifiers**, which are also new baselines: Llama Prompt Guard 2 (Meta, 2025), ProtectAI `deberta-v3-base-prompt-injection-v2`, and InjecGuard/PIGuard (InjecGuard is already ref [11]). [R1]
- [ ] **Add adaptive-attack work:** Nasr et al., *The Attacker Moves Second*, arXiv:2510.09023 (2025). [R1, R2]
- [ ] **Add a short "Novelty vs Rebuff" paragraph.** Rebuff = heuristics + LLM + vector store of past attacks + canary tokens. Say honestly what's different here: the parallel multi-category detector mesh (PII/secrets/abuse alongside injection); policy profiles with redaction; an explicit write-back policy plus an admin allowlist; early-exit cost accounting. Don't report Rebuff numbers unless you actually run Rebuff. [R1]
- [ ] **Optional:** if you use the sub-agent framing for the new baseline, cite *Recursive Language Models* (Zhang, Kraska, Khattab, arXiv:2512.24601). [R2]

## 4. Method (§3): make the text match the implementation

### 4.1 §3 intro and §3.1 (entry layer)

- [ ] **"A lightweight regex engine then scans ... and immediately blocks matching prompts"** → "Regex pre-screens (20 injection patterns, 20 unsafe-content patterns and 3 threat-intel signatures) run in parallel with the deterministic detectors. If this stage already yields BLOCK, the LLM stage is skipped (early exit)." [Audit]
- [ ] **"average latency of 5.21 ms"** → the measured p50 of the `regex_only` run (15.2 ms on the test split, including orchestration overhead). [Audit]

### 4.2 §3.2 (semantic control plane and memory) [R2, R3]

- [ ] **"We set τ = 0.90 based on a sweep over the validation set (Section 5, Table 3)"** → "τ is selected on the neuralchemy validation split (Table 3) and fixed before evaluating on test." [R1]
- [ ] **Embedding:** say "all-MiniLM-L6-v2 (384-d, L2-normalised; ChromaDB ONNX implementation), cosine HNSW index".
- [ ] **Replace the "symmetric, self-healing feedback loop" paragraph** with a precise description of the mechanism:
  - **Lookup order:** embed the prompt once → allowlist (similarity ≥ 0.90) → threat cache (similarity ≥ τ, top-1) → detectors.
  - **What is stored:** the raw prompt text and its embedding, plus metadata (threat type, severity, source detector, confidence, LLM-confirmed flag, origin, timestamp). Entries are de-duplicated by MD5 of the text.
  - **Write-back policy:** an entry is written only when an LLM-backed detector confirms the threat with confidence ≥ 0.85. Regex-only hits, detector failures/timeouts and allowlisted prompts are never written.
  - **Allowlist:** entries are added only by an administrator after reviewing a false positive. An allowlist hit skips the threat cache and the LLM stage, but **the deterministic detectors still run** (`allowlist_mode = skip_llm`).
  - **Size:** the cache is capped (default 10,000 entries) and evicts the oldest entries first. There is no TTL.
- [ ] **Session Judge sentence** → describe what it actually does: a cumulative per-session severity score (NONE 0, LOW 0.5, MEDIUM 1.5, HIGH 3, CRITICAL 5; FLAG at 8, BLOCK at 12); the last 3 turns concatenated and re-checked for injection; six rule-based multi-turn patterns. [Audit]

### 4.3 Table 1 and the detector paragraphs [R1, R2]

- [ ] **Secrets Detector** — "signature matching with Shannon entropy" → "about 65 provider-specific signatures (AWS, GCP, Azure, GitHub, Stripe, OpenAI, …) plus generic key/password assignments". The code has no entropy check here. [Audit]
- [ ] **Abuse Detector** — "Levenshtein-distance matching of obfuscated abusive terms" → "structural abuse heuristics: length/token inflation, character/word/phrase repetition, low Shannon entropy, special-character/digit/uppercase floods, homoglyphs, invisible characters, and script/SQL/XXE/SSRF/path-traversal/chat-delimiter injection". There is no Levenshtein code. [Audit]
- [ ] **Context Flooding Detector** — category "LLM-assisted" → "Fast / deterministic" (it's a 4,000-character length check). [Audit, R1 (Fig. 1 vs Table 1)]
- [ ] **Injection / Unsafe Content** → "regex pre-screen, plus an LLM verdict when the LLM stage runs".

### 4.4 Equations [R2 "formulas need cleaning", R1 minor]

- [ ] **Eq. (3)** → the detector stages run in two parallel phases:
  `T_mesh ≈ max_{i∈D_det} T_i + 1[LLM stage runs] · max_{j∈D_LLM} T_j + T_ovh`
- [ ] **Eq. (4) entropy:** sum over distinct symbols, not positions [R1], and move it to the Abuse Detector paragraph:
  `H(X) = − Σ_{s∈Σ(X)} p(s) log2 p(s),  p(s) = count(s)/n`
  It flags inputs of 50 or more characters with H < 2.5 bits (highly repetitive content).
- [ ] **Eq. (5) Levenshtein:** delete it and its paragraph (not implemented). This also frees space for the new sections.
- [ ] **Add an equation for the LLM gate** (the actual escalation rule; it's missing from the paper):
  `g(q) = 1.5·1[|q| ≥ 800] + 2·1[any deterministic finding] + 1·1[q matches a suspicious-keyword list]`
  The LLM stage runs iff (no BLOCK so far) ∧ (g(q) ≥ γ), with γ = 1.0. Also describe the "all-unresolved" variant (every prompt not already blocked goes to the LLM), which is what Algorithm 1 currently shows.
- [ ] **Eq. (6) risk score** → split it in two and report θ [R1]:
  - **Decision rule (what actually decides):** the action is the most severe of the per-detector policy actions (threat type × severity table), escalated by an ensemble score
    `E(q) = Σ_i w_i · p_i · max(1, sev_i) / 2`,
    where p_i is the calibrated detector confidence, w_i ∈ [1.0, 1.3], and sev_i ∈ {0..4}. FLAG / REDACT / BLOCK trigger at E ≥ 1.8 / 2.6 / **θ = 3.5**.
  - **Risk score 0–100:** reported for explainability only. It is not used to decide.
    `R = min(100, S_sev + 30·c̄ + min(5·n_trig, 20) + min(S_session, 10))`, with S_sev ∈ {0, 16.7, 40, 60, 85}.
    S_session is now passed in. It used to be always 0 (bug fixed in code).
- [ ] **Add one sentence justifying the hand-set weights,** or a small sensitivity sweep of θ and w_i (±20%) on validation. [R1 minor] [NEW RESULT, optional]
- [ ] **Eq. (7):** keep the four-term form, but define the paths as semantic cache (P_c), deterministic stage including early exit (P_m) and LLM stage (P_l). The exact-match result cache (P_h) is disabled during evaluation. Use measured P_k and L_k only. [NEW RESULT: `prevalence_cost.py` prints these] [R1]

### 4.5 Algorithm 1 [R1 item 5: align with the text]

- [ ] Replace it with an algorithm that includes the allowlist, the threat cache, the deterministic + regex stage, early exit, the gate g(q), session risk, the FLAG/REDACT outputs, and the fail-closed fallback. Suggested body:

```
Input: prompt q, session s; thresholds τ (cache), τ_a (allowlist), γ (gate), θ (ensemble)
1:  v ← Embed(q)
2:  a ← [max_{u∈A} cos(v,u) ≥ τ_a]                      ▷ admin allowlist
3:  if ¬a and max_{c∈C} cos(v,c) ≥ τ then return BLOCK     ▷ semantic threat cache
4:  R ← ParallelRun(D_det ∪ RegexPrescreen(D_LLM), q)      ▷ no LLM calls
5:  d ← Policy(R)
6:  if ¬a and d ≠ BLOCK and g(q,R) ≥ γ then
7:      R ← R ∪ ParallelRun(D_LLM, q)                       ▷ LLM failure ⇒ SYSTEM_UNAVAILABLE (fail-closed: FLAG)
8:      d ← Policy(R)
9:  d ← max(d, Ensemble(R; θ), Session(s, R))              ▷ action order ALLOW < FLAG < REDACT < BLOCK
10: if LLM-confirmed threat with confidence ≥ 0.85 then C ← C ∪ {v}
11: return d   (REDACT returns the masked prompt)
```

### 4.6 §3.4 (LLM gate)

- [ ] **"Llama-3.3-70B-Instruct through the Groq API"** → "through OpenRouter (`meta-llama/llama-3.3-70b-instruct`)". Also update §4.1 and Fig. 1. [R1]
- [ ] **"returns ALLOW or BLOCK, a confidence value, and an explanation"** → the real output schema: `{is_injection, attacks_found[{type, confidence, evidence}], overall_risk}`. The unsafe-content agent's schema is analogous. Two LLM requests are made per invocation (injection + unsafe content). [Audit]
- [ ] **"applies the configured fallback policy"** → "fails closed: an unavailable LLM detector is reported as SYSTEM_UNAVAILABLE (HIGH) and the request is FLAGged, never silently allowed; such results are never written to the cache." [R1]

### 4.7 §3.9 (security considerations) → new "Threat Model" subsection [R1]

- [ ] **Attacker capabilities:** submits arbitrary prompts; observes ALLOW/BLOCK (and the explanation, if exposed); can query repeatedly (adaptive); knows the architecture. No access to the weights or cache contents.
- [ ] **Assets:** the protected LLM's instructions, user data/PII and secrets, and availability for benign users.
- [ ] **In-scope attacks:** direct injection/jailbreak, evasion of the regex and cache, attacks on the LLM gate itself (prompt injection aimed at the gate — the gate sees untrusted text), cache poisoning, allowlist abuse.
- [ ] **Out of scope:** indirect injection through tools/RAG content, multimodal inputs, a compromised provider.
- [ ] **Privacy:** unresolved prompts go to a third-party API (OpenRouter → model host), and the cache stores raw prompt text locally. Mention data-processing agreements, a self-hosted gate option, and storing redacted text. [R1]

## 5. Experimental setup (§4)

- [ ] **"neuralchemy ... (N = 2,000) ... balanced set"** → "neuralchemy *core* configuration with its official splits: train 4,391 / validation 941 / test 942 (test: 552 attacks, 390 benign). The splits share no `group_id` and no identical text. All thresholds are chosen on validation; results are reported once on test." The old N = 2,000 was the first 2,000 *train* rows: 1,654 attacks / 346 benign (83% attacks). [R1, R3, Audit]
- [ ] **Add a "Warm-cache protocol" paragraph** [R1, R3]:
  - *cold:* empty cache, online write-back.
  - *warm:* the cache is seeded with the 2,650 train-split attacks and then frozen (no write-back during test), so no test prompt is in the cache before it is evaluated.
  - Add a near-duplicate analysis: results stratified by each test prompt's similarity to its nearest cached train attack, plus results on the "novel" subset (similarity < 0.80). [NEW RESULT: `cache_overlap.py`]
- [ ] **"selected using a separate 500-sample validation set"** → "using the neuralchemy validation split (941 prompts)". [R1]
- [ ] **ai4privacy:** "N = 209,261" → "209,261 rows (EN/FR/DE/IT) containing 133,564 ground-truth entities of 12 types; value-level matching". [Audit]
- [ ] **walledai/Multi-Turn-Jailbreak (200 conversations):** the old run never loaded this dataset. It fell back to 10 hand-written sessions repeated 20 times. Either run the real dataset and add benign multi-turn conversations to measure FPR [R1], or remove the multi-turn claim. [NEW RESULT]
- [ ] **deepset/prompt-injections:** the old "deepset" row was really neuralchemy test rows 500–942. Run on deepset itself (546 train + 116 test) or remove it. The only real deepset run so far gave F1 6.6%. [NEW RESULT] [Audit]
- [ ] **"a synthetic corpus covering violence and illegal activities"** → it was lmsys/toxic-chat (train split, 384 toxic / 1,616 benign, seed 42). Name it correctly. [Audit]
- [ ] **"benign subset of HuggingFaceH4/no_robots"** → the old run used tatsu-lab/alpaca (2,000 prompts) and measured **FPR 7.8%** (156/2,000), not 4.20%. Re-run on no_robots or report alpaca at 7.8%. [Audit]
- [ ] **Hardware:** "Intel Xeon E5-2686 v4, 8 vCPUs, 32 GB" → the machine that actually produced the results. All result files came from a Windows laptop (Intel Core Ultra 5 125H, 16 GB). Correct this unless you re-run on the Xeon. [Audit]
- [ ] **Baselines (§4.1)** [R1, R2]:
  - **Exact model IDs** for every API model, and the date of the runs.
  - **Footnote:** `gpt-4.0-2024-05-13`, `claude-3-5-haiku-20240307` and `gemini-2.5-flash-001` aren't valid IDs. The run script used **`openai/gpt-4o-mini`**, `anthropic/claude-3-5-haiku` and `google/gemini-2.5-flash` — confirm which models you actually ran.
  - **Prompt:** print the zero-shot prompt in an appendix (`run_baselines.py`, `ZERO_SHOT_PROMPT`). [R1]
  - **New baselines on the same 942 test prompts:**
    - Llama Prompt Guard 2 (86M)
    - ProtectAI DeBERTa-v3 prompt-injection v2
    - PIGuard (InjecGuard) [R1]
    - one reasoning model [R2 "RLM"]
    - **"LLM agents only"**: the two LLM agents in parallel, with no regex, deterministic detectors or cache [R2]
    - Llama Guard 4 on the same split
- [ ] **Metrics:** report FPR for **every** system, plus TPR at FPR = 1% / 5% for baselines that output a score. McNemar's test with **Holm–Bonferroni correction**, and paired bootstrap CIs. [R1]

## 6. Results (§5)

### 6.1 Table 2a and §5.1 baseline comparison

- [ ] Rebuild on the 942-prompt test split. Columns: N, P, R, F1, **FPR**, LLM-call rate, p50/p95 latency. Rows: Semantic Firewall (warm, frozen); Semantic Firewall (cold); all baselines from §5 above. [NEW RESULT] [R1, R2]
- [ ] Delete "41.93% of requests are resolved by the fast-path layers" and "χ² = 4.28, p = 0.038". Replace them with the new McNemar/Holm results. [NEW RESULT: `significance.py`] [R1]
- [ ] Delete "reducing effective cold-start latency to 459.85 ms, below that of the evaluated APIs". The only controlled latency measurement in the old results was about 2 s median for every configuration. Report measured p50/p95/p99 instead. (A code fix reduced the cache overhead from ~750 ms to ~30 ms per request, so new numbers will be much lower.) [R1]

### 6.2 Table 2b (cross-dataset) → rename to "Related tasks" [R1]

- [ ] **BeaverTails:**
  - Evaluate both systems on the **same** 2,000 prompts. The old run scored Semantic Firewall on 898 prompts and Llama Guard 4 on a different 2,000.
  - Add an FPR column; the old Semantic Firewall FPR was 67.8%.
  - Fix the Llama Guard 4 P/R: the file has 75.98 / 59.69, not 73.68 / 61.20. [NEW RESULT]
- [ ] **deepset row:** replace it with a real deepset run. Remove the Rebuff row unless Rebuff is actually run. [NEW RESULT]
- [ ] **ToxicChat (optional, if you keep a harmful-content comparison):** the old result was Semantic Firewall F1 49.6 vs Llama Guard 4 F1 51.2 on ToxicChat test. Don't drop unfavourable results selectively.

### 6.3 Table 2c (specialised evaluations)

- [ ] **PII row** — measured today with the current detector, on all 209,261 rows and all 12 types:
  - **Type-mapped protocol:** P **70.33**, R **75.27**, F1 **72.72**. A detection that equals a ground-truth value of a non-target PII type is counted as out of scope, not as a false positive.
  - **Legacy protocol** (every non-target detection counts as FP): P 53.78, R 75.31, F1 62.75. This is identical to the raw June result.
  - **Latency:** p50 **0.27 ms**, p99 **0.73 ms** per prompt on CPU.
  - Source: `results_camera_ready/pii/pii_eval.json`. Replace "85.90 / 96.70 / 91.00 on 209K" with these, and state the protocol.
- [ ] **PII per-type sentence** → "Structured identifiers reach ≥ 99.7% recall (email 99.9, credit card 99.7, IBAN 99.95, IPv4 99.8, IPv6 99.7, MAC 99.9, ZIP 99.9); free-form types are hard for regexes: phone 26.6%, DOB 33.2%, account number 37.0%, CVV 45.7%, SSN 50.2%." (The old 74.8% phone recall wasn't measured.)
- [ ] **Multi-turn row:** replace it with a real run that includes benign conversations for FPR [R1], or remove it. Note in the text that a regex-only configuration also scored 90% on the old synthetic sessions.
- [ ] **Safety row:** rename it to lmsys/toxic-chat and use measured values (old raw result: P 69.0, R 79.0, F1 73.7, FPR 7.8%). Re-run with the current code if you keep it.
- [ ] **Benign row:** alpaca FPR 7.8%, or a new no_robots run.

### 6.4 §5.2 and Table 3 (τ) [R1, R3]

- [ ] **Table 3a** → the validation-split sweep from `tune_threshold.py`. Columns: τ, P, R, F1, FPR, cache-hit rate, hit precision. F1 is now computed from P and R. (In the old table, F1 didn't match P/R in 4 of 5 rows — R1 checked τ = 0.95 and τ = 0.99.) [NEW RESULT]
- [ ] **Text:** delete "F1 peaks at τ = 0.99, which we adopt as the operating threshold". Replace it with "τ* = (value) maximises F1 on validation (ties → larger τ); we fix τ* and report test once". [R3, R1]
- [ ] **Table 3b ("OOD", ~98 items):** delete it. Its numbers match the project's own 98-item unit-test set (63 threats / 35 clean), which isn't out-of-distribution data. Replace it with the near-duplicate analysis from `cache_overlap.py`. [R1, R3, Audit]
- [ ] **"lower values ... risk misclassifying benign prompts, while higher values are more conservative"** → keep, but make sure the numbers follow this direction.

### 6.5 §5.3 and Table 4 (ablation) [R1, R2, R3]

- [ ] **Rebuild every row with correctly configured runs** (`run_system.py`). The old configs were mislabelled: "LLM Gate Only" had the LLM switched off; "All Fast (No LLM)" called the LLM on every prompt; Regex-only, Cache-only and Regex+Cache used identical settings. "Warm" also evaluated prompts that were already in the cache.
- [ ] **New rows:**
  - `regex_only`
  - `deterministic_only`
  - `cache_only` (warm)
  - `fast_layers` (regex + cache + detectors, no LLM)
  - `injection_llm_only`
  - `llm_agents_only` [R2]
  - `no_cache`
  - `no_early_exit`
  - `full_all_unresolved` (Algorithm 1 as written)
  - `full` (keyword gate)
  - per-detector leave-one-out `loo:<detector>` [R1]
- [ ] **Already measured** on test (942 prompts), current code, no LLM. Source: `results_camera_ready/runs/*/summary.json`.

  | Config | P | R | F1 | FPR | p50 |
  |---|---|---|---|---|---|
  | `regex_only` | 94.37 | 12.14 | 21.51 | 1.03% | 15.2 ms |
  | `deterministic_only` | 97.55 | 28.80 | 44.48 | 1.03% | 28.5 ms |

  The validation split for `deterministic_only` gives F1 39.59.
  **Point to make in the text:** the non-LLM layers are very precise (FPR ≈ 1%), but almost all recall comes from the LLM stage. Say this plainly; it's the honest version of "the components are complementary".
- [ ] **Text:** delete "improving F1 by 23.37 points over C1; McNemar χ² = 135.2" unless the new runs support it. Delete "both C4 and C6 recording 83.45 ms".

### 6.6 "Cost" paragraph → new subsection "Cost and latency at realistic attack rates" [R1]

- [ ] Replace the 66.88% / P_h, P_c, P_l / 444 ms / $500 → $166 text with a table at attack prevalence 1%, 5%, 10% and 50%. Columns: LLM-call rate, mean/p95/p99 latency, cost per 1M requests. [NEW RESULT: `prevalence_cost.py`]
- [ ] State the assumptions: price per million tokens, **2 LLM requests per LLM-stage invocation**, and tokens per request.
- [ ] Add one sentence noting that with the "all-unresolved" policy every benign prompt reaches the LLM, so savings depend on attack prevalence. That's exactly R1's point.

### 6.7 New subsection: security of the semantic memory [R1 item 4, R3 item 2]

- [ ] Run `cache_robustness.py all`. It's local and free (no API needed). Report:
  - **Error propagation:** induced FPR on unseen benign prompts after K benign false positives are cached (K = 1–200, τ = 0.85 / 0.90 / 0.95).
  - **Targeted poisoning:** poison = benign request + injection payload. Report the rate at which the benign request (and trivially edited copies) is then blocked, plus collateral blocking of other benign prompts.
  - **Allowlist abuse:** approved prompt + injection payload. Compare the share allowed under the old "bypass" behaviour with the new `skip_llm` mode, where the detectors still run.
  - **Growth and eviction:** lookup latency at 1k / 5k / 20k entries; the size cap is enforced.
- [ ] **Mitigations to state** (all implemented in code):
  - write-back only for LLM-confirmed, high-confidence detections
  - detector failures never cached
  - allowlist hits still pass the deterministic detectors
  - size cap with eviction
  - admin review of the allowlist

### 6.8 New subsection: adaptive attacks [R1 item 4, R2 item 4]

- [ ] **Offline rule-based mutation search** (synonyms, homoglyphs, spacing, encodings, framing) against the full pipeline and the fast layers. Report attack success rate at k = 1/5/10/20 queries. [NEW RESULT: `adaptive_attack.py --attacker rules`]
- [ ] **PAIR-style attacker** with a small open-weights model (default `meta-llama/llama-3.1-8b-instruct`) plus a judge model for goal preservation. Run with decision-only feedback and with explanation feedback, to quantify how much the explainability module helps an attacker. [NEW RESULT: `--attacker llm`]
- [ ] **Discuss injection against the LLM gate itself:** the gate reads attacker-controlled text. [R1]

### 6.9 New short subsection: evaluating the other components [R1 item 5]

- [ ] Per-detector leave-one-out (from §6.5).
- [ ] Session Judge: multi-turn attack sessions *and* benign multi-turn conversations, reporting recall and FPR.
- [ ] Policy profiles: one small table of strict / balanced / developer on the same test set, or a qualitative example.
- [ ] Explainability: one worked example of the JSON/Markdown report (qualitative is fine; R1 accepts that).

## 7. Limitations (§6)

- [ ] **"The 4.20% false-positive rate ... lower-threshold settings trade some accuracy for fewer false positives"** → this is reversed: *higher* τ gives fewer false positives. Use the new FPR. [R1, Audit]
- [ ] **Delete** "strong performance on the unseen BeaverTails dataset suggests the results reflect real detection ability rather than pretraining memorization". [R1]
- [ ] **"handles only about 45 requests per second"** → measure it or delete it. No measurement exists. [Audit]
- [ ] **Add:**
  - privacy of sending prompts to a third-party API, and raw-text storage in the cache [R1]
  - residual cache-poisoning risk [R3]
  - explanation leakage helping adaptive attackers [R1]
  - the regexes' low recall on paraphrased attacks
  - weak PII types (phone, DOB, SSN, CVV)
  - evaluation limited to direct (not indirect/RAG) injection

## 8. Conclusion

- [ ] Replace "92.64% F1 and 99.82% recall ... outperforming GPT-4.0, Gemini 2.5 Flash, and Claude 3.5 Haiku" and "resolves 66.88% of traffic ... cuts per-token API cost by roughly 3×" with the new test-split numbers and the prevalence-based cost result. [R1]
- [ ] **"Cross-dataset results ... confirm that the architecture generalizes"** → "Results on related tasks (harmful content, PII, multi-turn) show ...". [R1]

## 9. Figures, references, misc.

- [ ] **Fig. 1:**
  - Move Context Flooding to the deterministic group.
  - "Groq LLM API" → "OpenRouter (Llama-3.3-70B)".
  - Show the write-back policy box and the allowlist path.
  - "self-healing" → "self-updating".
  - Check that the circled numbers match §3 (R1: Fig. 1 vs Table 1 agents). [R1]
- [ ] **Model names in text and tables:** use the exact names of the models you ran (e.g. "GPT-4o-mini" if that's what `openai/gpt-4o-mini` was). Never write "GPT-4.0".
- [ ] **Ref [32]** (GPT-4o system card): keep it only if GPT-4o itself was evaluated.
- [ ] **Add a code/data availability statement** with the repo link (after the cleanup listed in §10). [R1 minor]
- [ ] **Generative-AI declaration:** keep as is.
- [ ] **Page limit:** check the ICISS camera-ready limit. Deleting Eq. 5, Table 3b and the long cache paragraph frees space for the threat model, cache security and adaptive-attack subsections.

---

## 10. Reviewer comment → where it is addressed

| Reviewer comment | Section of this list |
|---|---|
| R1.1 internal inconsistencies (83.93/87.59 pairing, τ 0.99 vs 0.90, Table 3a F1, Table 3b N, 41.93 vs 66.88, Eq. 7, FPR 4.20, latency 602.83, Groq/OpenRouter, model IDs, "six baselines", ±) | §0, 1, 4.4, 4.6, 5, 6.1, 6.4, 6.5, 6.6, 9 |
| R1.2 protocol: τ/θ on validation, warm-cache description, FPR for all, purpose-built baselines + prompts, significance with correction | §4.2, 4.4, 5, 6.1, 6.4 |
| R1.3 moderate claims: related tasks, multi-turn benign FPR, cost at 1–10% prevalence + p95/p99, BeaverTails memorization sentence | §1, 5, 6.2, 6.3, 6.6, 7 |
| R1.4 threat model, adaptive attacker, attacks on the gate, cache poisoning, allowlist abuse, growth/eviction, privacy | §4.7, 6.7, 6.8, 7 |
| R1.5 per-detector ablation, Session Judge / Policy / Explainability, Algorithm 1 alignment, Rebuff novelty, StruQ/SecAlign/CaMeL/instruction hierarchy | §3, 4.5, 6.5, 6.9 |
| R1.6 entropy formula, Fig. 1 vs Table 1, risk-weight justification, code release | §4.3, 4.4, 9 |
| R2.1 weak baselines; RLM / sub-agents; parallel agents without the deterministic part | §5 (baselines), 6.1, 6.5 (`llm_agents_only`) |
| R2.2 formulas | §4.4, 4.5 |
| R2.3 "self-healing" is inaccurate; describe the cache mechanism in depth | §1, 4.2 |
| R2.4 coordinated / optimised bypass, small open-weights attacker | §6.8 |
| R3.1 warm-cache protocol, memorisation of near-duplicates | §5, 6.4 |
| R3.2 cache error propagation and poisoning | §6.7 |
| R3.3 inconsistencies (τ 0.90 vs 0.99), parameter justification | §6.4, 4.4 |

---

## 11. For reference: code changes already made (2026-10-04)

All 205 tests pass. Before these changes: 179 passed, 12 failed.

- **Pipeline now matches the method above.**
  - Regex pre-screens always run; only the LLM call is gated.
  - Early exit when the deterministic stage blocks.
  - Allowlist `skip_llm` mode.
  - Configurable τ, cache path, cache on/off, write-back policy (LLM-confirmed + confidence floor), size cap with eviction.
  - Detector failures no longer cached.
  - Session score passed into the risk score.
  - Decisions record `resolution_stage`, `llm_called` and per-stage latency.
  - `SEMANTIC_FIREWALL_DISABLED_AGENTS` for leave-one-out ablations.
- **Latency:** the cache embeds each prompt once with a shared ONNX session. Per-request overhead dropped from ~750 ms to ~30 ms; median non-LLM path ≈ 46 ms.
- **LLM client:**
  - The default model ID was the Groq name `llama-3.3-70b-versatile`, which isn't valid on OpenRouter; it's now `meta-llama/llama-3.3-70b-instruct`.
  - Optional JSON mode.
  - Request timeout.
- **PII detector:** when several patterns match the same value, the most specific type wins (a card number is no longer labelled as a phone number).
- **Scripts fixed:**
  - removed the hardcoded "published" baseline rows (exp. 14/15)
  - gate-swap model variable (exp. 19)
  - multi-turn script fails loudly instead of silently using synthetic sessions
  - removed the hardcoded row in `run_nogate_neuralchemy.py`
  - removed the fake regex "LLM Guard / Rebuff / Guardrails" adapters
  - `.gitignore` repaired
- **New harness:** `camera_ready/` — `run_system.py`, `run_baselines.py`, `tune_threshold.py`, `cache_overlap.py`, `significance.py`, `prevalence_cost.py`, `cache_robustness.py`, `adaptive_attack.py`, `pii_eval.py`.
- **Still to do by you before releasing the repo.** Moving these was blocked by the tool's permission check, so they are untouched:
  - remove `benchmarks/evaluate_nemo_guardrails.py` (contains a comment saying the result was faked)
  - remove `benchmarks/evaluate_rebuff.py`
  - remove `results/tables/scale_pii.py` and `results/tables/revert_stats.py`
  - remove `experiments/07_chunked_ablation.py`, `07b_…`, `07c_…`, `08_generate_figures.py`, `10_cost_analysis.py`, `16_threshold_sensitivity.py`
  - remove `results/figures/` and the edited `results/tables/` CSVs
  - remove `data/var/chroma_db/` (4,284 cached entries from earlier evaluation runs)
