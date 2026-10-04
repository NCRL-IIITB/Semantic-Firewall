<div align="center">
  <h1>🛡️ Semantic Firewall</h1>
  <p><b>A Hybrid Defense-in-Depth Architecture for LLM Security</b></p>

  <a href="https://github.com/NCRL-IIITB/Semantic_Firewall/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10+-success.svg" alt="Python 3.10+"></a>
</div>

<br/>

> **Abstract:** Current LLM guardrails rely on routing every user prompt through a dedicated safety LLM, introducing prohibitive latency and cost. The **Semantic Firewall** replaces this with a highly parallelized, hybrid pipeline that intercepts structurally predictable adversarial attacks early, reserving deep LLM evaluation only for complex, novel zero-day threats. 

---

## 🏗️ Architecture

The firewall implements a rigorous defense-in-depth pipeline consisting of four cascading layers:

1. **Semantic memory:** a ChromaDB vector cache (all-MiniLM-L6-v2, cosine). Prompts within similarity τ of a confirmed threat are blocked; an admin allowlist holds approved false positives. Write-back is limited to LLM-confirmed, high-confidence detections, and the cache is size-capped.
2. **Deterministic stage:** regex pre-screens (injection, unsafe content, threat-intel signatures) run in parallel with the PII, secrets, abuse, context-flooding and custom-rule detectors. If this stage already blocks, the LLM is not called.
3. **LLM stage:** Llama-3.3-70B-Instruct (via OpenRouter, temperature 0) returns a JSON verdict from the injection and unsafe-content agents. It runs only for prompts the earlier stages did not resolve, and fails closed if unavailable.
4. **Policy and explainability:** per-detector policy actions (ALLOW/FLAG/REDACT/BLOCK) with ensemble escalation, configurable profiles, redaction, audit logging and human-readable explanations.

---

## 🚀 Quickstart & Installation

**Prerequisites:**
- Python 3.10+
- 16GB+ RAM (for local ChromaDB embedding generation)
- API Keys (OpenAI/Anthropic/OpenRouter) for the baseline evaluation and LLM Gate fallback.

```bash
# 1. Clone the repository
git clone https://github.com/NCRL-IIITB/Semantic-Firewall.git
cd Semantic-Firewall

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup your environment keys
cp .env.example .env
# Edit .env with your API keys
```

### Running Benchmarks
Datasets are downloaded from Hugging Face on first use. The evaluation protocol and all scripts are described in
[`semantic_firewall/benchmarks/research/README.md`](semantic_firewall/benchmarks/research/README.md).

```bash
cd semantic_firewall/benchmarks/research/camera_ready

# Full pipeline on the neuralchemy test split (warm cache seeded from the train split only)
python run_system.py --config full --split test --cache warm_frozen

# Ablations
python run_system.py --config regex_only --split test --cache none
python run_system.py --config cache_only --split test --cache warm_frozen
python run_system.py --config llm_agents_only --split test --cache none
```

---

## 📊 Evaluation Datasets

The evaluation utilizes the following public HuggingFace datasets to ensure broad, out-of-distribution coverage:

- **[neuralchemy/Prompt-injection-dataset](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset)**: Primary evaluation corpus containing a balanced mix of benign queries and complex injections.
- **[ai4privacy/pii-masking-200k](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)**: Large multilingual corpus with embedded PII entities for scaled detection testing.
- **[walledai/Multi-Turn-Jailbreak](https://huggingface.co/datasets/walledai/Multi-Turn-Jailbreak)**: Adversarial conversational sessions for session-level recall testing.
- **[PKU-Alignment/BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails)**: Safety benchmark spanning 14 harm categories for out-of-distribution generalization.
- **[deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections)**: Used for direct head-to-head evaluation against baseline heuristic pipelines.
- **[HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)**: Benign corpus used for false positive rate stress testing.

---

## 📝 Citation

*This paper is currently under peer review. A formal BibTeX citation will be added upon acceptance and publication.*
