# Contributing

Thanks for contributing to Semantic Firewall.

## 1) Local Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install --upgrade pip
pip install -e .[dev]
```

Optional `.env` values for full detector behavior:

- `OPENROUTER_API_KEY` (LLM-assisted detectors; see `.env.example`)
- Alerting variables (see README)

## 2) Run The Project

API:

```bash
python -m uvicorn semantic_firewall.apps.api_server:app --host 0.0.0.0 --port 8000
```

UI:

```bash
streamlit run semantic_firewall/apps/dashboard.py
```

## 3) Run Tests Before PR

Quick smoke (Windows):

```powershell
.\semantic_firewall\tools\smoke_test.ps1 -SkipPytest
```

Full tests:

```bash
pytest -q
```

Focused safety suites:

```bash
pytest -q tests/test_injection_detector.py tests/test_unsafe_content_detector.py tests/test_firewall_safety.py tests/test_canary_prompts.py
```

## 4) Benchmark + Tuning Workflow

```bash
python semantic_firewall/tools/run_tuning_cycle.py --quiet
```

Generated outputs are written to `data/results/`.

## 5) Code Organization Rules

- Add new API endpoints in `semantic_firewall/api/routes/`.
- Add request/response models in `semantic_firewall/api/schemas.py`.
- Add UI pages in `semantic_firewall/ui/pages/`.
- Add detector logic in `semantic_firewall/core/agents/` and orchestration logic in `semantic_firewall/core/orchestrator/`.
- Put helper/maintenance scripts in `semantic_firewall/tools/`.

## 6) Pull Request Checklist

- Tests pass locally (`pytest -q`).
- New behavior has tests.
- README/docs updated for new flags, scripts, or endpoints.
- No unrelated generated artifacts committed from local runs.

