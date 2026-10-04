import shutil
import sys
import os
from pathlib import Path
from uuid import uuid4

import pytest


def pytest_configure():
    os.environ["GROQ_API_KEY"] = ""
    os.environ["OPENROUTER_API_KEY"] = ""
    os.environ["SEMANTIC_FIREWALL_DISABLE_LLM_DETECTORS"] = "0"
    os.environ["SEMANTIC_FIREWALL_LLM_GATE_ENABLED"] = "0"
    # Never read or write the repository's persistent semantic cache from tests.
    os.environ["SEMANTIC_FIREWALL_SEMANTIC_CACHE_PATH"] = str(
        Path(__file__).resolve().parent / ".tmp" / "chroma_session"
    )


@pytest.fixture
def workspace_tmp_path():
    path = Path(__file__).resolve().parent / ".tmp" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def isolate_legacy_temp_artifacts(workspace_tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("SEMANTIC_FIREWALL_DISABLE_LLM_DETECTORS", "0")
    monkeypatch.setenv("SEMANTIC_FIREWALL_LLM_GATE_ENABLED", "0")

    for module_name in ("tests.test_firewall_safety", "tests.test_threat_intel"):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "TMP_DIR"):
            monkeypatch.setattr(module, "TMP_DIR", workspace_tmp_path)


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(Path(__file__).resolve().parent / ".tmp", ignore_errors=True)
