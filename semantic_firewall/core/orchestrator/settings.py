import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be one of {_TRUE_VALUES | _FALSE_VALUES}, got {raw!r}")


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _read_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


def load_calibration_pair(key: str, slope_default: float, bias_default: float) -> tuple[float, float]:
    return (
        _read_float(f"SEMANTIC_FIREWALL_CALIBRATION_{key}_SLOPE", slope_default),
        _read_float(f"SEMANTIC_FIREWALL_CALIBRATION_{key}_BIAS", bias_default),
    )


class AlertingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    webhook_url: str = ""
    slack_webhook_url: str = ""
    smtp_host: str = ""
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_user: str = ""
    smtp_password: str = ""
    email_from: str = ""
    email_to: str = ""
    min_severity: Literal["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"] = "HIGH"

    @classmethod
    def from_env(cls) -> "AlertingSettings":
        return cls(
            webhook_url=os.getenv("SEMANTIC_FIREWALL_ALERT_WEBHOOK_URL", ""),
            slack_webhook_url=os.getenv("SEMANTIC_FIREWALL_SLACK_WEBHOOK_URL", ""),
            smtp_host=os.getenv("SEMANTIC_FIREWALL_SMTP_HOST", ""),
            smtp_port=_read_int("SEMANTIC_FIREWALL_SMTP_PORT", 587),
            smtp_user=os.getenv("SEMANTIC_FIREWALL_SMTP_USER", ""),
            smtp_password=os.getenv("SEMANTIC_FIREWALL_SMTP_PASSWORD", ""),
            email_from=os.getenv("SEMANTIC_FIREWALL_ALERT_FROM", ""),
            email_to=os.getenv("SEMANTIC_FIREWALL_ALERT_TO", ""),
            min_severity=os.getenv("SEMANTIC_FIREWALL_ALERT_MIN_SEVERITY", "HIGH").upper(),
        )


class OrchestratorSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_max_size: int = Field(default=500, ge=1)
    cache_ttl_sec: float = Field(default=300.0, ge=0.0)
    similar_cache_enabled: bool = True
    similar_cache_min_chars: int = Field(default=40, ge=1)

    ensemble_enabled: bool = True
    ensemble_flag_threshold: float = Field(default=1.8, ge=0.0)
    ensemble_redact_threshold: float = Field(default=2.6, ge=0.0)
    ensemble_block_threshold: float = Field(default=3.5, ge=0.0)
    calibration_enabled: bool = True

    weight_pii: float = Field(default=1.0, gt=0.0)
    weight_secrets: float = Field(default=1.2, gt=0.0)
    weight_abuse: float = Field(default=1.0, gt=0.0)
    weight_injection: float = Field(default=1.3, gt=0.0)
    weight_unsafe_content: float = Field(default=1.3, gt=0.0)
    weight_threat_intel: float = Field(default=1.1, gt=0.0)
    weight_custom_rules: float = Field(default=1.0, gt=0.0)

    llm_gate_enabled: bool = True
    llm_gate_threshold: float = Field(default=1.0, ge=0.0)
    disable_llm_detectors: bool = False
    # Skip the LLM stage when the deterministic stage already returns BLOCK.
    early_exit_on_block: bool = True
    # Comma-separated agent names to leave out (used for per-detector ablations).
    disabled_agents: str = ""

    # Semantic memory (ChromaDB). Thresholds are cosine similarities in [0, 1].
    semantic_cache_enabled: bool = True
    semantic_cache_path: str = ""
    cache_similarity_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    allowlist_similarity_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    cache_writeback: bool = True
    cache_writeback_require_llm: bool = True
    cache_writeback_min_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    cache_max_entries: int = Field(default=10000, ge=1)
    # "skip_llm": an allowlist hit skips the threat cache and the LLM stage but the
    # deterministic detectors still run. "bypass": legacy behaviour, skips everything.
    allowlist_mode: Literal["skip_llm", "bypass"] = "skip_llm"

    audit_enabled: bool = True

    agent_timeout_default_sec: float = Field(default=8.0, ge=0.05)
    agent_timeout_pii_sec: float = Field(default=6.0, ge=0.05)
    agent_timeout_secrets_sec: float = Field(default=6.0, ge=0.05)
    agent_timeout_abuse_sec: float = Field(default=6.0, ge=0.05)
    agent_timeout_threat_intel_sec: float = Field(default=6.0, ge=0.05)
    agent_timeout_custom_rules_sec: float = Field(default=6.0, ge=0.05)
    agent_timeout_injection_sec: float = Field(default=8.0, ge=0.05)
    agent_timeout_unsafe_content_sec: float = Field(default=8.0, ge=0.05)

    @classmethod
    def from_env(cls) -> "OrchestratorSettings":
        return cls(
            cache_max_size=_read_int("SEMANTIC_FIREWALL_CACHE_MAX_SIZE", 500),
            cache_ttl_sec=_read_float("SEMANTIC_FIREWALL_CACHE_TTL_SEC", 300.0),
            similar_cache_enabled=_read_bool("SEMANTIC_FIREWALL_SIMILAR_CACHE_ENABLED", True),
            similar_cache_min_chars=_read_int("SEMANTIC_FIREWALL_SIMILAR_CACHE_MIN_CHARS", 40),
            ensemble_enabled=_read_bool("SEMANTIC_FIREWALL_ENSEMBLE_ENABLED", True),
            ensemble_flag_threshold=_read_float("SEMANTIC_FIREWALL_ENSEMBLE_FLAG_THRESHOLD", 1.8),
            ensemble_redact_threshold=_read_float("SEMANTIC_FIREWALL_ENSEMBLE_REDACT_THRESHOLD", 2.6),
            ensemble_block_threshold=_read_float("SEMANTIC_FIREWALL_ENSEMBLE_BLOCK_THRESHOLD", 3.5),
            calibration_enabled=_read_bool("SEMANTIC_FIREWALL_CALIBRATION_ENABLED", True),
            weight_pii=_read_float("SEMANTIC_FIREWALL_WEIGHT_PII", 1.0),
            weight_secrets=_read_float("SEMANTIC_FIREWALL_WEIGHT_SECRETS", 1.2),
            weight_abuse=_read_float("SEMANTIC_FIREWALL_WEIGHT_ABUSE", 1.0),
            weight_injection=_read_float("SEMANTIC_FIREWALL_WEIGHT_INJECTION", 1.3),
            weight_unsafe_content=_read_float("SEMANTIC_FIREWALL_WEIGHT_UNSAFE_CONTENT", 1.3),
            weight_threat_intel=_read_float("SEMANTIC_FIREWALL_WEIGHT_THREAT_INTEL", 1.1),
            weight_custom_rules=_read_float("SEMANTIC_FIREWALL_WEIGHT_CUSTOM_RULES", 1.0),
            llm_gate_enabled=_read_bool("SEMANTIC_FIREWALL_LLM_GATE_ENABLED", True),
            llm_gate_threshold=_read_float("SEMANTIC_FIREWALL_LLM_GATE_THRESHOLD", 1.0),
            disable_llm_detectors=_read_bool("SEMANTIC_FIREWALL_DISABLE_LLM_DETECTORS", False),
            early_exit_on_block=_read_bool("SEMANTIC_FIREWALL_EARLY_EXIT_ON_BLOCK", True),
            disabled_agents=os.getenv("SEMANTIC_FIREWALL_DISABLED_AGENTS", ""),
            semantic_cache_enabled=_read_bool("SEMANTIC_FIREWALL_SEMANTIC_CACHE_ENABLED", True),
            semantic_cache_path=os.getenv("SEMANTIC_FIREWALL_SEMANTIC_CACHE_PATH", ""),
            cache_similarity_threshold=_read_float("SEMANTIC_FIREWALL_CACHE_SIM_THRESHOLD", 0.90),
            allowlist_similarity_threshold=_read_float("SEMANTIC_FIREWALL_ALLOWLIST_SIM_THRESHOLD", 0.90),
            cache_writeback=_read_bool("SEMANTIC_FIREWALL_CACHE_WRITEBACK", True),
            cache_writeback_require_llm=_read_bool("SEMANTIC_FIREWALL_CACHE_WRITEBACK_REQUIRE_LLM", True),
            cache_writeback_min_confidence=_read_float("SEMANTIC_FIREWALL_CACHE_WRITEBACK_MIN_CONFIDENCE", 0.85),
            cache_max_entries=_read_int("SEMANTIC_FIREWALL_CACHE_MAX_ENTRIES", 10000),
            allowlist_mode=os.getenv("SEMANTIC_FIREWALL_ALLOWLIST_MODE", "skip_llm").strip().lower(),
            audit_enabled=_read_bool("SEMANTIC_FIREWALL_AUDIT_ENABLED", True),
            agent_timeout_default_sec=_read_float("SEMANTIC_FIREWALL_AGENT_TIMEOUT_DEFAULT_SEC", 8.0),
            agent_timeout_pii_sec=_read_float("SEMANTIC_FIREWALL_AGENT_TIMEOUT_PII_SEC", 6.0),
            agent_timeout_secrets_sec=_read_float("SEMANTIC_FIREWALL_AGENT_TIMEOUT_SECRETS_SEC", 6.0),
            agent_timeout_abuse_sec=_read_float("SEMANTIC_FIREWALL_AGENT_TIMEOUT_ABUSE_SEC", 6.0),
            agent_timeout_threat_intel_sec=_read_float("SEMANTIC_FIREWALL_AGENT_TIMEOUT_THREAT_INTEL_SEC", 6.0),
            agent_timeout_custom_rules_sec=_read_float("SEMANTIC_FIREWALL_AGENT_TIMEOUT_CUSTOM_RULES_SEC", 6.0),
            agent_timeout_injection_sec=_read_float("SEMANTIC_FIREWALL_AGENT_TIMEOUT_INJECTION_SEC", 8.0),
            agent_timeout_unsafe_content_sec=_read_float("SEMANTIC_FIREWALL_AGENT_TIMEOUT_UNSAFE_CONTENT_SEC", 8.0),
        )
