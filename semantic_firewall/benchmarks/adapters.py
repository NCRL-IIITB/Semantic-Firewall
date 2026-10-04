import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from semantic_firewall.core.orchestrator.orchestrator import SemanticFirewallOrchestrator


@dataclass
class ToolResult:
    tool: str
    predicted_label: str  # "threat" or "clean"
    latency_ms: float
    error: Optional[str] = None
    meta: dict[str, Any] | None = None


class ToolAdapter:
    name = "unknown"

    def available(self) -> bool:
        return True

    def predict(self, text: str) -> ToolResult:
        raise NotImplementedError


class SemanticFirewallAdapter(ToolAdapter):
    name = "Semantic Firewall"

    def __init__(self, policy_profile: str = "balanced"):
        self.orchestrator = SemanticFirewallOrchestrator()
        self.policy_profile = policy_profile
        self.name = f"Semantic Firewall ({policy_profile})"

    def predict(self, text: str) -> ToolResult:
        start = time.perf_counter()
        decision = self.orchestrator.analyze(
            text=text,
            policy_profile=self.policy_profile,
            workspace_id="default",
        )
        latency = (time.perf_counter() - start) * 1000
        predicted = "threat" if decision.action != "ALLOW" else "clean"
        llm_detector_stats = {}
        timeout_events = 0
        for agent_result in decision.agent_results:
            if agent_result.agent_name not in {"Injection Detector", "Unsafe Content Detector"}:
                if (agent_result.meta or {}).get("error_type") == "timeout":
                    timeout_events += 1
                continue
            meta = agent_result.meta or {}
            llm_detector_stats[agent_result.agent_name] = {
                "llm_called": bool(meta.get("llm_called", False)),
                "regex_only": bool(meta.get("regex_only", False)),
                "llm_skipped_reason": meta.get("llm_skipped_reason"),
                "skipped_by_orchestrator_gate": bool(meta.get("skipped_by_orchestrator_gate", False)),
            }
            if meta.get("error_type") == "timeout":
                timeout_events += 1
        return ToolResult(
            tool=self.name,
            predicted_label=predicted,
            latency_ms=latency,
            meta={
                "llm_detectors": llm_detector_stats,
                "timeout_events": timeout_events,
                "degraded": bool(getattr(decision, "degraded", False)),
            },
        )


class RegexHeuristicBaselineAdapter(ToolAdapter):
    name = "Regex Heuristic Baseline"

    def __init__(self):
        import re

        self._regex = re.compile(
            r"(ignore\s+previous|system\s+prompt|jailbreak|AKIA[0-9A-Z]{16}|"
            r"sk_live_[A-Za-z0-9]{16,}|password\s*=|bomb|methamphetamine|"
            r"\b[A-Z]{5}[0-9]{4}[A-Z]\b|\b\d{4}\s?\d{4}\s?\d{4}\b)",
            re.IGNORECASE,
        )

    def predict(self, text: str) -> ToolResult:
        start = time.perf_counter()
        matched = bool(self._regex.search(text))
        latency = (time.perf_counter() - start) * 1000
        return ToolResult(
            tool=self.name,
            predicted_label="threat" if matched else "clean",
            latency_ms=latency,
        )


class OptionalImportAdapter(ToolAdapter):
    def __init__(self, name: str, import_name: str, predictor_fn: Callable[[str], bool]):
        import importlib

        self.name = name
        self.import_name = import_name
        self.predictor_fn = predictor_fn
        self._available = importlib.util.find_spec(import_name) is not None

    def available(self) -> bool:
        return self._available

    def predict(self, text: str) -> ToolResult:
        if not self._available:
            return ToolResult(tool=self.name, predicted_label="clean", latency_ms=0.0, error="not_installed")
        start = time.perf_counter()
        try:
            is_threat = bool(self.predictor_fn(text))
            latency = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool=self.name,
                predicted_label="threat" if is_threat else "clean",
                latency_ms=latency,
            )
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            return ToolResult(tool=self.name, predicted_label="clean", latency_ms=latency, error=str(exc))


def build_optional_market_adapters() -> list[ToolAdapter]:
    """Adapters for third-party guardrail libraries.

    Only real library calls belong here. The previous versions of these adapters ran
    small regexes and reported them under the product names (LLM Guard, Rebuff,
    Guardrails AI), which is not a valid comparison, so they were removed. Add an
    adapter back only once it calls the library's own scanner.
    """
    return []

