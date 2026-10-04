import hashlib
import re
import threading
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from semantic_firewall.core.agents.abuse_detector import AbuseDetectorAgent
from semantic_firewall.core.agents.custom_rules_detector import CustomRulesDetectorAgent
from semantic_firewall.core.agents.injection_detector import InjectionDetectorAgent
from semantic_firewall.core.agents.pii_detector import PIIDetectorAgent
from semantic_firewall.core.agents.secrets_detector import SecretsDetectorAgent
from semantic_firewall.core.agents.threat_intel_detector import ThreatIntelDetectorAgent
from semantic_firewall.core.agents.unsafe_content_detector import UnsafeContentDetectorAgent
from semantic_firewall.core.agents.dos_detector import ContextFloodingDetectorAgent
from semantic_firewall.core.orchestrator.audit_logger import AuditLogger
from semantic_firewall.core.orchestrator.policy_store import PolicyStore
from semantic_firewall.core.orchestrator.session_store import SessionStore
from semantic_firewall.core.orchestrator.session_judge import SessionJudge
from semantic_firewall.core.orchestrator.risk_scorer import RiskScorer
from semantic_firewall.core.orchestrator.explainability import ExplainabilityGenerator, ExplainabilityReport
from semantic_firewall.core.orchestrator.alerting import AlertManager
from semantic_firewall.core.orchestrator.compliance import ComplianceProfileManager
from semantic_firewall.core.orchestrator.session_patterns import MultiTurnAttackDetector, EnhancedSessionStore
from semantic_firewall.core.orchestrator.settings import OrchestratorSettings, load_calibration_pair
from semantic_firewall.core.orchestrator.paths import var_path
from semantic_firewall.core.orchestrator.semantic_cache import SemanticCache


@dataclass
class AgentResult:
    agent_name: str
    threat_found: bool
    threat_type: str
    severity: str
    summary: str
    matched: list
    agent_available: bool = True
    fail_closed: bool = False
    meta: dict = field(default_factory=dict)


@dataclass
class FirewallDecision:
    action: str
    reason: str
    severity: str
    triggered_agents: List[str]
    agent_results: List[AgentResult]
    redacted_text: str
    processing_time_ms: float
    original_text: str
    scan_target: str = "input"
    from_cache: bool = False
    degraded: bool = False
    unavailable_agents: List[str] = field(default_factory=list)
    policy_profile: str = "balanced"
    risk_score: float = 0.0
    risk_level: str = "NONE"
    explanation: Optional[ExplainabilityReport] = None
    detected_patterns: List[dict] = field(default_factory=list)
    adversary_tier: str = "NONE"
    # Where the request was resolved: exact_cache | allowlist | semantic_cache | deterministic | llm
    resolution_stage: str = "deterministic"
    llm_called: bool = False
    stage_latency_ms: Dict[str, float] = field(default_factory=dict)
    allowlist_hit: bool = False
    semantic_similarity: Optional[float] = None


@dataclass
class InteractionDecision:
    prompt_decision: FirewallDecision
    output_decision: FirewallDecision
    combined_action: str
    combined_severity: str
    combined_reason: str
    triggered_agents: List[str]
    interaction_alerts: List[dict] = field(default_factory=list)
    drift_detected: bool = False
    contradiction_detected: bool = False
    session_risk_score: float = 0.0


@dataclass
class CacheEntry:
    decision: FirewallDecision
    created_at: float


class SemanticFirewallOrchestrator:
    def __init__(self, db_path: str | None = None):
        print("[Orchestrator] Initializing all agents...")
        settings = OrchestratorSettings.from_env()
        self.settings = settings
        resolved_db_path = db_path or str(var_path("audit.db"))
        all_agents = {
            "Context Flooding Detector": ContextFloodingDetectorAgent(max_chars=4000),
            "PII Detector": PIIDetectorAgent(),
            "Secrets Detector": SecretsDetectorAgent(),
            "Abuse Detector": AbuseDetectorAgent(),
            "Injection Detector": InjectionDetectorAgent(),
            "Unsafe Content Detector": UnsafeContentDetectorAgent(),
            "Threat Intel Detector": ThreatIntelDetectorAgent(),
            "Custom Rules Detector": CustomRulesDetectorAgent(),
        }
        disabled = {name.strip() for name in settings.disabled_agents.split(",") if name.strip()}
        unknown = disabled - set(all_agents)
        if unknown:
            raise ValueError(f"SEMANTIC_FIREWALL_DISABLED_AGENTS has unknown agent names: {sorted(unknown)}")
        self.agents = {name: agent for name, agent in all_agents.items() if name not in disabled}
        self.disabled_agents = sorted(disabled)
        self.fail_closed_agents = {
            "Injection Detector",
            "Unsafe Content Detector",
        }
        self.llm_agents = {
            "Injection Detector",
            "Unsafe Content Detector",
        }

        self.policy: Dict[str, Dict[str, str]] = {
            "PII": {
                "CRITICAL": "BLOCK",
                "HIGH": "REDACT",
                "MEDIUM": "REDACT",
                "LOW": "FLAG",
                "NONE": "ALLOW",
            },
            "SECRET": {
                "CRITICAL": "BLOCK",
                "HIGH": "BLOCK",
                "MEDIUM": "REDACT",
                "LOW": "FLAG",
                "NONE": "ALLOW",
            },
            "ABUSE": {
                "CRITICAL": "BLOCK",
                "HIGH": "BLOCK",
                "MEDIUM": "FLAG",
                "LOW": "FLAG",
                "NONE": "ALLOW",
            },
            "INJECTION": {
                "CRITICAL": "BLOCK",
                "HIGH": "BLOCK",
                "MEDIUM": "BLOCK",
                "LOW": "FLAG",
                "NONE": "ALLOW",
            },
            "UNSAFE_CONTENT": {
                "CRITICAL": "BLOCK",
                "HIGH": "BLOCK",
                "MEDIUM": "FLAG",
                "LOW": "FLAG",
                "NONE": "ALLOW",
            },
            "DOS": {
                "CRITICAL": "BLOCK",
                "HIGH": "BLOCK",
                "MEDIUM": "BLOCK",
                "LOW": "FLAG",
                "NONE": "ALLOW",
            },
            "SYSTEM_UNAVAILABLE": {
                "CRITICAL": "BLOCK",
                "HIGH": "FLAG",
                "MEDIUM": "FLAG",
                "LOW": "FLAG",
                "NONE": "ALLOW",
            },
            "CUSTOM_RULE": {
                "CRITICAL": "BLOCK",
                "HIGH": "BLOCK",
                "MEDIUM": "REDACT",
                "LOW": "FLAG",
                "NONE": "ALLOW",
            },
            "THREAT_INTEL": {
                "CRITICAL": "BLOCK",
                "HIGH": "BLOCK",
                "MEDIUM": "FLAG",
                "LOW": "FLAG",
                "NONE": "ALLOW",
            },
        }
        self.action_priority = ["ALLOW", "FLAG", "REDACT", "BLOCK"]
        self.session_store = SessionStore()
        self.audit_logger = AuditLogger(db_path=resolved_db_path)
        self.policy_store = PolicyStore()
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_max_size = settings.cache_max_size
        self._cache_ttl_sec = settings.cache_ttl_sec
        self._similar_cache_enabled = settings.similar_cache_enabled
        self._similar_cache_min_chars = settings.similar_cache_min_chars
        self._cache_lock = threading.Lock()
        self.ensemble_enabled = settings.ensemble_enabled
        self.ensemble_flag_threshold = settings.ensemble_flag_threshold
        self.ensemble_redact_threshold = settings.ensemble_redact_threshold
        self.ensemble_block_threshold = settings.ensemble_block_threshold
        self.calibration_enabled = settings.calibration_enabled
        self._calibration_params = {
            "DEFAULT": self._calibration_pair_from_env("DEFAULT", 1.0, 0.0),
            "PII": self._calibration_pair_from_env("PII", 1.0, 0.0),
            "SECRET": self._calibration_pair_from_env("SECRET", 1.0, 0.0),
            "ABUSE": self._calibration_pair_from_env("ABUSE", 1.0, 0.0),
            "INJECTION": self._calibration_pair_from_env("INJECTION", 1.05, -0.02),
            "UNSAFE_CONTENT": self._calibration_pair_from_env("UNSAFE_CONTENT", 1.05, -0.02),
            "THREAT_INTEL": self._calibration_pair_from_env("THREAT_INTEL", 1.0, 0.0),
            "CUSTOM_RULE": self._calibration_pair_from_env("CUSTOM_RULE", 1.0, 0.0),
        }
        self._ensemble_weights = {
            "PII Detector": settings.weight_pii,
            "Secrets Detector": settings.weight_secrets,
            "Abuse Detector": settings.weight_abuse,
            "Injection Detector": settings.weight_injection,
            "Unsafe Content Detector": settings.weight_unsafe_content,
            "Threat Intel Detector": settings.weight_threat_intel,
            "Custom Rules Detector": settings.weight_custom_rules,
            "Context Flooding Detector": 1.0,
        }
        self.llm_gate_enabled = settings.llm_gate_enabled
        self.llm_gate_threshold = settings.llm_gate_threshold
        self.disable_llm_detectors = settings.disable_llm_detectors
        self.early_exit_on_block = settings.early_exit_on_block
        self.allowlist_mode = settings.allowlist_mode
        self.audit_enabled = settings.audit_enabled
        self.agent_timeouts = {
            "default": settings.agent_timeout_default_sec,
            "PII Detector": settings.agent_timeout_pii_sec,
            "Secrets Detector": settings.agent_timeout_secrets_sec,
            "Abuse Detector": settings.agent_timeout_abuse_sec,
            "Threat Intel Detector": settings.agent_timeout_threat_intel_sec,
            "Custom Rules Detector": settings.agent_timeout_custom_rules_sec,
            "Injection Detector": settings.agent_timeout_injection_sec,
            "Unsafe Content Detector": settings.agent_timeout_unsafe_content_sec,
            "Context Flooding Detector": 0.5,
        }

        # Initialize production feature modules
        print("[Orchestrator] Initializing production modules...")
        self.risk_scorer = RiskScorer()
        self.explainability_generator = ExplainabilityGenerator()
        self.alert_manager = AlertManager.from_env()
        self.compliance_manager = ComplianceProfileManager()
        self.multi_turn_detector = MultiTurnAttackDetector()
        self.enhanced_session_store = EnhancedSessionStore()
        self.session_judge = SessionJudge(
            session_store=self.session_store,
            enhanced_session_store=self.enhanced_session_store,
            multi_turn_detector=self.multi_turn_detector,
            injection_agent_getter=lambda: self.agents["Injection Detector"],
            action_rank=self._action_rank,
        )
        self.semantic_cache = SemanticCache(
            db_path=settings.semantic_cache_path or str(var_path("chroma_db")),
            enabled=settings.semantic_cache_enabled,
            similarity_threshold=settings.cache_similarity_threshold,
            allowlist_threshold=settings.allowlist_similarity_threshold,
            writeback=settings.cache_writeback,
            writeback_require_llm=settings.cache_writeback_require_llm,
            writeback_min_confidence=settings.cache_writeback_min_confidence,
            max_entries=settings.cache_max_entries,
        )
        print("[Orchestrator] All production modules ready.")

        print("[Orchestrator] All agents ready.\n")

    def _calibration_pair_from_env(self, key: str, slope_default: float, bias_default: float) -> tuple[float, float]:
        return load_calibration_pair(key, slope_default, bias_default)

    def _agent_timeout(self, name: str) -> float:
        return max(0.05, float(self.agent_timeouts.get(name, self.agent_timeouts["default"])))

    def _severity_rank(self, severity: str) -> int:
        severity_order = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        try:
            return severity_order.index(severity)
        except ValueError:
            return 0

    def _action_rank(self, action: str) -> int:
        try:
            return self.action_priority.index(action)
        except ValueError:
            return 0

    def _cache_key(
        self,
        text: str,
        scan_target: str = "input",
        policy_profile: str = "balanced",
        workspace_id: str = "default",
    ) -> str:
        return hashlib.md5(f"{scan_target}:{policy_profile}:{workspace_id}:{text.strip()}".encode()).hexdigest()

    def _similar_cache_key(
        self,
        text: str,
        scan_target: str = "input",
        policy_profile: str = "balanced",
        workspace_id: str = "default",
    ) -> Optional[str]:
        if not self._similar_cache_enabled:
            return None
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        if len(normalized) < self._similar_cache_min_chars:
            return None
        return hashlib.md5(f"similar:{scan_target}:{policy_profile}:{workspace_id}:{normalized}".encode()).hexdigest()

    def _prune_cache_locked(self, now: float):
        if self._cache_ttl_sec <= 0:
            self._cache.clear()
            return

        expired = [
            key
            for key, entry in self._cache.items()
            if (now - entry.created_at) > self._cache_ttl_sec
        ]
        for key in expired:
            del self._cache[key]

        while len(self._cache) > self._cache_max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]

    def _get_from_cache(
        self,
        text: str,
        scan_target: str = "input",
        policy_profile: str = "balanced",
        workspace_id: str = "default",
    ) -> Optional[FirewallDecision]:
        now = time.time()
        exact_key = self._cache_key(text, scan_target, policy_profile, workspace_id)
        similar_key = self._similar_cache_key(text, scan_target, policy_profile, workspace_id)
        with self._cache_lock:
            self._prune_cache_locked(now)
            cached_entry = self._cache.get(exact_key)
            if cached_entry is None and similar_key:
                cached_entry = self._cache.get(similar_key)
        if not cached_entry:
            return None

        cached_copy = deepcopy(cached_entry.decision)
        cached_copy.from_cache = True
        cached_copy.processing_time_ms = 0.0
        return cached_copy

    def _save_to_cache(
        self,
        text: str,
        decision: FirewallDecision,
        scan_target: str = "input",
        policy_profile: str = "balanced",
        workspace_id: str = "default",
    ):
        now = time.time()
        exact_key = self._cache_key(text, scan_target, policy_profile, workspace_id)
        similar_key = self._similar_cache_key(text, scan_target, policy_profile, workspace_id)
        entry = CacheEntry(decision=decision, created_at=now)
        with self._cache_lock:
            self._prune_cache_locked(now)
            self._cache[exact_key] = entry
            if similar_key:
                self._cache[similar_key] = entry
            self._prune_cache_locked(now)

    def _run_agent(
        self,
        name: str,
        agent,
        text: str,
        scan_target: str = "input",
        workspace_id: str = "default",
        detector_threshold_overrides: Optional[Dict[str, Dict[str, float]]] = None,
        llm_mode: bool = True,
    ) -> AgentResult:
        try:
            timeout_seconds = self._agent_timeout(name)
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(
                    self._invoke_agent,
                    name,
                    agent,
                    text,
                    scan_target,
                    workspace_id,
                    detector_threshold_overrides,
                    llm_mode,
                )
                result = future.result(timeout=timeout_seconds)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            return AgentResult(
                agent_name=result.agent_name,
                threat_found=result.threat_found,
                threat_type=result.threat_type,
                severity=result.severity,
                summary=result.summary,
                matched=result.matched,
                agent_available=True,
                fail_closed=name in self.fail_closed_agents,
                meta=getattr(result, "meta", {}) or {},
            )
        except FuturesTimeoutError:
            print(f"[Orchestrator] Agent '{name}' timed out after {self._agent_timeout(name):.2f}s")
            fail_closed = name in self.fail_closed_agents
            return AgentResult(
                agent_name=name,
                threat_found=fail_closed,
                threat_type="SYSTEM_UNAVAILABLE",
                severity="HIGH" if fail_closed else "LOW",
                summary=(
                    f"Detector unavailable: {name} timed out."
                    + (" Fail-closed policy applied." if fail_closed else " Review recommended.")
                ),
                matched=[],
                agent_available=False,
                fail_closed=fail_closed,
                meta={"error_type": "timeout"},
            )
        except Exception as exc:
            print(f"[Orchestrator] Agent '{name}' failed: {exc}")
            fail_closed = name in self.fail_closed_agents
            return AgentResult(
                agent_name=name,
                threat_found=fail_closed,
                threat_type="SYSTEM_UNAVAILABLE",
                severity="HIGH" if fail_closed else "LOW",
                summary=(
                    f"Detector unavailable: {name} failed."
                    + (" Fail-closed policy applied." if fail_closed else " Review recommended.")
                ),
                matched=[],
                agent_available=False,
                fail_closed=fail_closed,
                meta={"error_type": "exception"},
            )

    def _invoke_agent(
        self,
        name: str,
        agent,
        text: str,
        scan_target: str,
        workspace_id: str,
        detector_threshold_overrides: Optional[Dict[str, Dict[str, float]]] = None,
        llm_mode: bool = True,
    ):
        detector_threshold_overrides = detector_threshold_overrides or {}
        detector_config = detector_threshold_overrides.get(name, {})
        # Only real LLM detectors understand use_llm; test stubs and older agents do not.
        llm_kwargs = {"use_llm": llm_mode} if getattr(agent, "supports_regex_only", False) else {}
        if name == "Custom Rules Detector":
            return agent.run(text, scan_target=scan_target, workspace_id=workspace_id)
        if name == "Threat Intel Detector":
            return agent.run(text, scan_target=scan_target)
        if name == "Injection Detector":
            return agent.run(
                text,
                scan_target=scan_target,
                confidence_threshold_override=detector_config.get("confidence_threshold"),
                **llm_kwargs,
            )
        if name == "Unsafe Content Detector":
            return agent.run(
                text,
                confidence_threshold_override=detector_config.get("confidence_threshold"),
                **llm_kwargs,
            )
        return agent.run(text)

    def _run_agents_parallel(
        self,
        agent_names: List[str],
        text: str,
        scan_target: str,
        workspace_id: str,
        detector_threshold_overrides: Optional[Dict[str, Dict[str, float]]] = None,
        llm_mode: bool = True,
    ) -> List[AgentResult]:
        if not agent_names:
            return []
        results: List[AgentResult] = []
        with ThreadPoolExecutor(max_workers=len(agent_names)) as executor:
            futures = {
                executor.submit(
                    self._run_agent,
                    name,
                    self.agents[name],
                    text,
                    scan_target,
                    workspace_id,
                    detector_threshold_overrides,
                    llm_mode,
                ): name
                for name in agent_names
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                status = "THREAT" if result.threat_found else "CLEAN"
                print(f"  [{status}] {result.agent_name}: {result.summary}")
        return results

    def _llm_gate_score(self, text: str, cheap_results: List[AgentResult]) -> float:
        score = 0.0
        if len(text) >= 800:
            score += 1.5

        if any(result.threat_found for result in cheap_results):
            score += 2.0

        suspicious_pattern = re.search(
            r"(ignore\s+previous|system\s+prompt|jailbreak|do\s+anything\s+now|bypass|override|"
            r"password|api\s*key|token|secret|bomb|kill|harm)",
            text,
            re.IGNORECASE,
        )
        if suspicious_pattern:
            score += 1.0

        return score

    def _should_run_llm_agents(self, text: str, cheap_results: List[AgentResult]) -> tuple[bool, float]:
        if self.disable_llm_detectors:
            return False, 0.0
        if not self.llm_gate_enabled:
            return True, self.llm_gate_threshold
        score = self._llm_gate_score(text, cheap_results)
        return score >= self.llm_gate_threshold, score

    def _detector_threshold_overrides(
        self,
        policy_profile: str,
        workspace_id: str = "default",
    ) -> Dict[str, Dict[str, float]]:
        profile = self.policy_store.get_preset(policy_profile, workspace_id=workspace_id)
        raw = profile.get("detector_thresholds", {})
        if not isinstance(raw, dict):
            return {}
        normalized: Dict[str, Dict[str, float]] = {}
        for detector_name, config in raw.items():
            if not isinstance(config, dict):
                continue
            confidence = config.get("confidence_threshold")
            if confidence is None:
                continue
            try:
                value = float(confidence)
            except (TypeError, ValueError):
                continue
            normalized[str(detector_name)] = {"confidence_threshold": max(0.0, min(1.0, value))}
        return normalized

    def _get_redacted_text(
        self,
        text: str,
        agent_results: List[AgentResult],
        scan_target: str = "input",
        workspace_id: str = "default",
    ) -> str:
        redacted = text
        for result in agent_results:
            if result.threat_found:
                agent = self.agents.get(result.agent_name)
                if agent and hasattr(agent, "redact"):
                    if result.agent_name == "Custom Rules Detector":
                        redacted = agent.redact(redacted, scan_target=scan_target, workspace_id=workspace_id)
                    else:
                        redacted = agent.redact(redacted)
        return redacted

    def _overall_severity(self, agent_results: List[AgentResult]) -> str:
        max_severity = "NONE"
        for result in agent_results:
            if result.threat_found and self._severity_rank(result.severity) > self._severity_rank(max_severity):
                max_severity = result.severity
        return max_severity

    def _resolve_policy_action(
        self,
        threat_type: str,
        severity: str,
        policy_profile: str,
        workspace_id: str = "default",
    ) -> str:
        base_policy = self.policy.get(threat_type, {})
        action = base_policy.get(severity, "FLAG")
        profile = self.policy_store.get_preset(policy_profile, workspace_id=workspace_id)
        overrides = profile.get("action_overrides", {}).get(threat_type, {})
        return overrides.get(severity, action)

    def _apply_allowlist(
        self,
        text: str,
        agent_results: List[AgentResult],
        policy_profile: str,
        workspace_id: str = "default",
    ) -> List[AgentResult]:
        profile = self.policy_store.get_preset(policy_profile, workspace_id=workspace_id)
        patterns = profile.get("allowlist_patterns", [])
        if not patterns:
            return agent_results

        try:
            allowlist_hit = any(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in patterns)
        except re.error:
            allowlist_hit = False
        if not allowlist_hit:
            return agent_results

        max_allowed_rank = self._severity_rank(profile.get("allowlist_max_severity", "LOW"))
        adjusted: List[AgentResult] = []
        for result in agent_results:
            adjusted_result = deepcopy(result)
            if adjusted_result.threat_found and self._severity_rank(adjusted_result.severity) <= max_allowed_rank:
                adjusted_result.threat_found = False
                adjusted_result.summary = f"{adjusted_result.summary} Allowlisted by profile '{policy_profile}'."
                adjusted_result.matched = []
            adjusted.append(adjusted_result)
        return adjusted

    def _apply_policy(
        self,
        agent_results: List[AgentResult],
        policy_profile: str = "balanced",
        workspace_id: str = "default",
    ) -> tuple:
        final_action = "ALLOW"
        reasons = []
        triggered = []

        for result in agent_results:
            if result.threat_found:
                action = self._resolve_policy_action(
                    result.threat_type,
                    result.severity,
                    policy_profile,
                    workspace_id=workspace_id,
                )

                if self.action_priority.index(action) > self.action_priority.index(final_action):
                    final_action = action

                triggered.append(result.agent_name)
                reasons.append(f"{result.agent_name} [{result.severity}]: {result.summary}")

        reason = " | ".join(reasons) if reasons else "All agents passed. Content is clean."
        return final_action, reason, triggered

    def _confidence_from_result(self, result: AgentResult) -> float:
        confidences = [
            float(getattr(match, "confidence", 0.0))
            for match in result.matched
            if getattr(match, "confidence", None) is not None
        ]
        if not confidences:
            return 0.5 if result.threat_found else 0.0
        return max(0.0, min(1.0, max(confidences)))

    def _calibrate_confidence(self, threat_type: str, confidence: float) -> float:
        if not self.calibration_enabled:
            return max(0.0, min(1.0, confidence))
        slope, bias = self._calibration_params.get(threat_type, self._calibration_params["DEFAULT"])
        calibrated = (confidence * slope) + bias
        return max(0.0, min(1.0, calibrated))

    def _attach_calibrated_probabilities(self, agent_results: List[AgentResult]):
        for result in agent_results:
            base_conf = self._confidence_from_result(result)
            calibrated = self._calibrate_confidence(result.threat_type, base_conf)
            result.meta["raw_confidence"] = round(base_conf, 4)
            result.meta["calibrated_probability"] = round(calibrated, 4)

    def _ensemble_action(self, agent_results: List[AgentResult]) -> tuple[str, float, list[str]]:
        if not self.ensemble_enabled:
            return "ALLOW", 0.0, []
        weighted_risk = 0.0
        contributors: list[str] = []
        for result in agent_results:
            if not result.threat_found:
                continue
            probability = float((result.meta or {}).get("calibrated_probability", 0.0))
            weight = self._ensemble_weights.get(result.agent_name, 1.0)
            severity_component = max(1, self._severity_rank(result.severity))
            contribution = weight * probability * (severity_component / 2.0)
            if contribution > 0:
                contributors.append(result.agent_name)
            weighted_risk += contribution

        if weighted_risk >= self.ensemble_block_threshold:
            return "BLOCK", weighted_risk, contributors
        if weighted_risk >= self.ensemble_redact_threshold:
            return "REDACT", weighted_risk, contributors
        if weighted_risk >= self.ensemble_flag_threshold:
            return "FLAG", weighted_risk, contributors
        return "ALLOW", weighted_risk, contributors

    def _collect_matched_threats(self, agent_results: List[AgentResult]) -> dict:
        matched_threats = {}
        for result in agent_results:
            if result.threat_found and result.matched:
                matched_threats[result.agent_name] = [
                    getattr(match, "pii_type", None)
                    or getattr(match, "secret_type", None)
                    or getattr(match, "abuse_type", None)
                    or getattr(match, "injection_type", None)
                    or getattr(match, "content_type", None)
                    or getattr(match, "intel_type", None)
                    or getattr(match, "custom_type", None)
                    or "unknown"
                    for match in result.matched[:5]
                ]
        return matched_threats

    def _analyze_text(
        self,
        text: str,
        scan_target: str,
        policy_profile: str = "balanced",
        workspace_id: str = "default",
        session_id: Optional[str] = None,
    ) -> FirewallDecision:
        start_time = time.time()
        stage_latency_ms: Dict[str, float] = {}

        if not session_id:
            cached = self._get_from_cache(text, scan_target, policy_profile, workspace_id)
            if cached:
                print("[Orchestrator] Cache hit - skipping agent analysis")
                cached.resolution_stage = "exact_cache"
                cached.llm_called = False
                cached.stage_latency_ms = {}
                return cached

        print(f"[Orchestrator] Analyzing {scan_target} ({len(text)} chars)...")

        # Stage 2: semantic memory. Embed once, check the allowlist, then the threat cache.
        allowlist_hit = None
        semantic_hit = None
        semantic_similarity = None
        if self.semantic_cache.enabled:
            stage_start = time.time()
            embedding = self.semantic_cache.embed(text)
            allowlist_hit = self.semantic_cache.check_allowlist(text, embedding=embedding)
            if not allowlist_hit:
                semantic_similarity, semantic_hit = self.semantic_cache.lookup_threat(text, embedding=embedding)
            stage_latency_ms["semantic_memory"] = (time.time() - stage_start) * 1000

        if allowlist_hit and self.allowlist_mode == "bypass":
            print(f"[Orchestrator] Semantic Allowlist hit! (reason: {allowlist_hit['reason']}) Bypassing agents.")
            elapsed = (time.time() - start_time) * 1000
            decision = FirewallDecision(
                action="ALLOW",
                reason=f"Semantically matched a known safe prompt (Allowlist: {allowlist_hit['reason']}).",
                triggered_agents=[],
                severity="NONE",
                agent_results=[],
                redacted_text=text,
                original_text=text,
                processing_time_ms=elapsed,
                scan_target=scan_target,
                adversary_tier="NONE",
                resolution_stage="allowlist",
                stage_latency_ms=stage_latency_ms,
                allowlist_hit=True,
            )
            if self.audit_enabled and self.audit_logger:
                self.audit_logger.log(
                    input_text=text,
                    action="ALLOW",
                    severity="NONE",
                    triggered_agents=[],
                    reason=decision.reason,
                    processing_time_ms=decision.processing_time_ms,
                    matched_threats={},
                    scan_target=scan_target,
                )
            return decision

        llm_called = False
        if semantic_hit:
            print(f"[Orchestrator] Semantic Cache hit! Identified as {semantic_hit['threat_type']}")
            resolution_stage = "semantic_cache"
            agent_results = [
                AgentResult(
                    agent_name="Semantic Cache",
                    threat_found=True,
                    threat_type=semantic_hit["threat_type"],
                    severity=semantic_hit["severity"],
                    summary=f"Semantically matched a known threat (similarity: {semantic_hit['similarity_score']:.2f}). Originally detected by {semantic_hit['source_agent']}.",
                    matched=[],
                    meta={"semantic_cache_hit": True, "distance": semantic_hit["distance"]}
                )
            ]
        else:
            print(f"[Orchestrator] Running all {len(self.agents)} agents in parallel...\n")
            cheap_agents = [name for name in self.agents if name not in self.llm_agents]
            llm_agents = [name for name in self.agents if name in self.llm_agents]
            # LLM detectors that can run their regex pre-screen without calling the LLM.
            prescreen_agents = [
                name for name in llm_agents if getattr(self.agents[name], "supports_regex_only", False)
            ]
            detector_threshold_overrides = self._detector_threshold_overrides(policy_profile, workspace_id=workspace_id)

            # Stage 3a: deterministic detectors + regex pre-screen, in parallel, no LLM.
            stage_start = time.time()
            cheap_results = self._run_agents_parallel(
                cheap_agents + prescreen_agents,
                text,
                scan_target,
                workspace_id,
                detector_threshold_overrides=detector_threshold_overrides,
                llm_mode=False,
            )
            stage_latency_ms["deterministic"] = (time.time() - stage_start) * 1000

            deterministic_action, _, _ = self._apply_policy(
                self._apply_allowlist(text, cheap_results, policy_profile, workspace_id=workspace_id),
                policy_profile=policy_profile,
                workspace_id=workspace_id,
            )
            if allowlist_hit:
                run_llm_agents, llm_gate_score, skip_reason = False, 0.0, "Skipped: semantic allowlist hit."
            elif self.early_exit_on_block and deterministic_action == "BLOCK":
                run_llm_agents, llm_gate_score, skip_reason = False, 0.0, "Skipped: deterministic stage already blocked."
            else:
                run_llm_agents, llm_gate_score = self._should_run_llm_agents(text, cheap_results)
                skip_reason = "Skipped by low-risk gate."

            agent_results: List[AgentResult] = list(cheap_results)
            if run_llm_agents and llm_agents:
                # Stage 3b: LLM-backed detectors. Their results replace the regex-only pre-screen.
                stage_start = time.time()
                llm_results = self._run_agents_parallel(
                    llm_agents,
                    text,
                    scan_target,
                    workspace_id,
                    detector_threshold_overrides=detector_threshold_overrides,
                    llm_mode=True,
                )
                stage_latency_ms["llm"] = (time.time() - stage_start) * 1000
                llm_called = any((result.meta or {}).get("llm_called", True) for result in llm_results)
                agent_results = [result for result in agent_results if result.agent_name not in llm_agents]
                agent_results.extend(llm_results)
                resolution_stage = "llm"
            else:
                resolution_stage = "deterministic"
                print(f"[Orchestrator] LLM stage not run - {skip_reason}")
                prescreened = set(prescreen_agents)
                for result in agent_results:
                    if result.agent_name in prescreened:
                        result.meta = {**(result.meta or {}), "llm_skipped_reason": skip_reason}
                for name in llm_agents:
                    if name in prescreened:
                        continue
                    skipped = AgentResult(
                        agent_name=name,
                        threat_found=False,
                        threat_type="NONE",
                        severity="NONE",
                        summary=skip_reason,
                        matched=[],
                        agent_available=True,
                        fail_closed=False,
                        meta={
                            "llm_called": False,
                            "regex_only": False,
                            "skipped_by_orchestrator_gate": True,
                            "llm_gate_score": round(llm_gate_score, 3),
                            "llm_gate_threshold": self.llm_gate_threshold,
                        },
                    )
                    agent_results.append(skipped)

        agent_results = self._apply_allowlist(text, agent_results, policy_profile, workspace_id=workspace_id)
        self._attach_calibrated_probabilities(agent_results)
        final_action, reason, triggered = self._apply_policy(
            agent_results,
            policy_profile=policy_profile,
            workspace_id=workspace_id,
        )
        ensemble_action, ensemble_risk, ensemble_contributors = self._ensemble_action(agent_results)
        if self._action_rank(ensemble_action) > self._action_rank(final_action):
            final_action = ensemble_action
            if ensemble_contributors:
                reason += (
                    f" | Ensemble escalation -> {ensemble_action} "
                    f"(risk={ensemble_risk:.2f}, contributors={', '.join(sorted(set(ensemble_contributors)))})"
                )
        overall_severity = self._overall_severity(agent_results)
        unavailable_agents = [result.agent_name for result in agent_results if not result.agent_available]
        degraded = bool(unavailable_agents)

        if degraded:
            degraded_reason = "Unavailable detectors: " + ", ".join(unavailable_agents)
            reason = f"{reason} | {degraded_reason}" if reason else degraded_reason

        # Risk scoring is based on current turn results, so compute it before
        # session aggregation uses it for cross-turn tracking.
        prior_session_score = self.session_store.get_threat_score(session_id) if session_id else 0.0
        risk_breakdown = self.risk_scorer.calculate_risk_score(
            agent_results=agent_results,
            overall_severity=overall_severity,
            session_risk_score=prior_session_score,
            triggered_agents=triggered,
            text_length=len(text),
        )
        risk_score = risk_breakdown.overall_score
        risk_level = risk_breakdown.risk_level

        profile = self.policy_store.get_preset(policy_profile, workspace_id=workspace_id)
        flag_threshold = float(profile.get("session_flag_threshold", self.session_store.FLAG_THRESHOLD))
        block_threshold = float(profile.get("session_block_threshold", self.session_store.BLOCK_THRESHOLD))
        session_info = None
        detected_patterns = []

        if scan_target == "input" and session_id:
            session_decision = self.session_judge.apply(
                session_id=session_id,
                text=text,
                final_action=final_action,
                reason=reason,
                overall_severity=overall_severity,
                agent_results=agent_results,
                triggered=triggered,
                risk_score=risk_score,
                flag_threshold=flag_threshold,
                block_threshold=block_threshold,
            )
            final_action = session_decision.action
            reason = session_decision.reason
            session_info = session_decision.session_info
            detected_patterns = session_decision.detected_patterns
            triggered = session_decision.triggered_agents

        # ========== PRODUCTION FEATURES: Risk Scoring, Explainability, Compliance, Multi-turn Detection ==========

        # 1. Risk score already calculated above so session aggregation can use it.
        # 2. Generate explainability report after constructing the decision object.
        explanation = None
        
        # 3. Apply compliance profiles and adjust decision if needed
        try:
            compliance_profile = self.compliance_manager.get_profile(policy_profile)
            if compliance_profile and scan_target == "input":
                threshold_by_threat = {
                    "PII": compliance_profile.pii_severity_threshold,
                    "SECRET": compliance_profile.secrets_severity_threshold,
                    "ABUSE": compliance_profile.abuse_severity_threshold,
                    "INJECTION": compliance_profile.injection_severity_threshold,
                    "UNSAFE_CONTENT": compliance_profile.unsafe_content_severity_threshold,
                }
                # Check if any agent results violate compliance thresholds
                for result in agent_results:
                    if result.threat_found:
                        threat_threshold = threshold_by_threat.get(result.threat_type, "MEDIUM")
                        
                        # If compliance is stricter, potentially upgrade action
                        if self._severity_rank(result.severity) >= self._severity_rank(threat_threshold):
                            action_override = compliance_profile.action_overrides.get(result.threat_type, {}).get(result.severity)
                            if action_override and self._action_rank(action_override) > self._action_rank(final_action):
                                final_action = action_override
                                reason += f" | Compliance '{policy_profile}' escalated: {action_override}"
        except Exception as e:
            print(f"[Orchestrator] Compliance profile application failed: {e}")
        
        # ========== END PRODUCTION FEATURES ==========

        redacted_text = (
            self._get_redacted_text(text, agent_results, scan_target, workspace_id=workspace_id)
            if final_action == "REDACT"
            else text
        )
        processing_time = (time.time() - start_time) * 1000

        if self.audit_enabled:
            audit_text = self._get_redacted_text(text, agent_results, scan_target, workspace_id=workspace_id)
            self.audit_logger.log(
                input_text=audit_text,
                action=final_action,
                severity=overall_severity,
                triggered_agents=triggered,
                reason=reason,
                processing_time_ms=processing_time,
                matched_threats=self._collect_matched_threats(agent_results),
                scan_target=scan_target,
                policy_profile=policy_profile,
                workspace_id=workspace_id,
                session_id=session_id,
            )

        adversary_tier = "NONE"
        if final_action != "ALLOW":
            llm_agents_set = {"Injection Detector", "Unsafe Content Detector"}
            if any(agent in llm_agents_set for agent in triggered):
                adversary_tier = "ADVANCED_RED_TEAM"
            else:
                adversary_tier = "SCRIPT_KIDDIE"

        decision = FirewallDecision(
            action=final_action,
            reason=reason,
            severity=overall_severity,
            triggered_agents=triggered,
            agent_results=agent_results,
            redacted_text=redacted_text,
            processing_time_ms=processing_time,
            original_text=text,
            scan_target=scan_target,
            from_cache=False,
            degraded=degraded,
            unavailable_agents=unavailable_agents,
            policy_profile=policy_profile,
            risk_score=risk_score,
            risk_level=risk_level,
            explanation=explanation,
            detected_patterns=detected_patterns,
            adversary_tier=adversary_tier,
            resolution_stage=resolution_stage,
            llm_called=llm_called,
            stage_latency_ms=stage_latency_ms,
            allowlist_hit=bool(allowlist_hit),
            semantic_similarity=semantic_similarity,
        )

        try:
            decision.explanation = self.explainability_generator.generate_report(
                decision=decision,
                risk_score_breakdown=risk_breakdown,
                session_info=session_info,
                workspace_id=workspace_id,
                policy_name=policy_profile,
            )
        except Exception as e:
            print(f"[Orchestrator] Explainability generation failed: {e}")

        try:
            self.alert_manager.dispatch(
                {
                    "action": decision.action,
                    "severity": decision.severity,
                    "reason": decision.reason,
                    "scan_target": decision.scan_target,
                    "policy_profile": decision.policy_profile,
                    "triggered_agents": decision.triggered_agents,
                    "risk_score": decision.risk_score,
                    "risk_level": decision.risk_level,
                    "session_id": session_id,
                }
            )
        except Exception as e:
            print(f"[Orchestrator] Alert dispatch failed: {e}")

        if not session_id:
            self._save_to_cache(text, decision, scan_target, policy_profile, workspace_id)

        # Write-back: store threats found by the LLM-backed detectors. The cache's policy
        # decides whether the entry is persisted (LLM-confirmed, confidence floor, size cap).
        # Detector failures (SYSTEM_UNAVAILABLE) and allowlisted prompts are never written.
        if not allowlist_hit:
            for res in agent_results:
                if (
                    res.threat_found
                    and res.agent_name in self.llm_agents
                    and res.threat_type not in {"NONE", "SYSTEM_UNAVAILABLE"}
                ):
                    meta = res.meta or {}
                    self.semantic_cache.add_threat(
                        text=text,
                        threat_type=res.threat_type,
                        severity=res.severity,
                        source_agent=res.agent_name,
                        confidence=self._confidence_from_result(res),
                        llm_confirmed=int(meta.get("llm_match_count", 0) or 0) > 0,
                    )

        self._print_decision(decision)
        return decision

    def _interaction_severity_to_action(self, severity: str) -> str:
        if severity == "CRITICAL":
            return "BLOCK"
        if severity in {"HIGH", "MEDIUM"}:
            return "FLAG"
        return "ALLOW"

    def _collect_decision_labels(self, decision: FirewallDecision) -> set:
        labels = set()
        for result in decision.agent_results:
            if not result.threat_found:
                continue
            labels.add(result.threat_type)
            for match in result.matched:
                label = (
                    getattr(match, "pii_type", None)
                    or getattr(match, "secret_type", None)
                    or getattr(match, "abuse_type", None)
                    or getattr(match, "injection_type", None)
                    or getattr(match, "content_type", None)
                    or getattr(match, "intel_type", None)
                    or getattr(match, "custom_type", None)
                )
                if label:
                    labels.add(str(label).upper())
        return labels

    def _detect_contradiction_alerts(
        self,
        prompt_text: str,
        output_text: str,
        prompt_decision: FirewallDecision,
        output_decision: FirewallDecision,
    ) -> List[dict]:
        alerts: List[dict] = []
        prompt_labels = self._collect_decision_labels(prompt_decision)
        output_labels = self._collect_decision_labels(output_decision)
        new_output_labels = sorted(output_labels - prompt_labels)

        output_is_riskier = self._severity_rank(output_decision.severity) > self._severity_rank(prompt_decision.severity)
        output_action_stronger = self._action_rank(output_decision.action) > self._action_rank(prompt_decision.action)

        if new_output_labels and output_action_stronger:
            alerts.append(
                {
                    "alert_type": "output_policy_contradiction",
                    "severity": "HIGH" if self._severity_rank(output_decision.severity) >= self._severity_rank("HIGH") else "MEDIUM",
                    "summary": "The model output introduced higher-risk content categories that were not present in the prompt.",
                    "evidence": ", ".join(new_output_labels[:5]),
                }
            )

        refusal_signals = [
            r"\bI can't help with that\b",
            r"\bI cannot assist with that\b",
            r"\bI won't provide\b",
            r"\bI must refuse\b",
            r"\bI can't provide instructions\b",
        ]
        prompt_refusal = any(re.search(pattern, prompt_text, re.IGNORECASE) for pattern in refusal_signals)
        output_refusal = any(re.search(pattern, output_text, re.IGNORECASE) for pattern in refusal_signals)
        if prompt_refusal and not output_refusal and output_action_stronger:
            alerts.append(
                {
                    "alert_type": "refusal_break",
                    "severity": "HIGH",
                    "summary": "The interaction appears to move from refusal language to a riskier output, which can indicate policy drift.",
                    "evidence": output_text[:160],
                }
            )

        if output_is_riskier and output_decision.action in {"REDACT", "BLOCK"} and prompt_decision.action == "ALLOW":
            alerts.append(
                {
                    "alert_type": "safe_prompt_unsafe_output",
                    "severity": "CRITICAL" if output_decision.action == "BLOCK" else "HIGH",
                    "summary": "A relatively clean prompt produced a materially riskier output.",
                    "evidence": output_decision.reason[:200],
                }
            )

        return alerts

    def _detect_drift_alerts(
        self,
        session_id: Optional[str],
        prompt_decision: FirewallDecision,
        output_decision: FirewallDecision,
    ) -> tuple[List[dict], float]:
        if not session_id:
            return [], 0.0

        history = self.session_store.get_history(session_id)
        if not history:
            return [], 0.0

        history_scores = [self._severity_rank(item.get("severity", "NONE")) for item in history]
        previous_scores = history_scores[:-1] if len(history_scores) > 1 else []
        session_risk_score = self.session_store.get_threat_score(session_id)
        current_peak = max(
            self._severity_rank(prompt_decision.severity),
            self._severity_rank(output_decision.severity),
        )

        alerts: List[dict] = []
        if previous_scores:
            prev_avg = sum(previous_scores) / len(previous_scores)
            risky_turns = sum(score >= self._severity_rank("MEDIUM") for score in previous_scores)

            if current_peak >= prev_avg + 2:
                alerts.append(
                    {
                        "alert_type": "conversation_risk_spike",
                        "severity": "HIGH",
                        "summary": "The current interaction is notably riskier than the recent conversation baseline.",
                        "evidence": f"previous_avg={prev_avg:.2f}, current_peak={current_peak}",
                    }
                )

            if risky_turns >= 2 and self._severity_rank(output_decision.severity) >= self._severity_rank("HIGH"):
                alerts.append(
                    {
                        "alert_type": "multi_turn_drift",
                        "severity": "CRITICAL",
                        "summary": "The conversation shows sustained risky turns and the latest output escalates that pattern.",
                        "evidence": f"risky_turns={risky_turns}, session_score={session_risk_score:.1f}",
                    }
                )

        if self.session_store.should_flag(session_id):
            alerts.append(
                {
                    "alert_type": "elevated_session_risk",
                    "severity": "MEDIUM" if not self.session_store.should_block(session_id) else "HIGH",
                    "summary": "The session has accumulated elevated risk across turns.",
                    "evidence": f"session_score={session_risk_score:.1f}",
                }
            )

        deduped = []
        seen = set()
        for alert in alerts:
            key = (alert["alert_type"], alert["summary"])
            if key not in seen:
                seen.add(key)
                deduped.append(alert)
        return deduped, session_risk_score

    def analyze(
        self,
        text: str,
        session_id: Optional[str] = None,
        policy_profile: str = "balanced",
        workspace_id: str = "default",
    ) -> FirewallDecision:
        return self._analyze_text(
            text=text,
            scan_target="input",
            policy_profile=policy_profile,
            workspace_id=workspace_id,
            session_id=session_id,
        )

    def analyze_output(
        self,
        text: str,
        policy_profile: str = "balanced",
        workspace_id: str = "default",
    ) -> FirewallDecision:
        return self._analyze_text(
            text=text,
            scan_target="output",
            policy_profile=policy_profile,
            workspace_id=workspace_id,
            session_id=None,
        )

    def get_session_summary(self, session_id: str) -> dict:
        preset = self.policy_store.get_preset("balanced")
        return self.session_store.summary(
            session_id,
            flag_threshold=float(preset.get("session_flag_threshold", self.session_store.FLAG_THRESHOLD)),
            block_threshold=float(preset.get("session_block_threshold", self.session_store.BLOCK_THRESHOLD)),
        )

    def clear_session(self, session_id: str):
        self.session_store.clear(session_id)

    def list_policy_presets(self, workspace_id: Optional[str] = None) -> List[dict]:
        return self.policy_store.list_presets(workspace_id=workspace_id)

    def list_policy_drafts(self, workspace_id: Optional[str] = None) -> List[dict]:
        return self.policy_store.list_drafts(workspace_id=workspace_id)

    def save_policy_preset(self, **kwargs) -> dict:
        return self.policy_store.save_preset(**kwargs)

    def delete_policy_preset(self, name: str, workspace_id: Optional[str] = None) -> bool:
        return self.policy_store.delete_preset(name, workspace_id=workspace_id)

    def create_policy_draft_from_adjustment(self, adjustment: dict, workspace_id: str = "global") -> dict:
        return self.policy_store.create_draft_from_adjustment(adjustment, workspace_id=workspace_id)

    def request_policy_approval(self, name: str, requested_by: str, workspace_id: str = "global") -> Optional[dict]:
        return self.policy_store.request_approval(name, requested_by, workspace_id=workspace_id)

    def approve_policy_draft(self, name: str, approved_by: str, workspace_id: str = "global") -> Optional[dict]:
        return self.policy_store.approve_draft(name, approved_by, workspace_id=workspace_id)

    def promote_policy_draft(
        self,
        name: str,
        note: str = "",
        workspace_id: str = "default",
    ) -> Optional[dict]:
        promoted = self.policy_store.promote_draft(name, workspace_id=workspace_id)
        if promoted:
            self.audit_logger.log_promotion(
                item_type="policy",
                item_id=promoted["name"],
                item_name=promoted["name"],
                note=note,
                metadata={"description": promoted.get("description", "")},
                workspace_id=workspace_id,
            )
        return promoted

    def preview_policy_draft(self, name: str, workspace_id: Optional[str] = None) -> Optional[dict]:
        return self.policy_store.preview_draft_diff(name, workspace_id=workspace_id)

    def get_promotion_history(self, limit: int = 50, workspace_id: Optional[str] = None) -> List[dict]:
        return self.audit_logger.get_promotion_history(limit=limit, workspace_id=workspace_id)

    def analyze_interaction(
        self,
        prompt_text: str,
        output_text: str,
        session_id: Optional[str] = None,
        policy_profile: str = "balanced",
        workspace_id: str = "default",
    ) -> InteractionDecision:
        prompt_decision = self.analyze(
            prompt_text,
            session_id=session_id,
            policy_profile=policy_profile,
            workspace_id=workspace_id,
        )
        output_decision = self.analyze_output(
            output_text,
            policy_profile=policy_profile,
            workspace_id=workspace_id,
        )

        combined_action = max([prompt_decision.action, output_decision.action], key=self._action_rank)
        combined_severity = max([prompt_decision.severity, output_decision.severity], key=self._severity_rank)
        triggered_agents = list(dict.fromkeys(
            [f"prompt:{agent}" for agent in prompt_decision.triggered_agents]
            + [f"output:{agent}" for agent in output_decision.triggered_agents]
        ))
        interaction_alerts = self._detect_contradiction_alerts(
            prompt_text=prompt_text,
            output_text=output_text,
            prompt_decision=prompt_decision,
            output_decision=output_decision,
        )
        drift_alerts, session_risk_score = self._detect_drift_alerts(
            session_id=session_id,
            prompt_decision=prompt_decision,
            output_decision=output_decision,
        )
        interaction_alerts.extend(drift_alerts)

        for alert in interaction_alerts:
            combined_severity = max([combined_severity, alert["severity"]], key=self._severity_rank)
            combined_action = max(
                [combined_action, self._interaction_severity_to_action(alert["severity"])],
                key=self._action_rank,
            )
            triggered_agents.append(f"interaction:{alert['alert_type']}")

        combined_reason = (
            f"Prompt: {prompt_decision.reason} | "
            f"Output: {output_decision.reason}"
        )
        if interaction_alerts:
            combined_reason += " | Interaction alerts: " + "; ".join(
                f"{alert['alert_type']} [{alert['severity']}]: {alert['summary']}"
                for alert in interaction_alerts
            )

        return InteractionDecision(
            prompt_decision=prompt_decision,
            output_decision=output_decision,
            combined_action=combined_action,
            combined_severity=combined_severity,
            combined_reason=combined_reason,
            triggered_agents=list(dict.fromkeys(triggered_agents)),
            interaction_alerts=interaction_alerts,
            drift_detected=any(alert["alert_type"] in {"conversation_risk_spike", "multi_turn_drift", "elevated_session_risk"} for alert in interaction_alerts),
            contradiction_detected=any(alert["alert_type"] in {"output_policy_contradiction", "refusal_break", "safe_prompt_unsafe_output"} for alert in interaction_alerts),
            session_risk_score=session_risk_score,
        )

    def _print_decision(self, decision: FirewallDecision):
        action_icons = {
            "ALLOW": "ALLOW",
            "FLAG": "FLAG",
            "REDACT": "REDACT",
            "BLOCK": "BLOCK",
        }
        severity_icons = {
            "NONE": "NONE",
            "LOW": "LOW",
            "MEDIUM": "MEDIUM",
            "HIGH": "HIGH",
            "CRITICAL": "CRITICAL",
        }
        cache_tag = " [CACHED]" if decision.from_cache else ""
        print(f"\n{'=' * 60}")
        print(f"  FIREWALL DECISION: {action_icons.get(decision.action, decision.action)}{cache_tag}")
        print(f"  Target    : {decision.scan_target}")
        print(f"  Severity  : {severity_icons.get(decision.severity, '')} {decision.severity}")
        print(
            f"  Triggered : "
            f"{', '.join(decision.triggered_agents) if decision.triggered_agents else 'None'}"
        )
        print(f"  Time      : {decision.processing_time_ms:.0f}ms")
        print(f"  Reason    : {decision.reason[:120]}...")
        if decision.action == "REDACT":
            print(f"  Redacted  : {decision.redacted_text[:100]}...")
        print(f"{'=' * 60}\n")

