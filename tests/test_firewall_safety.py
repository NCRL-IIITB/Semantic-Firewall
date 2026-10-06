from dataclasses import dataclass
import base64
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

import semantic_firewall.apps.api_server as api_module
from semantic_firewall.core.agents.custom_rules_detector import CustomRulesDetectorAgent
from semantic_firewall.core.orchestrator.custom_rules import CustomRulesManager
from semantic_firewall.core.orchestrator.audit_logger import AuditLogger
from semantic_firewall.core.orchestrator.document_extractor import DocumentExtractor, DocumentExtractionError
from semantic_firewall.core.orchestrator.orchestrator import (
    AgentResult,
    FirewallDecision,
    InteractionDecision,
    SemanticFirewallOrchestrator,
)

TMP_DIR = Path(__file__).resolve().parent / ".tmp"


def _tmp_file(stem: str, ext: str) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    return TMP_DIR / f"{stem}_{uuid4().hex}{ext}"


@dataclass
class StubMatch:
    pii_type: str


@dataclass
class StubDetection:
    agent_name: str
    threat_found: bool
    threat_type: str
    matched: list
    severity: str
    summary: str


class CleanAgent:
    def __init__(self, name: str, threat_type: str):
        self.name = name
        self.threat_type = threat_type

    def run(self, text: str, **kwargs):
        return StubDetection(
            agent_name=self.name,
            threat_found=False,
            threat_type=self.threat_type,
            matched=[],
            severity="NONE",
            summary="clean",
        )


class UnsafeOutputAgent:
    name = "Unsafe Content Detector"

    def run(self, text: str, **kwargs):
        is_unsafe = "bomb" in text.lower()
        return StubDetection(
            agent_name=self.name,
            threat_found=is_unsafe,
            threat_type="UNSAFE_CONTENT",
            matched=[StubMatch(pii_type="unsafe")] if is_unsafe else [],
            severity="HIGH" if is_unsafe else "NONE",
            summary="unsafe output found" if is_unsafe else "clean",
        )


class PIIAgent:
    name = "PII Detector"

    def run(self, text: str, **kwargs):
        matched = []
        if "alice@example.com" in text:
            matched.append(StubMatch(pii_type="email"))
        return StubDetection(
            agent_name=self.name,
            threat_found=bool(matched),
            threat_type="PII",
            matched=matched,
            severity="LOW" if matched else "NONE",
            summary="pii found" if matched else "clean",
        )

    def redact(self, text: str) -> str:
        return text.replace("alice@example.com", "[EMAIL]")


class FailingAgent:
    def run(self, text: str, **kwargs):
        raise RuntimeError("upstream unavailable")


class SlowAgent:
    def __init__(self, name: str, threat_type: str, sleep_seconds: float):
        self.name = name
        self.threat_type = threat_type
        self.sleep_seconds = sleep_seconds

    def run(self, text: str, **kwargs):
        import time

        time.sleep(self.sleep_seconds)
        return StubDetection(
            agent_name=self.name,
            threat_found=False,
            threat_type=self.threat_type,
            matched=[],
            severity="NONE",
            summary="slow clean",
        )


class CountingAgent(CleanAgent):
    def __init__(self, name: str, threat_type: str):
        super().__init__(name, threat_type)
        self.calls = 0

    def run(self, text: str, **kwargs):
        self.calls += 1
        return super().run(text, **kwargs)


class ThresholdAwareInjectionAgent(CleanAgent):
    def __init__(self):
        super().__init__("Injection Detector", "INJECTION")
        self.last_threshold = None

    def run(self, text: str, **kwargs):
        self.last_threshold = kwargs.get("confidence_threshold_override")
        return super().run(text, **kwargs)


class HighSeverityAgent(CleanAgent):
    def __init__(self, name: str, threat_type: str, severity: str = "HIGH"):
        super().__init__(name, threat_type)
        self._severity = severity

    def run(self, text: str, **kwargs):
        return StubDetection(
            agent_name=self.name,
            threat_found=True,
            threat_type=self.threat_type,
            matched=[StubMatch(pii_type="hit")],
            severity=self._severity,
            summary=f"{self.threat_type.lower()} found",
        )


def test_orchestrator_redacts_audit_logs_and_flags_fail_closed_agent():
    db_path = _tmp_file("tmp_audit", ".db")
    orchestrator = SemanticFirewallOrchestrator(db_path=str(db_path))
    orchestrator.agents = {
        "PII Detector": PIIAgent(),
        "Secrets Detector": CleanAgent("Secrets Detector", "SECRET"),
        "Abuse Detector": CleanAgent("Abuse Detector", "ABUSE"),
        "Injection Detector": FailingAgent(),
        "Unsafe Content Detector": CleanAgent("Unsafe Content Detector", "UNSAFE_CONTENT"),
    }

    decision = orchestrator.analyze("Email alice@example.com for help.", workspace_id="team-alpha")

    assert decision.action == "FLAG"
    assert decision.degraded is True
    assert "Injection Detector" in decision.unavailable_agents

    log_entry = orchestrator.audit_logger.get_recent(limit=1)[0]
    assert "alice@example.com" not in log_entry["input_text"]
    assert "[EMAIL]" in log_entry["input_text"]
    assert len(log_entry["input_hash"]) == 64


def test_orchestrator_low_risk_gate_skips_llm_detectors():
    db_path = _tmp_file("tmp_audit", ".db")
    orchestrator = SemanticFirewallOrchestrator(db_path=str(db_path))
    injection = CountingAgent("Injection Detector", "INJECTION")
    unsafe = CountingAgent("Unsafe Content Detector", "UNSAFE_CONTENT")
    orchestrator.agents = {
        "PII Detector": CleanAgent("PII Detector", "PII"),
        "Secrets Detector": CleanAgent("Secrets Detector", "SECRET"),
        "Abuse Detector": CleanAgent("Abuse Detector", "ABUSE"),
        "Threat Intel Detector": CleanAgent("Threat Intel Detector", "THREAT_INTEL"),
        "Custom Rules Detector": CleanAgent("Custom Rules Detector", "CUSTOM_RULE"),
        "Injection Detector": injection,
        "Unsafe Content Detector": unsafe,
    }
    orchestrator.llm_gate_enabled = True

    decision = orchestrator.analyze("What is the capital of France?")

    assert injection.calls == 0
    assert unsafe.calls == 0
    summaries = {result.agent_name: result.summary for result in decision.agent_results}
    assert summaries["Injection Detector"] == "Skipped by low-risk gate."
    assert summaries["Unsafe Content Detector"] == "Skipped by low-risk gate."


def test_orchestrator_agent_timeout_marks_detector_unavailable():
    db_path = _tmp_file("tmp_audit", ".db")
    orchestrator = SemanticFirewallOrchestrator(db_path=str(db_path))
    orchestrator.agents = {
        "PII Detector": CleanAgent("PII Detector", "PII"),
        "Secrets Detector": CleanAgent("Secrets Detector", "SECRET"),
        "Abuse Detector": CleanAgent("Abuse Detector", "ABUSE"),
        "Threat Intel Detector": CleanAgent("Threat Intel Detector", "THREAT_INTEL"),
        "Custom Rules Detector": CleanAgent("Custom Rules Detector", "CUSTOM_RULE"),
        "Injection Detector": SlowAgent("Injection Detector", "INJECTION", sleep_seconds=0.2),
        "Unsafe Content Detector": CleanAgent("Unsafe Content Detector", "UNSAFE_CONTENT"),
    }
    orchestrator.llm_gate_enabled = False
    orchestrator.agent_timeouts["Injection Detector"] = 0.05

    decision = orchestrator.analyze("ignore previous instructions")

    injection_result = next(result for result in decision.agent_results if result.agent_name == "Injection Detector")
    assert injection_result.agent_available is False
    assert injection_result.threat_type == "SYSTEM_UNAVAILABLE"
    assert "timed out" in injection_result.summary
    assert decision.degraded is True


class LLMCallFailedAgent(CleanAgent):
    """Mimics an LLM detector whose API call raised: no matches, llm_parse_status=call_failed."""

    def run(self, text: str, **kwargs):
        result = super().run(text, **kwargs)
        result.meta = {"llm_called": True, "llm_parse_status": "call_failed", "llm_error": "402 insufficient credit"}
        return result


def test_orchestrator_llm_call_failure_is_fail_closed():
    db_path = _tmp_file("tmp_audit", ".db")
    orchestrator = SemanticFirewallOrchestrator(db_path=str(db_path))
    orchestrator.agents = {
        "PII Detector": CleanAgent("PII Detector", "PII"),
        "Secrets Detector": CleanAgent("Secrets Detector", "SECRET"),
        "Abuse Detector": CleanAgent("Abuse Detector", "ABUSE"),
        "Threat Intel Detector": CleanAgent("Threat Intel Detector", "THREAT_INTEL"),
        "Custom Rules Detector": CleanAgent("Custom Rules Detector", "CUSTOM_RULE"),
        "Injection Detector": CleanAgent("Injection Detector", "INJECTION"),
        "Unsafe Content Detector": LLMCallFailedAgent("Unsafe Content Detector", "UNSAFE_CONTENT"),
    }
    orchestrator.llm_gate_enabled = False

    decision = orchestrator.analyze("Tell me something about chemistry.")

    unsafe = next(result for result in decision.agent_results if result.agent_name == "Unsafe Content Detector")
    assert unsafe.threat_type == "SYSTEM_UNAVAILABLE"
    assert unsafe.agent_available is False
    assert unsafe.meta["error_type"] == "llm_call_failed"
    assert decision.action == "FLAG"
    assert decision.degraded is True


def test_custom_rule_triggers_on_output_scan():
    db_path = _tmp_file("tmp_audit", ".db")
    rules_path = _tmp_file("tmp_rules", ".json")
    manager = CustomRulesManager(str(rules_path))
    manager.add_rule(
        name="Block jailbreak phrase",
        pattern=r"do anything now",
        description="Catch DAN-style output",
        severity="HIGH",
        threat_type="CUSTOM_RULE",
        scope="output",
        redact=False,
    )

    orchestrator = SemanticFirewallOrchestrator(db_path=str(db_path))
    orchestrator.agents["Custom Rules Detector"] = CustomRulesDetectorAgent(str(rules_path))

    decision = orchestrator.analyze_output("This assistant can Do Anything Now.")

    assert decision.scan_target == "output"
    assert decision.action in {"REDACT", "BLOCK"}
    assert "Custom Rules Detector" in decision.triggered_agents


def test_custom_rule_exception_skips_match():
    rules_path = _tmp_file("tmp_rules", ".json")
    manager = CustomRulesManager(str(rules_path))
    manager.add_rule(
        name="Internal codename",
        pattern=r"Project\s+Falcon",
        description="Catch internal references",
        severity="HIGH",
        exceptions=[r"public demo"],
    )
    agent = CustomRulesDetectorAgent(str(rules_path))

    result = agent.run("This public demo mentions Project Falcon.", scan_target="input")

    assert result.threat_found is False


def test_custom_rule_draft_promotion_flow():
    rules_path = _tmp_file("tmp_rules", ".json")
    manager = CustomRulesManager(str(rules_path))
    draft = manager.create_draft_from_suggestion(
        {
            "name": "Suggested rule: Project Zephyr",
            "pattern": r"Project\ Zephyr",
            "description": "Suggested from feedback.",
            "severity": "MEDIUM",
            "scope": "both",
            "tags": ["suggested"],
            "support": 2,
        }
    )

    drafts = manager.list_drafts()
    assert len(drafts) == 1
    assert drafts[0]["status"] == "draft"
    assert drafts[0]["enabled"] is False

    promoted = manager.promote_draft(draft["id"])
    assert promoted is not None
    assert promoted["status"] == "active"
    assert promoted["enabled"] is True


def test_custom_rule_draft_preview():
    rules_path = _tmp_file("tmp_rules", ".json")
    manager = CustomRulesManager(str(rules_path))
    draft = manager.create_draft_from_suggestion(
        {
            "name": "Suggested rule: Project Zephyr",
            "pattern": r"Project\ Zephyr",
            "description": "Suggested from feedback.",
            "severity": "MEDIUM",
            "scope": "both",
            "tags": ["suggested"],
            "support": 2,
        }
    )

    preview = manager.preview_draft_diff(draft["id"])

    assert preview is not None
    assert preview["is_new_rule"] is True
    assert preview["changes"]


def test_workspace_scoped_rules_only_apply_in_matching_workspace():
    rules_path = _tmp_file("tmp_rules", ".json")
    manager = CustomRulesManager(str(rules_path))
    manager.add_rule(
        name="Alpha secret phrase",
        pattern=r"alpha-only-secret",
        description="Workspace-specific rule",
        severity="HIGH",
        workspace_id="alpha",
    )
    agent = CustomRulesDetectorAgent(str(rules_path))

    alpha_result = agent.run("alpha-only-secret", scan_target="input", workspace_id="alpha")
    beta_result = agent.run("alpha-only-secret", scan_target="input", workspace_id="beta")

    assert alpha_result.threat_found is True
    assert beta_result.threat_found is False


def test_rule_approval_flow_requires_different_approver():
    rules_path = _tmp_file("tmp_rules", ".json")
    manager = CustomRulesManager(str(rules_path))
    draft = manager.create_draft_from_suggestion(
        {
            "name": "Suggested rule: Project Orion",
            "pattern": r"Project\ Orion",
            "description": "Suggested from feedback.",
            "severity": "MEDIUM",
            "scope": "both",
            "tags": ["suggested"],
            "support": 1,
        },
        workspace_id="alpha",
    )

    manager.request_approval(draft["id"], "alice")
    try:
        manager.approve_draft(draft["id"], "alice")
        assert False, "Expected same-user approval rejection"
    except ValueError:
        pass

    approved = manager.approve_draft(draft["id"], "bob")
    promoted = manager.promote_draft(draft["id"])

    assert approved is not None
    assert approved["approval_status"] == "approved"
    assert promoted is not None
    assert promoted["status"] == "active"


def test_document_extractor_reads_text_and_csv():
    extractor = DocumentExtractor()

    text_doc = extractor.extract_text("note.txt", b"hello\nworld")
    csv_doc = extractor.extract_text("table.csv", b"name,email\nAlice,alice@example.com")

    assert text_doc["text"] == "hello\nworld"
    assert "Alice | alice@example.com" in csv_doc["text"]
    assert text_doc["extraction_quality"] == "high"
    assert csv_doc["extraction_mode"] == "structured_csv"


def test_document_extractor_rejects_unsupported_extension():
    extractor = DocumentExtractor()
    try:
        extractor.extract_text("archive.zip", b"123")
        assert False, "Expected unsupported file type error"
    except DocumentExtractionError as exc:
        assert "Unsupported file type" in str(exc)


def test_audit_logger_filters_by_workspace_and_tracks_promotions():
    db_path = _tmp_file("tmp_audit", ".db")
    logger = AuditLogger(db_path=str(db_path))
    logger.log(
        input_text="alpha item",
        action="FLAG",
        severity="LOW",
        triggered_agents=["PII Detector"],
        reason="alpha",
        processing_time_ms=1.0,
        matched_threats={},
        workspace_id="alpha",
    )
    logger.log(
        input_text="beta item",
        action="BLOCK",
        severity="HIGH",
        triggered_agents=["Unsafe Content Detector"],
        reason="beta",
        processing_time_ms=1.0,
        matched_threats={},
        workspace_id="beta",
    )
    logger.log_promotion(
        item_type="rule",
        item_id="rule-1",
        item_name="Draft Rule",
        note="approved",
        workspace_id="alpha",
    )

    alpha_rows = logger.get_recent(limit=10, workspace_id="alpha")
    beta_rows = logger.get_recent(limit=10, workspace_id="beta")
    promotions = logger.get_promotion_history(workspace_id="alpha")

    assert len(alpha_rows) == 1
    assert alpha_rows[0]["workspace_id"] == "alpha"
    assert len(beta_rows) == 1
    assert promotions[0]["note"] == "approved"
    analytics = logger.get_review_queue_analytics(workspace_id="alpha")
    assert analytics["pending_total"] == 1


def test_policy_preset_allowlist_downgrades_low_severity_findings():
    db_path = _tmp_file("tmp_audit", ".db")
    orchestrator = SemanticFirewallOrchestrator(db_path=str(db_path))
    orchestrator.agents = {
        "PII Detector": PIIAgent(),
        "Secrets Detector": CleanAgent("Secrets Detector", "SECRET"),
        "Abuse Detector": CleanAgent("Abuse Detector", "ABUSE"),
        "Injection Detector": CleanAgent("Injection Detector", "INJECTION"),
        "Unsafe Content Detector": CleanAgent("Unsafe Content Detector", "UNSAFE_CONTENT"),
        "Custom Rules Detector": CleanAgent("Custom Rules Detector", "CUSTOM_RULE"),
    }
    orchestrator.save_policy_preset(
        name="allowlisted_demo",
        description="Allow demo emails",
        action_overrides={},
        session_flag_threshold=8.0,
        session_block_threshold=12.0,
        allowlist_patterns=[r"approved demo"],
        allowlist_max_severity="LOW",
    )

    decision = orchestrator.analyze(
        "approved demo alice@example.com",
        policy_profile="allowlisted_demo",
    )

    assert decision.action == "ALLOW"
    assert decision.policy_profile == "allowlisted_demo"


def test_policy_profile_applies_detector_threshold_overrides():
    db_path = _tmp_file("tmp_audit", ".db")
    orchestrator = SemanticFirewallOrchestrator(db_path=str(db_path))
    injection = ThresholdAwareInjectionAgent()
    orchestrator.agents = {
        "PII Detector": CleanAgent("PII Detector", "PII"),
        "Secrets Detector": CleanAgent("Secrets Detector", "SECRET"),
        "Abuse Detector": CleanAgent("Abuse Detector", "ABUSE"),
        "Injection Detector": injection,
        "Unsafe Content Detector": CleanAgent("Unsafe Content Detector", "UNSAFE_CONTENT"),
        "Threat Intel Detector": CleanAgent("Threat Intel Detector", "THREAT_INTEL"),
        "Custom Rules Detector": CleanAgent("Custom Rules Detector", "CUSTOM_RULE"),
    }
    orchestrator.llm_gate_enabled = False
    orchestrator.save_policy_preset(
        name="detector_threshold_demo",
        description="Profile with detector threshold overrides.",
        action_overrides={},
        session_flag_threshold=8.0,
        session_block_threshold=12.0,
        allowlist_patterns=[],
        allowlist_max_severity="LOW",
        detector_thresholds={"Injection Detector": {"confidence_threshold": 0.77}},
    )

    orchestrator.analyze("potential injection text", policy_profile="detector_threshold_demo")

    assert injection.last_threshold == 0.77


def test_policy_draft_from_adjustment_flow():
    from semantic_firewall.core.orchestrator.policy_store import PolicyStore

    presets_path = _tmp_file("tmp_policy", ".json")
    store = PolicyStore(str(presets_path))
    draft = store.create_draft_from_adjustment(
        {
            "profile": "balanced",
            "agent": "PII Detector",
            "feedback_type": "false_negative",
            "count": 2,
            "severity": "LOW",
        }
    )

    drafts = store.list_drafts()
    assert drafts
    assert draft["status"] == "draft"
    assert draft["enabled"] is False

    promoted = store.promote_draft(draft["name"])
    assert promoted is not None
    assert promoted["status"] == "active"
    assert promoted["enabled"] is True


def test_policy_draft_preview():
    from semantic_firewall.core.orchestrator.policy_store import PolicyStore

    presets_path = _tmp_file("tmp_policy", ".json")
    store = PolicyStore(str(presets_path))
    draft = store.create_draft_from_adjustment(
        {
            "profile": "balanced",
            "agent": "PII Detector",
            "feedback_type": "false_negative",
            "count": 2,
            "severity": "LOW",
        }
    )

    preview = store.preview_draft_diff(draft["name"])

    assert preview is not None
    assert preview["base_profile"] == "balanced"
    assert preview["changes"]


def test_workspace_scoped_policy_lookup_prefers_matching_workspace():
    from semantic_firewall.core.orchestrator.policy_store import PolicyStore

    presets_path = _tmp_file("tmp_policy", ".json")
    store = PolicyStore(str(presets_path))
    store.save_preset(
        name="balanced",
        description="workspace override",
        action_overrides={"PII": {"LOW": "BLOCK"}},
        workspace_id="alpha",
    )

    alpha = store.get_preset("balanced", workspace_id="alpha")
    beta = store.get_preset("balanced", workspace_id="beta")

    assert alpha["action_overrides"]["PII"]["LOW"] == "BLOCK"
    assert beta["description"] == "Default policy profile."


def test_feedback_is_stored_with_audit_entry():
    db_path = _tmp_file("tmp_audit", ".db")
    orchestrator = SemanticFirewallOrchestrator(db_path=str(db_path))
    orchestrator.agents = {
        "PII Detector": PIIAgent(),
        "Secrets Detector": CleanAgent("Secrets Detector", "SECRET"),
        "Abuse Detector": CleanAgent("Abuse Detector", "ABUSE"),
        "Injection Detector": CleanAgent("Injection Detector", "INJECTION"),
        "Unsafe Content Detector": CleanAgent("Unsafe Content Detector", "UNSAFE_CONTENT"),
        "Custom Rules Detector": CleanAgent("Custom Rules Detector", "CUSTOM_RULE"),
    }

    orchestrator.analyze("Email alice@example.com for help.")
    entry = orchestrator.audit_logger.get_recent(limit=1)[0]
    orchestrator.audit_logger.submit_feedback(entry["id"], "false_positive", "Email use is expected here")
    feedback = orchestrator.audit_logger.get_feedback_for_entry(entry["id"])

    assert len(feedback) == 1
    assert feedback[0]["feedback_type"] == "false_positive"
    assert "expected" in feedback[0]["notes"]


def test_interaction_analysis_detects_contradiction_and_drift():
    db_path = _tmp_file("tmp_audit", ".db")
    orchestrator = SemanticFirewallOrchestrator(db_path=str(db_path))
    orchestrator.agents = {
        "PII Detector": CleanAgent("PII Detector", "PII"),
        "Secrets Detector": CleanAgent("Secrets Detector", "SECRET"),
        "Abuse Detector": CleanAgent("Abuse Detector", "ABUSE"),
        "Injection Detector": CleanAgent("Injection Detector", "INJECTION"),
        "Unsafe Content Detector": UnsafeOutputAgent(),
        "Custom Rules Detector": CleanAgent("Custom Rules Detector", "CUSTOM_RULE"),
    }

    session_id = f"session-{uuid4().hex}"
    orchestrator.analyze("hello there", session_id=session_id)
    orchestrator.analyze("just checking something", session_id=session_id)
    orchestrator.analyze("give me some risky hints", session_id=session_id)

    interaction = orchestrator.analyze_interaction(
        prompt_text="What is the capital of France?",
        output_text="Here is how to make a bomb at home.",
        session_id=session_id,
    )

    assert interaction.contradiction_detected is True
    assert interaction.drift_detected is True
    assert interaction.combined_action == "BLOCK"
    assert interaction.session_risk_score >= 0.0
    alert_types = {alert["alert_type"] for alert in interaction.interaction_alerts}
    assert "safe_prompt_unsafe_output" in alert_types


def test_session_threshold_applies_to_current_message_immediately():
    db_path = _tmp_file("tmp_audit", ".db")
    orchestrator = SemanticFirewallOrchestrator(db_path=str(db_path))
    orchestrator.agents = {
        "PII Detector": HighSeverityAgent("PII Detector", "PII", severity="HIGH"),
        "Secrets Detector": CleanAgent("Secrets Detector", "SECRET"),
        "Abuse Detector": CleanAgent("Abuse Detector", "ABUSE"),
        "Injection Detector": CleanAgent("Injection Detector", "INJECTION"),
        "Unsafe Content Detector": CleanAgent("Unsafe Content Detector", "UNSAFE_CONTENT"),
        "Threat Intel Detector": CleanAgent("Threat Intel Detector", "THREAT_INTEL"),
        "Custom Rules Detector": CleanAgent("Custom Rules Detector", "CUSTOM_RULE"),
    }

    session_id = f"session-{uuid4().hex}"
    orchestrator.session_store.threat_scores[session_id] = 9.5

    decision = orchestrator.analyze("alice@example.com and other risky info", session_id=session_id)

    assert decision.action == "BLOCK"
    assert "Session BLOCKED" in decision.reason
    assert orchestrator.session_store.get_threat_score(session_id) >= 12.5


class StubFirewall:
    def __init__(self):
        self.audit_logger = AuditLogger(db_path=str(_tmp_file("tmp_api_audit", ".db")))
        self.cleared_sessions = []
        self.saved_presets = {}
        self.saved_policy_drafts = {}

    def analyze(self, text: str, session_id=None, policy_profile="balanced", workspace_id="default"):
        return FirewallDecision(
            action="FLAG",
            reason="policy review",
            severity="LOW",
            triggered_agents=["PII Detector"],
            agent_results=[
                AgentResult(
                    agent_name="PII Detector",
                    threat_found=True,
                    threat_type="PII",
                    severity="LOW",
                    summary="pii found",
                    matched=[],
                    agent_available=True,
                    fail_closed=False,
                )
            ],
            redacted_text=text,
            processing_time_ms=3.5,
            original_text=text,
            scan_target="input",
            from_cache=False,
            degraded=False,
            unavailable_agents=[],
            policy_profile=policy_profile,
        )

    def analyze_output(self, text: str, policy_profile="balanced", workspace_id="default"):
        return FirewallDecision(
            action="BLOCK",
            reason="unsafe output",
            severity="HIGH",
            triggered_agents=["Unsafe Content Detector"],
            agent_results=[
                AgentResult(
                    agent_name="Unsafe Content Detector",
                    threat_found=True,
                    threat_type="UNSAFE_CONTENT",
                    severity="HIGH",
                    summary="unsafe content found",
                    matched=[],
                    agent_available=True,
                    fail_closed=False,
                )
            ],
            redacted_text=text,
            processing_time_ms=4.2,
            original_text=text,
            scan_target="output",
            from_cache=False,
            degraded=False,
            unavailable_agents=[],
            policy_profile=policy_profile,
        )

    def _get_redacted_text(self, text: str, agent_results):
        return text.replace("alice@example.com", "[EMAIL]")

    def analyze_interaction(self, prompt_text: str, output_text: str, session_id=None, policy_profile="balanced", workspace_id="default"):
        prompt_decision = self.analyze(prompt_text, session_id=session_id, policy_profile=policy_profile, workspace_id=workspace_id)
        output_decision = self.analyze_output(output_text, policy_profile=policy_profile, workspace_id=workspace_id)
        return InteractionDecision(
            prompt_decision=prompt_decision,
            output_decision=output_decision,
            combined_action="BLOCK",
            combined_severity="HIGH",
            combined_reason="Prompt is risky and output is unsafe.",
            triggered_agents=["prompt:PII Detector", "output:Unsafe Content Detector"],
            interaction_alerts=[
                {
                    "alert_type": "safe_prompt_unsafe_output",
                    "severity": "CRITICAL",
                    "summary": "A relatively clean prompt produced a materially riskier output.",
                    "evidence": "unsafe output",
                }
            ],
            drift_detected=True,
            contradiction_detected=True,
            session_risk_score=3.5,
        )

    def get_session_summary(self, session_id: str):
        return {
            "session_id": session_id,
            "message_count": 4,
            "cumulative_score": 6.5,
            "should_flag": False,
            "should_block": False,
        }

    def clear_session(self, session_id: str):
        self.cleared_sessions.append(session_id)

    def list_policy_presets(self, workspace_id=None):
        base = [
            {
                "name": "balanced",
                "description": "Default",
                "action_overrides": {},
                "session_flag_threshold": 8.0,
                "session_block_threshold": 12.0,
                "allowlist_patterns": [],
                "allowlist_max_severity": "LOW",
                "enabled": True,
            }
        ]
        return base + list(self.saved_presets.values()) + list(self.saved_policy_drafts.values())

    def list_policy_drafts(self, workspace_id=None):
        return list(self.saved_policy_drafts.values())

    def save_policy_preset(self, **kwargs):
        self.saved_presets[kwargs["name"]] = dict(kwargs)
        return self.saved_presets[kwargs["name"]]

    def delete_policy_preset(self, name: str, workspace_id=None):
        return (
            self.saved_presets.pop(name, None) is not None
            or self.saved_policy_drafts.pop(name, None) is not None
        )

    def create_policy_draft_from_adjustment(self, adjustment: dict, workspace_id: str = "global"):
        name = f"{adjustment['profile']}_draft_{adjustment['agent'].lower().replace(' ', '_')}"
        draft = {
            "name": name,
            "description": "Draft from adjustment",
            "action_overrides": {},
            "session_flag_threshold": 7.0,
            "session_block_threshold": 11.0,
            "allowlist_patterns": [],
            "allowlist_max_severity": "LOW",
            "enabled": False,
            "status": "draft",
        }
        self.saved_policy_drafts[name] = draft
        return draft

    def promote_policy_draft(self, name: str, note: str = "", workspace_id: str = "default"):
        draft = self.saved_policy_drafts.get(name)
        if not draft:
            return None
        draft["status"] = "active"
        draft["enabled"] = True
        self.saved_presets[name] = draft
        self.saved_policy_drafts.pop(name, None)
        return draft

    def preview_policy_draft(self, name: str, workspace_id=None):
        draft = self.saved_policy_drafts.get(name)
        if not draft:
            return None
        return {
            "draft": draft,
            "baseline": {"name": "balanced"},
            "changes": [{"field": "session_flag_threshold", "before": 8.0, "after": 7.0}],
            "base_profile": "balanced",
        }

    def get_promotion_history(self, limit: int = 50, workspace_id: str | None = None):
        return self.audit_logger.get_promotion_history(limit=limit, workspace_id=workspace_id)

    def request_policy_approval(self, name: str, requested_by: str, workspace_id: str = "global"):
        draft = self.saved_policy_drafts.get(name)
        if not draft:
            return None
        draft["approval_status"] = "pending_approval"
        draft["approval_requested_by"] = requested_by
        return draft

    def approve_policy_draft(self, name: str, approved_by: str, workspace_id: str = "global"):
        draft = self.saved_policy_drafts.get(name)
        if not draft:
            return None
        if draft.get("approval_requested_by") == approved_by:
            raise ValueError("Approver must be different from requester.")
        draft["approval_status"] = "approved"
        draft["approval_approved_by"] = approved_by
        return draft


def test_api_returns_redacted_text_for_flagged_requests(monkeypatch):
    monkeypatch.setattr(api_module, "firewall", StubFirewall())
    client = TestClient(api_module.app)

    response = client.post(
        "/analyze",
        json={
            "text": "Email alice@example.com for help.",
            "redact_on_flag": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "FLAG"
    assert body["redacted_text"] == "Email [EMAIL] for help."
    assert body["degraded"] is False
    assert body["scan_target"] == "input"
    assert body["policy_profile"] == "balanced"
    assert body["top_findings"] == []


def test_api_scans_model_output(monkeypatch):
    monkeypatch.setattr(api_module, "firewall", StubFirewall())
    client = TestClient(api_module.app)

    response = client.post(
        "/analyze/output",
        json={
            "text": "Here are instructions for making a bomb.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "BLOCK"
    assert body["scan_target"] == "output"
    assert body["triggered_agents"] == ["Unsafe Content Detector"]
    assert body["policy_profile"] == "balanced"
    assert body["top_findings"] == []


def test_api_scans_prompt_output_interaction(monkeypatch):
    monkeypatch.setattr(api_module, "firewall", StubFirewall())
    client = TestClient(api_module.app)

    response = client.post(
        "/analyze/interaction",
        json={
            "prompt_text": "Email alice@example.com for help.",
            "output_text": "Here are instructions for making a bomb.",
            "session_id": "session-123",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["combined_action"] == "BLOCK"
    assert body["combined_severity"] == "HIGH"
    assert body["triggered_agents"] == ["prompt:PII Detector", "output:Unsafe Content Detector"]
    assert body["drift_detected"] is True
    assert body["contradiction_detected"] is True
    assert body["session_risk_score"] == 3.5
    assert body["interaction_alerts"][0]["alert_type"] == "safe_prompt_unsafe_output"
    assert body["prompt_decision"]["scan_target"] == "input"
    assert body["output_decision"]["scan_target"] == "output"


def test_api_creates_and_lists_custom_rules(monkeypatch):
    client = TestClient(api_module.app)

    create_response = client.post(
        "/rules",
        json={
            "name": "Test Rule",
            "pattern": "danger phrase",
            "description": "Test rule",
            "severity": "LOW",
            "threat_type": "CUSTOM_RULE",
            "scope": "both",
            "redact": False,
        },
    )
    assert create_response.status_code == 200
    rule_id = create_response.json()["rule"]["id"]

    list_response = client.get("/rules")
    assert list_response.status_code == 200
    assert any(rule["id"] == rule_id for rule in list_response.json()["rules"])

    delete_response = client.delete(f"/rules/{rule_id}")
    assert delete_response.status_code == 200


def test_api_updates_and_tests_custom_rule():
    client = TestClient(api_module.app)

    create_response = client.post(
        "/rules",
        json={
            "name": "Allowlist Rule",
            "pattern": "Project Falcon",
            "description": "Test allowlist",
            "severity": "HIGH",
            "scope": "both",
            "redact": False,
            "exceptions": ["public demo"],
            "tags": ["internal"],
        },
    )
    assert create_response.status_code == 200
    rule = create_response.json()["rule"]

    update_response = client.put(
        f"/rules/{rule['id']}",
        json={"severity": "LOW", "tags": ["internal", "demo"]},
    )
    assert update_response.status_code == 200
    assert update_response.json()["rule"]["severity"] == "LOW"
    assert "demo" in update_response.json()["rule"]["tags"]

    test_response = client.post(
        "/rules/test",
        json={"rule_id": rule["id"], "text": "This public demo mentions Project Falcon."},
    )
    assert test_response.status_code == 200
    assert test_response.json()["excepted"] is True


def test_api_rule_draft_endpoints(monkeypatch):
    stub_firewall = StubFirewall()
    stub_firewall.audit_logger.log(
        input_text="Project Zephyr details leaked",
        action="FLAG",
        severity="MEDIUM",
        triggered_agents=["Custom Rules Detector"],
        reason="Needs review",
        processing_time_ms=4.0,
        matched_threats={},
        scan_target="input",
        policy_profile="balanced",
    )
    audit_id = stub_firewall.audit_logger.get_recent(limit=1)[0]["id"]
    stub_firewall.audit_logger.submit_feedback(audit_id, "false_negative", 'Please catch "Project Zephyr" next time')
    monkeypatch.setattr(api_module, "firewall", stub_firewall)
    client = TestClient(api_module.app)

    draft_response = client.post("/rules/suggestions/draft")
    assert draft_response.status_code == 200
    draft_id = draft_response.json()["draft"]["id"]

    list_response = client.get("/rules/drafts")
    assert list_response.status_code == 200
    assert any(item["id"] == draft_id for item in list_response.json()["drafts"])

    preview_response = client.get(f"/rules/drafts/{draft_id}/preview")
    assert preview_response.status_code == 200
    assert preview_response.json()["changes"]

    promote_response = client.post(f"/rules/{draft_id}/promote", json={"note": "ship it", "workspace_id": "default"})
    assert promote_response.status_code == 200
    assert promote_response.json()["rule"]["status"] == "active"


def test_api_batch_scan_and_session_endpoints(monkeypatch):
    stub_firewall = StubFirewall()
    monkeypatch.setattr(api_module, "firewall", stub_firewall)
    client = TestClient(api_module.app)

    batch_response = client.post(
        "/analyze/batch",
        json={"texts": ["Email alice@example.com", "bomb instructions"], "scan_target": "output"},
    )
    assert batch_response.status_code == 200
    body = batch_response.json()
    assert body["total"] == 2
    assert body["scan_target"] == "output"
    assert body["policy_profile"] == "balanced"
    assert body["results"][0]["scan_target"] == "output"

    session_response = client.get("/sessions/session-42")
    assert session_response.status_code == 200
    assert session_response.json()["session_id"] == "session-42"
    assert session_response.json()["message_count"] == 4

    clear_response = client.delete("/sessions/session-42")
    assert clear_response.status_code == 200
    assert stub_firewall.cleared_sessions == ["session-42"]


def test_api_document_scan(monkeypatch):
    stub_firewall = StubFirewall()
    monkeypatch.setattr(api_module, "firewall", stub_firewall)
    client = TestClient(api_module.app)

    response = client.post(
        "/analyze/document",
        json={
            "filename": "note.txt",
            "content_base64": base64.b64encode(b"Email alice@example.com").decode("utf-8"),
            "scan_target": "input",
            "policy_profile": "balanced",
            "redact_on_flag": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "note.txt"
    assert body["extension"] == ".txt"
    assert body["extraction_mode"] == "native_text"
    assert body["extraction_quality"] == "high"
    assert body["decision"]["scan_target"] == "input"


def test_policy_preset_endpoints(monkeypatch):
    stub_firewall = StubFirewall()
    monkeypatch.setattr(api_module, "firewall", stub_firewall)
    client = TestClient(api_module.app)

    create_response = client.post(
        "/policies",
        json={
            "name": "enterprise_guard",
            "description": "Enterprise preset",
            "action_overrides": {"PII": {"LOW": "REDACT"}},
            "session_flag_threshold": 5.0,
            "session_block_threshold": 8.0,
            "allowlist_patterns": ["approved demo"],
            "allowlist_max_severity": "LOW",
            "enabled": True,
        },
    )
    assert create_response.status_code == 200
    assert create_response.json()["preset"]["name"] == "enterprise_guard"

    list_response = client.get("/policies")
    assert list_response.status_code == 200
    assert any(item["name"] == "enterprise_guard" for item in list_response.json()["presets"])

    delete_response = client.delete("/policies/enterprise_guard")
    assert delete_response.status_code == 200


def test_policy_adjustment_draft_endpoints(monkeypatch):
    stub_firewall = StubFirewall()
    stub_firewall.audit_logger.log(
        input_text="Need stronger pii policy",
        action="FLAG",
        severity="LOW",
        triggered_agents=["PII Detector"],
        reason="Needs review",
        processing_time_ms=2.0,
        matched_threats={},
        scan_target="input",
        policy_profile="balanced",
    )
    audit_id = stub_firewall.audit_logger.get_recent(limit=1)[0]["id"]
    stub_firewall.audit_logger.submit_feedback(audit_id, "false_negative", "PII should be stricter")
    monkeypatch.setattr(api_module, "firewall", stub_firewall)
    client = TestClient(api_module.app)

    create_response = client.post("/feedback/adjustments/draft")
    assert create_response.status_code == 200
    draft_name = create_response.json()["draft"]["name"]

    list_response = client.get("/policies/drafts")
    assert list_response.status_code == 200
    assert any(item["name"] == draft_name for item in list_response.json()["drafts"])

    preview_response = client.get(f"/policies/drafts/{draft_name}/preview")
    assert preview_response.status_code == 200
    assert preview_response.json()["changes"]

    promote_response = client.post(f"/policies/{draft_name}/promote", json={"note": "looks good", "workspace_id": "default"})
    assert promote_response.status_code == 200
    assert promote_response.json()["preset"]["status"] == "active"


def test_policy_approval_endpoints(monkeypatch):
    stub_firewall = StubFirewall()
    stub_firewall.saved_policy_drafts["demo_policy"] = {
        "name": "demo_policy",
        "status": "draft",
        "enabled": False,
    }
    monkeypatch.setattr(api_module, "firewall", stub_firewall)
    client = TestClient(api_module.app)

    request_response = client.post("/policies/demo_policy/request-approval", json={"actor": "alice", "workspace_id": "default"})
    assert request_response.status_code == 200

    approve_response = client.post("/policies/demo_policy/approve", json={"actor": "bob", "workspace_id": "default"})
    assert approve_response.status_code == 200
    assert approve_response.json()["draft"]["approval_status"] == "approved"


def test_rule_approval_endpoints():
    rules_path = _tmp_file("tmp_rules", ".json")
    manager = CustomRulesManager(str(rules_path))
    draft = manager.create_draft_from_suggestion(
        {
            "name": "Suggested rule: Needs Approval",
            "pattern": r"needs\ approval",
            "description": "Approval flow test.",
            "severity": "LOW",
            "scope": "both",
            "tags": ["suggested"],
            "support": 1,
        },
        workspace_id="alpha",
    )

    requested = manager.request_approval(draft["id"], "alice")
    approved = manager.approve_draft(draft["id"], "bob")

    assert requested["approval_status"] == "pending_approval"
    assert approved["approval_status"] == "approved"


def test_promotions_endpoint(monkeypatch):
    stub_firewall = StubFirewall()
    stub_firewall.audit_logger.log_promotion(
        item_type="rule",
        item_id="rule-123",
        item_name="Demo Rule",
        note="approved",
        workspace_id="alpha",
    )
    monkeypatch.setattr(api_module, "firewall", stub_firewall)
    client = TestClient(api_module.app)

    response = client.get("/promotions", params={"workspace_id": "alpha"})

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 1
    assert items[0]["item_name"] == "Demo Rule"


def test_feedback_insights_and_rule_suggestions():
    logger = AuditLogger(db_path=str(_tmp_file("tmp_api_audit", ".db")))
    logger.log(
        input_text="Project Zephyr details leaked",
        action="FLAG",
        severity="MEDIUM",
        triggered_agents=["Custom Rules Detector"],
        reason="Needs review",
        processing_time_ms=4.0,
        matched_threats={},
        scan_target="input",
        policy_profile="strict",
    )
    audit_id = logger.get_recent(limit=1)[0]["id"]
    logger.submit_feedback(audit_id, "false_negative", 'Please catch "Project Zephyr" next time')

    insights = logger.get_feedback_insights()
    suggestions = logger.suggest_rules_from_feedback(limit=3)

    assert insights["feedback_counts"]["false_negative"] == 1
    assert insights["profile_counts"]["strict"] == 1
    assert suggestions
    assert "Project\\ Zephyr" in suggestions[0]["pattern"] or "Project Zephyr" in suggestions[0]["name"]


def test_review_queue_endpoints(monkeypatch):
    stub_firewall = StubFirewall()
    stub_firewall.audit_logger.log(
        input_text="sanitized text",
        action="FLAG",
        severity="LOW",
        triggered_agents=["PII Detector"],
        reason="Needs human review",
        processing_time_ms=5.0,
        matched_threats={"PII Detector": ["email"]},
        scan_target="input",
        session_id=None,
    )
    monkeypatch.setattr(api_module, "firewall", stub_firewall)
    client = TestClient(api_module.app)

    queue_response = client.get("/review-queue")
    assert queue_response.status_code == 200
    items = queue_response.json()["items"]
    assert len(items) == 1
    assert items[0]["review_status"] == "pending"

    audit_log_id = items[0]["id"]
    update_response = client.post(
        f"/review-queue/{audit_log_id}",
        json={
            "review_status": "reviewed_true_positive",
            "review_assignee": "aksha",
            "review_notes": "confirmed true positive",
        },
    )
    assert update_response.status_code == 200
    assert update_response.json()["review_assignee"] == "aksha"

    refreshed_queue = client.get("/review-queue")
    assert refreshed_queue.status_code == 200
    assert refreshed_queue.json()["items"] == []


def test_workspaces_endpoints():
    client = TestClient(api_module.app)
    name = f"team_{uuid4().hex[:8]}"

    create_response = client.post(
        "/workspaces",
        json={
            "name": name,
            "description": "Safety reviews for a product team",
            "owner": "akira",
        },
    )
    assert create_response.status_code == 200

    list_response = client.get("/workspaces")
    assert list_response.status_code == 200
    assert any(item["name"] == name for item in list_response.json()["workspaces"])

    delete_response = client.delete(f"/workspaces/{name}")
    assert delete_response.status_code == 200


def test_feedback_insights_and_suggestions_endpoints(monkeypatch):
    stub_firewall = StubFirewall()
    stub_firewall.audit_logger.log(
        input_text="Project Zephyr details leaked",
        action="FLAG",
        severity="MEDIUM",
        triggered_agents=["Custom Rules Detector"],
        reason="Needs review",
        processing_time_ms=4.0,
        matched_threats={},
        scan_target="input",
        policy_profile="strict",
    )
    audit_id = stub_firewall.audit_logger.get_recent(limit=1)[0]["id"]
    stub_firewall.audit_logger.submit_feedback(audit_id, "false_negative", 'Please catch "Project Zephyr" next time')
    monkeypatch.setattr(api_module, "firewall", stub_firewall)
    client = TestClient(api_module.app)

    insights_response = client.get("/feedback/insights")
    assert insights_response.status_code == 200
    assert insights_response.json()["feedback_counts"]["false_negative"] == 1

    suggestions_response = client.get("/rules/suggestions")
    assert suggestions_response.status_code == 200
    assert suggestions_response.json()["suggestions"]

    adjustments_response = client.get("/feedback/adjustments")
    assert adjustments_response.status_code == 200
    assert "adjustments" in adjustments_response.json()



