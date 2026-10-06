from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class SessionContextDecision:
    action: str
    reason: str
    session_info: Dict[str, Any]
    detected_patterns: List[dict]
    triggered_agents: List[str]


class SessionJudge:
    def __init__(
        self,
        session_store: Any,
        enhanced_session_store: Any,
        multi_turn_detector: Any,
        injection_agent_getter: Callable[[], Any],
        action_rank: Callable[[str], int],
        unsafe_agent_getter: Optional[Callable[[], Any]] = None,
        llm_enabled: Callable[[], bool] = lambda: True,
        window_turns: int = 3,
    ):
        self.session_store = session_store
        self.enhanced_session_store = enhanced_session_store
        self.multi_turn_detector = multi_turn_detector
        self.injection_agent_getter = injection_agent_getter
        self.unsafe_agent_getter = unsafe_agent_getter
        self.llm_enabled = llm_enabled
        self.window_turns = window_turns
        self.action_rank = action_rank

    def _window_checks(self):
        """(label, agent getter, threat type, extra run kwargs) for the sliding-window re-check."""
        checks = [("Multi-turn Injection Detector", self.injection_agent_getter, "INJECTION", {"conversation": True})]
        if self.unsafe_agent_getter is not None:
            # conversation=True: the LLM judges the trajectory of the turns, not each turn alone
            checks.append(("Multi-turn Unsafe Content Detector", self.unsafe_agent_getter, "UNSAFE_CONTENT",
                           {"conversation": True}))
        return checks

    def _session_score_for_severity(self, severity: str) -> float:
        return float(self.session_store.score_map.get(severity, 0.0))

    def _session_action_for_pattern(self, pattern: dict) -> Optional[str]:
        confidence = float(pattern.get("confidence", 0.0))
        if confidence < 0.7:
            return None

        pattern_type = pattern.get("type")
        if pattern_type in {
            "escalating_injections",
            "credential_probing",
            "detector_evasion",
            "role_override_chain",
        }:
            return "BLOCK"
        if pattern_type in {"phased_unsafe_content", "severity_escalation"}:
            return "FLAG"
        return None

    def apply(
        self,
        session_id: str,
        text: str,
        final_action: str,
        reason: str,
        overall_severity: str,
        agent_results: List[Any],
        triggered: List[str],
        risk_score: float,
        flag_threshold: float,
        block_threshold: float,
    ) -> SessionContextDecision:
        detected_patterns: List[dict] = []
        updated_triggered = list(triggered)
        existing_score = self.session_store.get_threat_score(session_id)
        projected_score = existing_score + self._session_score_for_severity(overall_severity)

        if projected_score >= block_threshold:
            final_action = "BLOCK"
            reason += f" | Session BLOCKED: cumulative threat score {projected_score:.1f} exceeded threshold"
        elif projected_score >= flag_threshold and self.action_rank(final_action) < self.action_rank("FLAG"):
            final_action = "FLAG"
            reason += f" | Session FLAG: cumulative threat score {projected_score:.1f}"

        recent = self.session_store.get_recent_texts(session_id, n=self.window_turns)
        if recent:
            # Sliding window: re-check the recent user turns together, so an attack spread over
            # several individually harmless turns is judged as one request.
            combined_text = "\n".join(f"[Turn {i}] {t}" for i, t in enumerate(recent + [text], 1))
            for label, agent_getter, threat_type, extra in self._window_checks():
                if any(result.threat_found and result.threat_type == threat_type for result in agent_results):
                    continue
                try:
                    combined_result = agent_getter().run(combined_text, use_llm=self.llm_enabled(), **extra)
                except Exception as exc:
                    print(f"[Orchestrator] Multi-turn check failed: {exc}")
                    continue
                if combined_result.threat_found:
                    final_action = "BLOCK"
                    updated_triggered.append(label)
                    reason += f" | {label}: threat detected across conversation history"
                elif (getattr(combined_result, "meta", {}) or {}).get("llm_parse_status") == "call_failed":
                    # Same fail-closed rule as the per-turn LLM detectors.
                    if self.action_rank(final_action) < self.action_rank("FLAG"):
                        final_action = "FLAG"
                    reason += f" | {label}: LLM unavailable, fail-closed"

        self.session_store.add_message(session_id, text, final_action, overall_severity)

        try:
            self.enhanced_session_store.add_message(
                session_id=session_id,
                text=text,
                action=final_action,
                threat_types=[result.threat_type for result in agent_results if result.threat_found],
                severity=overall_severity,
                agents_triggered=updated_triggered,
                risk_score=risk_score,
            )
            patterns = self.multi_turn_detector.detect_patterns(
                session_id=session_id,
                session_store=self.enhanced_session_store,
            )
            if patterns:
                detected_patterns = patterns
                for pattern in patterns:
                    escalated_action = self._session_action_for_pattern(pattern)
                    if escalated_action and self.action_rank(escalated_action) > self.action_rank(final_action):
                        final_action = escalated_action
                        reason += f" | Multi-turn {pattern.get('type')} detected"
                    print(f"[Pattern Detected] {pattern.get('type')}: confidence={pattern.get('confidence', 0):.0%}")
        except Exception as exc:
            print(f"[Orchestrator] Multi-turn pattern detection failed: {exc}")

        session_info = self.session_store.summary(
            session_id,
            flag_threshold=flag_threshold,
            block_threshold=block_threshold,
        )
        print(
            f"[Session {session_id}] score={session_info['cumulative_score']:.1f} "
            f"msgs={session_info['message_count']}"
        )
        return SessionContextDecision(
            action=final_action,
            reason=reason,
            session_info=session_info,
            detected_patterns=detected_patterns,
            triggered_agents=updated_triggered,
        )
