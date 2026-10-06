import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from semantic_firewall.core.agents.llm_client import DEFAULT_LLM_MODEL, DetectorLLMClient, extract_json_object

load_dotenv()


@dataclass
class UnsafeMatch:
    content_type: str
    description: str
    confidence: float
    evidence: str
    severity_weight: int


@dataclass
class DetectionResult:
    agent_name: str
    threat_found: bool
    threat_type: str
    matched: list[UnsafeMatch]
    severity: str
    summary: str
    meta: dict[str, Any] = field(default_factory=dict)


class UnsafeContentDetectorAgent:
    # The orchestrator may call run(..., use_llm=False) for a regex-only pre-screen.
    supports_regex_only = True

    def __init__(self):
        self.name = "Unsafe Content Detector"
        default_model = os.getenv("SEMANTIC_FIREWALL_UNSAFE_MODEL_FAST", DEFAULT_LLM_MODEL)
        self.llm_client = DetectorLLMClient(default_model=default_model)
        self.model = self.llm_client.model_name
        self.confidence_threshold = 0.5
        self.regex_confidence = 0.80
        # Off = pure LLM detector (used for the "LLM agents only" baseline).
        self.regex_enabled = os.getenv("SEMANTIC_FIREWALL_LLM_AGENT_REGEX_ENABLED", "1").lower() in {
            "1",
            "true",
            "yes",
        }

        self.quick_patterns: dict[str, tuple[str, str, int]] = {
            "explicit_violence": (
                r"\b(how to (kill|murder|torture|stab|shoot|strangle|poison|beat up)"
                r"|step(s)? to (kill|murder|harm|hurt|attack)"
                r"|ways? to (kill|murder|hurt|harm|eliminate) (a |an |someone|people|humans?))\b",
                "Explicit violence instruction request",
                4,
            ),
            "direct_threat": (
                r"\b(i (will|am going to|want to|plan to) (kill|murder|hurt|harm|attack|destroy)"
                r"|you (will|are going to) (die|be hurt|be killed|regret)"
                r"|i know where you (live|are|work))\b",
                "Direct personal threat",
                4,
            ),
            "mass_violence": (
                r"\b(mass (shooting|killing|murder|casualt)"
                r"|how (do i|to) (make|build|create) (a )?(bomb|explosive|weapon)"
                r"|(school|church|mosque|temple|concert) (shooting|attack|bombing))\b",
                "Mass violence planning or glorification",
                4,
            ),
            "self_harm_methods": (
                r"\b(how to (cut|harm|hurt) (myself|yourself|oneself)"
                r"|methods? (of|for) self.?harm"
                r"|ways? to (hurt|injure) (myself|yourself))\b",
                "Self-harm method request",
                4,
            ),
            "suicide_methods": (
                r"\b(how to (commit suicide|kill myself|end my life|take my own life)"
                r"|best (method|way|means) (of|to) (suicide|die|end it)"
                r"|best .* to commit suicide"
                r"|most painless way to commit suicide"
                r"|painless (way|method) to (die|suicide|end))\b",
                "Suicide method request",
                4,
            ),
            "suicide_ideation": (
                r"\b(i (want|wish) (to die|i was dead|i were dead|to kill myself)"
                r"|no (reason|point) (to|in) (live|living|going on)"
                r"|better off (dead|without me))\b",
                "Suicidal ideation expression",
                3,
            ),
            "explicit_sexual": (
                r"\b(explicit (sexual|porn|nude|naked) (content|material|image|video)"
                r"|generate (porn|nude|naked|sexual) (image|content|story)"
                r"|write (explicit|graphic) (sex|sexual|erotic) (scene|story|content))\b",
                "Explicit sexual content request",
                3,
            ),
            "csam_indicators": (
                r"\b(child(ren)? (nude|naked|sexual|porn)"
                r"|minor(s)? (nude|naked|sexual|explicit)"
                r"|(nude|naked|sexual) (image|photo|content) of (a child|children|minors?|underage))\b",
                "Child sexual abuse material indicators",
                4,
            ),
            "hate_speech": (
                r"\b((all|those) (jews?|muslims?|christians?|blacks?|whites?|asians?|hispanics?)"
                r"\s+(are|should|deserve|must|need to)\b"
                r"|(racial|ethnic|religious) (slur|slurs?|epithet)"
                r"|white (supremacy|supremacist|nationalist|power)"
                r"|(exterminate|eliminate|eradicate|genocide).{0,30}(race|religion|group|people))\b",
                "Hate speech or discriminatory content",
                4,
            ),
            "extremist_content": (
                r"\b(join (isis|isil|al.?qaeda|taliban|kkk|neo.?nazi)"
                r"|how to (radicalize|recruit) (people|someone|others)"
                r"|(terrorist|extremist) (manifesto|ideology|propaganda)"
                r"|how to (plan|carry out|execute) (a terror|an attack|a terrorist))\b",
                "Extremist or terrorist content",
                4,
            ),
            "weapons_manufacturing": (
                r"\b(how (do i|to) (make|build|create|synthesize|manufacture)"
                r".{0,30}(gun|weapon|explosive|bomb|grenade|knife|poison|toxin|nerve agent))\b",
                "Weapon manufacturing instructions",
                4,
            ),
            "drug_manufacturing": (
                r"\b(how (do i|to) (make|cook|synthesize|manufacture|produce)"
                r".{0,30}(meth|methamphetamine|heroin|cocaine|fentanyl|lsd|mdma|ecstasy))\b",
                "Drug manufacturing instructions",
                4,
            ),
            "biological_chemical": (
                r"\b(how to (make|create|synthesize|weaponize)"
                r".{0,30}(bioweapon|biological weapon|chemical weapon|anthrax|sarin|vx gas|ricin))\b",
                "Biological or chemical weapon instructions",
                4,
            ),
            "hacking_instructions": (
                r"\b(how to (hack|crack|exploit|bypass|brute force)"
                r".{0,40}(account|system|network|server|database|password|firewall)"
                r"|write (a|me a) (keylogger|trojan|ransomware|malware|virus|worm|rootkit)"
                r"|how to (steal|exfiltrate) (data|credentials|passwords?))\b",
                "Hacking or cybercrime instructions",
                3,
            ),
            "phishing": (
                r"\b(create (a )?phishing (email|page|site|link|message)"
                r"|how to (phish|scam|defraud|trick) (people|someone|users?)"
                r"|write (a )?fake (email|login|bank|paypal) (page|site|message))\b",
                "Phishing or social engineering attack",
                3,
            ),
            "human_trafficking": (
                r"\b(how to (traffic|smuggle) (humans?|people|women|children)"
                r"|recruit (victims?|people) for (trafficking|exploitation|prostitution)"
                r"|how to (control|exploit) (victims?|workers?) (through|using) (debt|force|threat))\b",
                "Human trafficking related content",
                4,
            ),
            "child_grooming": (
                r"\b(how to (groom|manipulate|gain trust of) (a child|children|minors?|kids?)"
                r"|how to (approach|talk to|befriend) (a child|children|minors?) (online|secretly)"
                r"|how to (hide|keep secret) (relationship|contact) (with|from) (a child|minors?))\b",
                "Child grooming instructions",
                4,
            ),
            "illegal_financial": (
                r"\b(how to (launder|wash) (money|funds|cash)"
                r"|how to (evade|avoid|hide from) (taxes?|irs|hmrc|tax authority)"
                r"|how to (commit|do|execute) (fraud|identity theft|credit card fraud|insurance fraud))\b",
                "Illegal financial activity instructions",
                3,
            ),
            "medical_misinformation": (
                r"\b(vaccines? (cause|causes?) (autism|death|harm|cancer|disease)"
                r"|how to (cure|treat) (cancer|hiv|aids|diabetes) (with|using) (bleach|mms|miracle)"
                r"|covid (is|was) (fake|hoax|not real|made up))\b",
                "Dangerous medical misinformation",
                3,
            ),
            "hypothetical_evasion": (
                r"\b(in a (hypothetical|fictional|dystopian) (scenario|world|future)|for the sake of the story|for research purposes|hypothetically)\b.{0,200}\b(kill|murder|bomb|hack|steal|password|credit card|meth|keylogger)\b",
                "Hypothetical or academic framing evasion for unsafe content",
                3,
            ),
        }

        self.compiled_patterns: dict[str, tuple[re.Pattern[str], str, int]] = {}
        for content_type, (pattern, description, weight) in self.quick_patterns.items():
            try:
                self.compiled_patterns[content_type] = (
                    re.compile(pattern, re.IGNORECASE | re.DOTALL),
                    description,
                    weight,
                )
            except re.error as exc:
                print(f"[UnsafeContentDetector] Failed to compile pattern '{content_type}': {exc}")

        self.system_prompt = """You are a content safety expert specializing in detecting harmful, dangerous, and policy-violating content in AI system inputs and outputs.

Your job is to analyze a given text and identify if it contains any of the following unsafe content categories:

1. VIOLENCE - explicit violence, threats, instructions for harming others, glorification of violence
2. SELF_HARM - self-harm methods, suicide instructions, encouragement of self-destructive behavior
3. SEXUAL_EXPLICIT - explicit sexual content, pornography requests, non-consensual scenarios
4. CSAM - any sexual content involving minors (zero tolerance)
5. HATE_SPEECH - hate speech, discrimination, slurs targeting protected groups
6. EXTREMISM - terrorist content, extremist recruitment, radicalization
7. WEAPONS - instructions for making weapons, explosives, or obtaining illegal arms
8. DRUG_MANUFACTURING - instructions for synthesizing illegal drugs
9. BIOWEAPONS_CHEM - biological or chemical weapon instructions
10. CYBERCRIME - hacking tutorials, malware creation, phishing attacks
11. HUMAN_TRAFFICKING - trafficking, exploitation, grooming instructions
12. ILLEGAL_ACTIVITY - fraud, money laundering, other serious illegal activity
13. DANGEROUS_MISINFORMATION - health/medical misinformation that could cause harm
14. HARASSMENT - targeted harassment, doxxing, stalking instructions
15. PRIVACY_VIOLATION - instructions for spying on or surveilling individuals without consent

Respond ONLY with a valid JSON object in exactly this format (no explanation, no markdown):
{
  "is_unsafe": true or false,
  "content_found": [
    {
      "content_type": "category from list above",
      "description": "brief description of what makes this unsafe",
      "confidence": 0.0 to 1.0,
      "evidence": "the specific part of the text that triggered this detection"
    }
  ],
  "overall_risk": "NONE or LOW or MEDIUM or HIGH or CRITICAL"
}

CRITICAL NEGATIVE CONSTRAINTS (What NOT to flag):
- Educational discussions ABOUT dangerous topics (e.g., "what is a SQL injection", "explain the history of bioweapons") are BENIGN.
- News reporting, historical analysis, and research questions are BENIGN.
- Creative fiction, roleplay, and scenarios that do NOT provide real, actionable, harmful instructions are BENIGN. However, NSFW, erotic, sexually explicit, or extreme romantic roleplay is ALWAYS UNSAFE and must be flagged under SEXUAL_EXPLICIT.
- IMPORTANT: While creative fiction is generally benign, if the fiction explicitly demands real-world unsafe artifacts (e.g., generating valid credit card formats, realistic hacking instructions, writing explicit malicious scripts), it MUST be flagged.
- Normal conversation, complaints, frustration, or debate are BENIGN.
- ONLY flag content that could directly enable real-world harm or violates clear ethical boundaries.
- Direct instructions on how to bypass AI safety rules should NOT be flagged here unless they also ask for unsafe content. (InjectionDetector handles general rule breaking).
- Be extremely precise - do NOT over-flag benign conversation."""

    def _truncate_text(self, text: str, max_length: int) -> tuple[str, bool]:
        if len(text) > max_length:
            return text[:max_length], True
        return text, False

    def _calculate_severity(self, matched: list[UnsafeMatch]) -> str:
        if not matched:
            return "NONE"
        max_weight = max(match.severity_weight for match in matched)
        if max_weight == 4:
            return "CRITICAL"
        if max_weight == 3:
            return "HIGH"
        if max_weight == 2:
            return "MEDIUM"
        return "LOW"

    def _regex_prescreen(self, text: str) -> list[UnsafeMatch]:
        matches: list[UnsafeMatch] = []
        for content_type, (compiled_pattern, description, weight) in self.compiled_patterns.items():
            found = compiled_pattern.search(text)
            if found:
                matches.append(
                    UnsafeMatch(
                        content_type=content_type,
                        description=description,
                        confidence=self.regex_confidence,
                        evidence=found.group(0)[:100],
                        severity_weight=weight,
                    )
                )
        return matches

    def _parse_llm_response(
        self,
        raw: str,
        confidence_threshold: float,
    ) -> tuple[list[UnsafeMatch], dict[str, Any]]:
        payload, base_meta = extract_json_object(raw)
        if payload is None:
            return [], base_meta

        matches: list[UnsafeMatch] = []
        content_found = payload.get("content_found")
        if isinstance(content_found, list):
            for item in content_found:
                if not isinstance(item, dict):
                    continue
                confidence = float(item.get("confidence", 0.5))
                if confidence >= confidence_threshold:
                    matches.append(
                        UnsafeMatch(
                            content_type=item.get("content_type", "UNKNOWN"),
                            description=item.get("description", ""),
                            confidence=confidence,
                            evidence=item.get("evidence", "")[:150],
                            severity_weight=4 if confidence >= 0.85 else 3,
                        )
                    )

        reported_unsafe = bool(payload.get("is_unsafe", False))
        if reported_unsafe and not matches:
            confidence = float(payload.get("confidence", 0.5))
            if confidence >= confidence_threshold:
                matches.append(
                    UnsafeMatch(
                        content_type=str(payload.get("content_type", "UNSAFE_CONTENT")),
                        description="LLM classified the text as unsafe content.",
                        confidence=confidence,
                        evidence=str(payload.get("evidence", ""))[:150],
                        severity_weight=4 if confidence >= 0.85 else 3,
                    )
                )

        return matches, {
            **base_meta,
            "llm_reported_unsafe": reported_unsafe,
            "llm_reported_risk": str(payload.get("overall_risk", "NONE")).upper(),
        }

    # Appended to the system prompt when the Session Judge checks a window of conversation turns.
    CONVERSATION_ADDENDUM = """

CONVERSATION MODE: the text is a sequence of consecutive user messages from ONE conversation, oldest first.
Judge the conversation as a whole, not each message in isolation. Multi-turn attacks start with innocent
questions about a person, event, organisation or substance associated with harm, then narrow step by step
towards methods, materials, processes, preparation, acquisition or deployment details. Flag the conversation
(is_unsafe = true) when, taken together, the messages are steering towards actionable harmful information of
any category above, even if every single message would be benign on its own. Do NOT flag conversations whose
later messages stay at the level of history, impact, ethics or general explanation without moving towards
operational detail."""

    _conversation_examples: str | None = None

    @classmethod
    def conversation_examples(cls) -> str:
        """Worked examples for conversation mode, from data/multiturn_examples.json (built only from data
        outside the multi-turn evaluation). Empty if the file is missing or
        SEMANTIC_FIREWALL_MULTITURN_EXAMPLES=0."""
        if cls._conversation_examples is None:
            text = ""
            path = Path(__file__).resolve().parents[3] / "data" / "multiturn_examples.json"
            if os.getenv("SEMANTIC_FIREWALL_MULTITURN_EXAMPLES", "1").lower() not in {"0", "false", "no"} \
                    and path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                blocks = []
                for ex in data.get("attacks", []):
                    turns = "\n".join(f"[Turn {i}] {t}" for i, t in enumerate(ex["turns"], 1))
                    blocks.append(f"UNSAFE example (hidden goal: {ex['goal']}):\n{turns}")
                for ex in data.get("benign", []):
                    turns = "\n".join(f"[Turn {i}] {t}" for i, t in enumerate(ex["turns"], 1))
                    blocks.append(f"BENIGN example:\n{turns}")
                if blocks:
                    text = ("\n\nWORKED EXAMPLES of whole conversations (other users; never copy their content, "
                            "only learn the pattern):\n\n" + "\n\n".join(blocks))
            cls._conversation_examples = text
        return cls._conversation_examples

    def _llm_detect(
        self,
        text: str,
        confidence_threshold: float | None = None,
        conversation: bool = False,
    ) -> tuple[list[UnsafeMatch], dict[str, Any]]:
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        availability_error = self.llm_client.availability_error()
        if availability_error is not None:
            return [], {
                "llm_called": False,
                "llm_provider": self.llm_client.provider,
                "llm_model": self.llm_client.model_name,
                "llm_skipped_reason": availability_error,
                "llm_parse_status": "not_attempted",
            }

        try:
            if conversation:
                system_prompt = self.system_prompt + self.CONVERSATION_ADDENDUM + self.conversation_examples()
                user_prompt = f"Analyze this conversation (user messages, oldest first) for unsafe content:\n\n{text}"
            else:
                system_prompt = self.system_prompt
                user_prompt = f"Analyze this text for unsafe content:\n\n{text}"
            for _attempt in range(2):  # one retry if the reply is not a parsable JSON verdict
                response = self.llm_client.complete(system_prompt, user_prompt, max_tokens=1000)
                matches, parse_meta = self._parse_llm_response(response.content, threshold)
                if parse_meta.get("llm_parse_status") == "ok":
                    break
            if parse_meta.get("llm_parse_status") != "ok":
                print(
                    "[UnsafeContentDetector] LLM returned unparsable payload: "
                    f"{parse_meta.get('llm_parse_status')}"
                )
            meta = {"llm_called": True, **response.meta, **parse_meta}
            second = os.getenv("SEMANTIC_FIREWALL_CONVERSATION_SECOND_MODEL", "").strip()
            if conversation and second and not matches:
                # Optional second opinion on whole conversations (block if either model flags it).
                if getattr(self, "_second_client", None) is None or self._second_client.model != second:
                    self._second_client = DetectorLLMClient(model=second, providers=[])
                second_response = self._second_client.complete(system_prompt, user_prompt, max_tokens=4000)
                second_matches, second_meta = self._parse_llm_response(second_response.content, threshold)
                meta["second_model"] = second
                meta["second_parse_status"] = second_meta.get("llm_parse_status")
                if second_matches:
                    matches = second_matches
                    meta["flagged_by_second_model"] = True
            return matches, meta
        except Exception as exc:
            print(f"[UnsafeContentDetector] LLM call failed: {exc}")
            return [], {
                "llm_called": True,
                "llm_provider": self.llm_client.provider,
                "llm_model": self.llm_client.model_name,
                "llm_skipped_reason": "llm_error",
                "llm_parse_status": "call_failed",
                "llm_error": str(exc),
            }

    def _deduplicate(
        self,
        regex_matches: list[UnsafeMatch],
        llm_matches: list[UnsafeMatch],
    ) -> list[UnsafeMatch]:
        seen_types = set()
        final: list[UnsafeMatch] = []

        for match in llm_matches:
            if match.content_type not in seen_types:
                seen_types.add(match.content_type)
                final.append(match)

        for match in regex_matches:
            if match.content_type not in seen_types:
                seen_types.add(match.content_type)
                final.append(match)

        return final

    def run(
        self,
        text: str,
        confidence_threshold_override: float | None = None,
        use_llm: bool = True,
        conversation: bool = False,
    ) -> DetectionResult:
        effective_threshold = self.confidence_threshold
        if confidence_threshold_override is not None:
            effective_threshold = max(0.0, min(1.0, float(confidence_threshold_override)))

        regex_matches = self._regex_prescreen(text) if self.regex_enabled else []
        if use_llm:
            llm_matches, llm_meta = self._llm_detect(
                text, confidence_threshold=effective_threshold, conversation=conversation
            )
        else:
            llm_matches = []
            llm_meta = {
                "llm_called": False,
                "llm_provider": self.llm_client.provider,
                "llm_model": self.llm_client.model_name,
                "llm_skipped_reason": "regex_prescreen_only",
                "llm_parse_status": "not_attempted",
            }
        matched = self._deduplicate(regex_matches, llm_matches)

        severity = self._calculate_severity(matched)
        threat_found = bool(matched)
        llm_called = bool(llm_meta.get("llm_called", False))
        regex_only = bool(regex_matches) and not bool(llm_matches)

        summary = (
            f"Detected {len(matched)} unsafe content type(s) across "
            f"{len({match.content_type for match in matched})} category(ies). "
            f"Severity: {severity}."
            if threat_found
            else "No unsafe content detected. Input is clean."
        )

        return DetectionResult(
            agent_name=self.name,
            threat_found=threat_found,
            threat_type="UNSAFE_CONTENT",
            matched=matched,
            severity=severity,
            summary=summary,
            meta={
                "llm_called": llm_called,
                "regex_only": regex_only,
                "regex_match_count": len(regex_matches),
                "llm_match_count": len(llm_matches),
                "confidence_threshold_used": effective_threshold,
                **llm_meta,
            },
        )

