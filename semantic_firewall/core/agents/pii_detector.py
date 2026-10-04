import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set


@dataclass
class PIIMatch:
    pii_type: str
    value: str
    description: str
    confidence: float


@dataclass
class DetectionResult:
    agent_name: str
    threat_found: bool
    threat_type: str
    matched: List[PIIMatch]
    severity: str  # NONE / LOW / MEDIUM / HIGH / CRITICAL
    summary: str


class PIIDetectorAgent:
    def __init__(self):
        self.name = "PII Detector"
        self.max_text_length = 100_000
        self.redaction_labels: Dict[str, str] = {
            "aadhaar": "[AADHAAR]",
            "pan_card": "[PAN_CARD]",
            "passport_india": "[PASSPORT]",
            "voter_id": "[VOTER_ID]",
            "driving_license_india": "[DRIVING_LICENSE]",
            "ssn_us": "[SSN]",
            "nhs_uk": "[NHS_NUMBER]",
            "email": "[EMAIL]",
            "phone_india": "[PHONE]",
            "phone_us": "[PHONE]",
            "phone_uk": "[PHONE]",
            "phone_generic": "[PHONE]",
            "credit_card_visa": "[CARD_NUMBER]",
            "credit_card_mastercard": "[CARD_NUMBER]",
            "credit_card_amex": "[CARD_NUMBER]",
            "credit_card_generic": "[CARD_NUMBER]",
            "cvv": "[CVV]",
            "bank_account_india": "[BANK_ACCOUNT]",
            "ifsc_code": "[IFSC_CODE]",
            "iban": "[IBAN]",
            "upi_id": "[UPI_ID]",
            "ipv4": "[IP_ADDRESS]",
            "ipv6": "[IP_ADDRESS]",
            "mac_address": "[MAC_ADDRESS]",
            "pincode_india": "[PIN_CODE]",
            "zipcode_us": "[ZIP_CODE]",
            "coordinates": "[COORDINATES]",
            "blood_group": "[BLOOD_GROUP]",
            "health_id_india": "[HEALTH_ID]",
            "vehicle_reg_india": "[VEHICLE_REGISTRATION]",
            "date_of_birth": "[DATE_OF_BIRTH]",
            "age_with_context": "[AGE]",
            "gender": "[GENDER]",
            "name_with_title": "[FULL_NAME]",
            "religion": "[RELIGION]",
            "caste": "[COMMUNITY]",
        }

        # Each pattern: (regex, description)
        self.patterns: Dict[str, Tuple[str, str]] = {

            # ── Identity ──────────────────────────────────────────────────────
            "aadhaar": (
                r'\b\d{4}\s?\d{4}\s?\d{4}\b',
                "Indian Aadhaar number"
            ),
            "pan_card": (
                r'\b[A-Z]{5}[0-9]{4}[A-Z]\b',
                "Indian PAN card number"
            ),
            "passport_india": (
                r'\b[A-Z][0-9]{7}\b',
                "Indian passport number"
            ),
            "voter_id": (
                r'\b[A-Z]{3}[0-9]{7}\b',
                "Indian Voter ID"
            ),
            "driving_license_india": (
                r'\b[A-Z]{2}[0-9]{2}\s?[0-9]{11}\b',
                "Indian driving license"
            ),
            "ssn_us": (
                r'\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b',
                "US Social Security Number"
            ),
            "nhs_uk": (
                r'\b\d{3}\s?\d{3}\s?\d{4}\b',
                "UK NHS number"
            ),

            # ── Contact ───────────────────────────────────────────────────────
            "email": (
                r'\b(?!(?:test|example|dummy)@(?:example|test|dummy)\.com)[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b',
                "Email address"
            ),
            "phone_india": (
                r'\b(?:\+91[-\s]?)?[6-9]\d{9}\b',
                "Indian mobile number"
            ),
            "phone_us": (
                r'\b(?:\+1[-\s.]?)?\(?\d{3}\)?[-\s.]?\d{3}[-\s.]?\d{4}\b',
                "US phone number"
            ),
            "phone_uk": (
                r'\b(?:\+44[-\s]?)?0?7\d{3}[-\s]?\d{6}\b',
                "UK phone number"
            ),
            "phone_generic": (
                r'\b(?:(?:\+\d{1,4}[\s-]?)?(?:\(\d{1,5}\)[\s-]?)?(?:\d{2,4}[\s-]){1,3}\d{2,4}|\+\d{7,15})\b',
                "Generic international phone number"
            ),

            # ── Financial ─────────────────────────────────────────────────────
            "credit_card_visa": (
                r'\b4[0-9]{12}(?:[0-9]{3})?\b',
                "Visa credit card number"
            ),
            "credit_card_mastercard": (
                r'\b5[1-5][0-9]{14}\b',
                "Mastercard credit card number"
            ),
            "credit_card_amex": (
                r'\b3[47][0-9]{13}\b',
                "American Express card number"
            ),
            "credit_card_generic": (
                r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
                "Generic credit/debit card number"
            ),
            "cvv": (
                r'\b(?:cvv|cvc|cvv2|cvc2|security code|card verification value)\b(?:[^0-9]{0,20}?)\b(\d{3,4})\b',
                "Card CVV/CVC code (contextual)"
            ),
            "bank_account_india": (
                r'\b(?:account|acc|acct|bank)\b(?:[^0-9]{0,25}?)\b(\d{8,18})\b',
                "Indian bank account number (contextual)"
            ),
            "ifsc_code": (
                r'\b[A-Z]{4}0[A-Z0-9]{6}\b',
                "Indian IFSC code"
            ),
            "iban": (
                r'\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b',
                "International Bank Account Number (IBAN)"
            ),
            "account_number": (
                r'\b(?:acct|account|acc(?:\s*no|#))[\s:]*[A-Za-z0-9]{8,16}\b',
                "Account Number (contextual)"
            ),
            "upi_id": (
                r'\b[a-zA-Z0-9.\-_]+@(?:ybl|okaxis|okhdfcbank|okicici|upi|sbi|paytm)\b',
                "UPI payment ID"
            ),

            # ── Networking & Tech ─────────────────────────────────────────────
            "mac_address": (
                r'\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b',
                "MAC address"
            ),
            "ipv4": (
                r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
                "IPv4 address"
            ),
            "ipv6": (
                r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
                "IPv6 address"
            ),

            "zipcode_us": (
                r'\b\d{5}(?:[-\s]\d{4})?\b',
                "US Zipcode"
            ),
            "pincode_india": (
                r'\b[1-9][0-9]{5}\b',
                "Indian PIN code"
            ),
            "coordinates": (
                r'\b[-+]?(?:[1-8]?\d(?:\.\d+)?|90(?:\.0+)?),\s*[-+]?(?:180(?:\.0+)?|(?:1[0-7]\d|[1-9]?\d)(?:\.\d+)?)\b',
                "GPS Coordinates"
            ),

            # ── Medical ───────────────────────────────────────────────────────
            "blood_group": (
                r'\b(?:AB|A|B|O)\s*[+\-]',
                "Blood group"
            ),
            "health_id_india": (
                r'\b\d{2}-\d{4}-\d{4}-\d{4}\b',
                "Indian Health ID (ABHA)"
            ),

            # ── Vehicle ───────────────────────────────────────────────────────
            "vehicle_reg_india": (
                r'\b[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}\b',
                "Indian vehicle registration number"
            ),

            # ── Personal ──────────────────────────────────────────────────────
            "date_of_birth": (
                r'\b(?:(?:19|20)\d{2}[/\-\.](?:0?[1-9]|1[0-2])[/\-\.](?:0?[1-9]|[12][0-9]|3[01])|(?:0?[1-9]|[12][0-9]|3[01])[/\-\.](?:0?[1-9]|1[0-2])[/\-\.](?:19|20)\d{2})\b',
                "Date of birth (strict format)"
            ),
            "age_with_context": (
                r'\b(?:age|aged|i am|i\'m)[\s:]*(\d{1,3})\s*(?:years?|yrs?)?\b',
                "Age with context"
            ),
            "gender": (
                r'\b(?:gender|sex)[\s:]*(male|female|transgender|non-binary|m|f)\b',
                "Gender information"
            ),
            "name_with_title": (
                r'\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Shri|Smt)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b',
                "Full name with title"
            ),
            "religion": (
                r'\b(?:religion|religious|faith|i am a?)[\s:]*(hindu|muslim|christian|sikh|buddhist|jain|jewish|parsi)\b',
                "Religion (contextual)"
            ),
            "caste": (
                r'\b(?:caste|community|belong to)[\s:]*([A-Za-z\s]{3,30})\b',
                "Caste/community (contextual)"
            ),
        }
        
        # Compile regex patterns once at initialization for performance
        self.compiled_patterns: Dict[str, Tuple[re.Pattern, str]] = {}
        for pii_type, (pattern, description) in self.patterns.items():
            try:
                # Use case-insensitive for contextual patterns
                flags = re.IGNORECASE | re.DOTALL if "contextual" in description.lower() or "context" in description.lower() else re.DOTALL
                self.compiled_patterns[pii_type] = (
                    re.compile(pattern, flags),
                    description
                )
            except re.error as e:
                print(f"[PIIDetector] Failed to compile pattern '{pii_type}': {e}")

    # Lower rank = more specific. Generic numeric patterns rank last so they never
    # relabel a value that a structured pattern (card, IBAN, phone) also matched.
    _SPECIFICITY_ORDER = (
        "email", "upi_id", "iban", "ifsc_code", "pan_card", "voter_id", "passport_india",
        "driving_license_india", "vehicle_reg_india", "health_id_india",
        "credit_card_visa", "credit_card_mastercard", "credit_card_amex", "credit_card_generic",
        "cvv", "ssn_us", "aadhaar", "phone_india", "phone_uk", "phone_us", "nhs_uk",
        "bank_account_india", "account_number", "mac_address", "ipv6", "ipv4",
        "date_of_birth", "coordinates", "phone_generic", "zipcode_us", "pincode_india",
    )

    def _specificity(self, pii_type: str) -> int:
        try:
            return self._SPECIFICITY_ORDER.index(pii_type)
        except ValueError:
            return len(self._SPECIFICITY_ORDER)

    def _truncate_text(self, text: str, max_length: int) -> Tuple[str, bool]:
        """Truncate text if it exceeds max length."""
        if len(text) > max_length:
            return text[:max_length], True
        return text, False

    def _calculate_severity(self, matched):
        if not matched:
            return "NONE"

        critical_types = {"aadhaar", "ssn_us", "credit_card_visa",
                          "credit_card_mastercard", "credit_card_amex",
                          "credit_card_generic", "cvv", "passport_india",
                          "health_id_india"}

        high_types = {"pan_card", "bank_account_india", "iban",
                      "ifsc_code", "date_of_birth", "driving_license_india"}

        types_found: Set[str] = {m.pii_type for m in matched}

        if types_found & critical_types:
            return "CRITICAL"
        elif types_found & high_types:
            return "HIGH"
        elif len(matched) >= 3:
            return "HIGH"   # combination of multiple PII = high risk
        elif len(matched) == 2:
            return "MEDIUM"
        else:
            return "LOW"

    def _confidence_for(self, pii_type: str, description: str) -> float:
        """Calculate confidence score for PII type detection.
        
        Args:
            pii_type: Type of PII detected
            description: Pattern description
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # High confidence patterns (structured format)
        if pii_type in {
            "aadhaar", "pan_card", "passport_india", "voter_id", "driving_license_india",
            "ssn_us", "nhs_uk", "ifsc_code", "health_id_india", "vehicle_reg_india",
            "email", "credit_card_visa", "credit_card_mastercard", "credit_card_amex",
            "ipv4", "ipv6", "mac_address",
        }:
            return 0.95
        # Medium-high confidence
        if pii_type in {
            "credit_card_generic", "cvv", "upi_id", "bank_account_india", "iban",
            "coordinates", "name_with_title", "date_of_birth",
        }:
            return 0.85
        # Contextual patterns (lower confidence)
        if "contextual" in description.lower() or "context" in description.lower():
            return 0.72
        # Generic patterns (moderate confidence)
        if pii_type in {"phone_generic", "pincode_india", "zipcode_us", "blood_group"}:
            return 0.65
        return 0.8

    def run(self, text: str) -> DetectionResult:
        """Scan text for PII and return detection results with severity.
        
        Args:
            text: Input text to scan for PII
            
        Returns:
            DetectionResult containing matched PII items, severity, and summary
        """
        # Step 0: Input validation
        if not text or not isinstance(text, str):
            return DetectionResult(
                agent_name=self.name,
                threat_found=False,
                threat_type="PII",
                matched=[],
                severity="NONE",
                summary="Input is empty or invalid."
            )

        # Step 0.5: Truncate very long text
        processed_text, was_truncated = self._truncate_text(text, self.max_text_length)
        if was_truncated:
            print(f"[PIIDetector] Input truncated from {len(text)} to {self.max_text_length} chars")

        matched: List[PIIMatch] = []

        for pii_type, (compiled_pattern, description) in self.compiled_patterns.items():
            try:
                found = compiled_pattern.findall(processed_text)
                for value in found:
                    # re.findall returns strings or tuples (for groups)
                    val = value if isinstance(value, str) else value[0]
                    if val.strip():
                        matched.append(PIIMatch(
                            pii_type=pii_type,
                            value=val.strip(),
                            description=description,
                            confidence=self._confidence_for(pii_type, description),
                        ))
            except (re.error, IndexError) as e:
                print(f"[PIIDetector] Error scanning pattern '{pii_type}': {e}")
                continue  # skip malformed patterns gracefully

        # Deduplication: when several patterns match the same value, keep the most
        # specific type (e.g. a 16-digit card beats the generic phone pattern).
        best: Dict[str, PIIMatch] = {}
        for match in matched:
            normalized_value = match.value.strip().lower()
            current = best.get(normalized_value)
            if current is None or self._specificity(match.pii_type) < self._specificity(current.pii_type):
                best[normalized_value] = match
        matched = list(best.values())

        severity = self._calculate_severity(matched)
        threat_found = len(matched) > 0

        summary = (
            f"Detected {len(matched)} PII item(s) across "
            f"{len({m.pii_type for m in matched})} category(ies). "
            f"Severity: {severity}."
            if threat_found else "No PII detected. Input is clean."
        )

        return DetectionResult(
            agent_name=self.name,
            threat_found=threat_found,
            threat_type="PII",
            matched=matched,
            severity=severity,
            summary=summary
        )

    def redact(self, text):
        """Redact detected PII by replacing with typed placeholders.
        
        Args:
            text: Input text containing PII
            
        Returns:
            Text with PII items replaced by redaction labels
        """
        redacted = text
        for pii_type, (compiled_pattern, description) in self.compiled_patterns.items():
            placeholder = self.redaction_labels.get(pii_type, "[PII]")
            redacted = compiled_pattern.sub(placeholder, redacted)
        return redacted

