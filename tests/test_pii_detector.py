import pytest
from semantic_firewall.core.agents.pii_detector import PIIDetectorAgent

@pytest.fixture
def agent():
    return PIIDetectorAgent()


# â”€â”€ True Positive Tests (should detect) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestPIITruePositives:

    def test_detects_email(self, agent):
        result = agent.run("Contact me at john.doe@gmail.com")
        assert result.threat_found is True
        types = [m.pii_type for m in result.matched]
        assert "email" in types

    def test_detects_aadhaar(self, agent):
        result = agent.run("My Aadhaar is 1234 5678 9012")
        assert result.threat_found is True
        types = [m.pii_type for m in result.matched]
        assert "aadhaar" in types

    def test_detects_pan_card(self, agent):
        result = agent.run("PAN number: ABCDE1234F")
        assert result.threat_found is True
        types = [m.pii_type for m in result.matched]
        assert "pan_card" in types

    def test_detects_indian_phone(self, agent):
        result = agent.run("Call me at +91-9876543210")
        assert result.threat_found is True
        types = [m.pii_type for m in result.matched]
        assert "phone_india" in types

    def test_detects_credit_card(self, agent):
        result = agent.run("Card number: 4111 1111 1111 1111")
        assert result.threat_found is True
        types = [m.pii_type for m in result.matched]
        assert "credit_card_visa" in types or "credit_card_generic" in types

    def test_detects_passport(self, agent):
        result = agent.run("Passport no: J8369854")
        assert result.threat_found is True

    def test_detects_ipv4(self, agent):
        result = agent.run("Server IP is 192.168.1.100")
        assert result.threat_found is True
        types = [m.pii_type for m in result.matched]
        assert "ipv4" in types

    def test_detects_ifsc(self, agent):
        result = agent.run("IFSC code: SBIN0001234")
        assert result.threat_found is True
        types = [m.pii_type for m in result.matched]
        assert "ifsc_code" in types

    def test_detects_vehicle_registration(self, agent):
        result = agent.run("My car number is MH12AB1234")
        assert result.threat_found is True
        types = [m.pii_type for m in result.matched]
        assert "vehicle_reg_india" in types

    def test_detects_blood_group(self, agent):
        result = agent.run("Patient blood group is A+")
        assert result.threat_found is True

    def test_detects_mac_address(self, agent):
        result = agent.run("MAC: 00:1A:2B:3C:4D:5E")
        assert result.threat_found is True
        types = [m.pii_type for m in result.matched]
        assert "mac_address" in types

    def test_detects_multiple_pii(self, agent):
        result = agent.run("Email: a@b.com, Aadhaar: 1234 5678 9012, PAN: ABCDE1234F")
        assert result.threat_found is True
        assert len(result.matched) >= 3

    def test_detects_upi_id(self, agent):
        result = agent.run("Pay me at john@okaxis")
        assert result.threat_found is True


# â”€â”€ True Negative Tests (should NOT detect) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestPIITrueNegatives:

    def test_clean_question(self, agent):
        result = agent.run("What is the capital of France?")
        assert result.threat_found is False
        assert result.severity == "NONE"

    def test_clean_greeting(self, agent):
        result = agent.run("Hello, how are you today?")
        assert result.threat_found is False

    def test_clean_code_snippet(self, agent):
        result = agent.run("def add(a, b): return a + b")
        assert result.threat_found is False

    def test_clean_general_text(self, agent):
        result = agent.run("The quick brown fox jumps over the lazy dog.")
        assert result.threat_found is False


# â”€â”€ Severity Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestPIISeverity:

    def test_aadhaar_is_critical(self, agent):
        result = agent.run("Aadhaar: 1234 5678 9012")
        assert result.severity == "CRITICAL"

    def test_credit_card_is_critical(self, agent):
        result = agent.run("Card: 4111 1111 1111 1111")
        assert result.severity == "CRITICAL"

    def test_single_email_is_low_or_medium(self, agent):
        result = agent.run("email: alice.sharma@company.org")
        assert result.severity in ["LOW", "MEDIUM"]

    def test_clean_input_is_none(self, agent):
        result = agent.run("Hello world")
        assert result.severity == "NONE"


# â”€â”€ Redact Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestPIIRedact:

    def test_redacts_email(self, agent):
        redacted = agent.redact("Email: john@gmail.com")
        assert "john@gmail.com" not in redacted
        assert "[EMAIL]" in redacted

    def test_redacts_aadhaar(self, agent):
        redacted = agent.redact("Aadhaar: 1234 5678 9012")
        assert "1234 5678 9012" not in redacted
        assert "[AADHAAR]" in redacted

    def test_clean_text_unchanged(self, agent):
        text = "Hello, how are you?"
        redacted = agent.redact(text)
        assert "[EMAIL]" not in redacted


class TestPIIConfidence:

    def test_email_confidence_is_present(self, agent):
        result = agent.run("Contact me at john.doe@gmail.com")
        email_match = next(match for match in result.matched if match.pii_type == "email")
        assert 0.9 <= email_match.confidence <= 1.0


# â”€â”€ Edge Case Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestPIIEdgeCases:

    def test_empty_string(self, agent):
        result = agent.run("")
        assert result.threat_found is False

    def test_very_long_clean_text(self, agent):
        result = agent.run("Hello world " * 500)
        assert result.threat_found is False

    def test_mixed_case_email(self, agent):
        result = agent.run("Email: John.DOE@Gmail.COM")
        assert result.threat_found is True

    def test_email_in_sentence(self, agent):
        result = agent.run("Please send the report to alice@company.org by Friday.")
        assert result.threat_found is True

