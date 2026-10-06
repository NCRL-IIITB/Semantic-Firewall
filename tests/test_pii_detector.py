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



# -- Formats added for the camera-ready (phone separators, written dates, overlap) --

def _values(result):
    return {m.value for m in result.matched}


class TestPIIFormats:

    @pytest.mark.parametrize("text,value", [
        ("Contact +60 219.924 4892 for details", "+60 219.924 4892"),
        ("Reach us at +330-186-841-1608.", "+330-186-841-1608"),
        ("Contact: (352) 9324865. Sessions start soon.", "(352) 9324865"),
        ("For assistance call 05843-76140.", "05843-76140"),
    ])
    def test_phone_is_matched_whole(self, agent, text, value):
        assert value in _values(agent.run(text))

    def test_phone_is_not_split_into_fragments(self, agent):
        values = _values(agent.run("Contact +60 219.924 4892 for details"))
        assert values == {"+60 219.924 4892"}

    @pytest.mark.parametrize("text,value", [
        ("I was born on August 25, 1918 and live here.", "August 25, 1918"),
        ("Date of birth: 4th April 1990.", "4th April 1990"),
        ("DOB 1952-09-28T21:31:54.180Z on file.", "1952-09-28T21:31:54.180Z"),
    ])
    def test_written_and_iso_dates(self, agent, text, value):
        assert value in _values(agent.run(text))

    def test_account_number_value_excludes_keyword(self, agent):
        values = _values(agent.run("Please credit account: 46281517 today."))
        assert "46281517" in values
        assert not any("account" in v.lower() for v in values)

    def test_account_keyword_without_number_is_not_pii(self, agent):
        assert agent.run("Your account information is required.").threat_found is False

    def test_bare_six_digit_number_is_not_a_pincode(self, agent):
        result = agent.run("The invoice total is 305139 rupees.")
        assert "pincode_india" not in {m.pii_type for m in result.matched}

    def test_pincode_with_context_is_detected(self, agent):
        result = agent.run("Delivery address PIN code: 560001, Bengaluru.")
        assert "560001" in _values(result)


class TestPIIDatasetProfile:

    def test_bare_account_number_needs_profile(self, monkeypatch):
        text = "Please find 45224502 as a reference."
        monkeypatch.delenv("SEMANTIC_FIREWALL_PII_PROFILE", raising=False)
        assert "45224502" not in _values(PIIDetectorAgent().run(text))
        monkeypatch.setenv("SEMANTIC_FIREWALL_PII_PROFILE", "ai4privacy")
        assert "45224502" in _values(PIIDetectorAgent().run(text))

    def test_swiss_style_ssn_in_profile(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_FIREWALL_PII_PROFILE", "ai4privacy")
        assert "75635653663" in _values(PIIDetectorAgent().run("Your SSN 75635653663 is on file."))
