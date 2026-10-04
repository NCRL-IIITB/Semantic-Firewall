from semantic_firewall.core.agents.llm_client import DEFAULT_LLM_MODEL, DetectorLLMClient, extract_json_object


def test_extract_json_object_handles_thinking_prefix():
    payload, meta = extract_json_object(
        "Thinking...\nThis is analysis text.\n{\"is_injection\": true, \"confidence\": 0.95}"
    )

    assert meta["llm_parse_status"] == "ok"
    assert payload == {"is_injection": True, "confidence": 0.95}


def test_openrouter_provider_reports_missing_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.delenv("SEMANTIC_FIREWALL_LLM_MODEL", raising=False)

    client = DetectorLLMClient()

    assert client.provider == "openrouter"
    assert client.model_name == DEFAULT_LLM_MODEL
    assert client.availability_error() == "missing_api_key"


def _fake_client(captured: dict):
    class FakeResponse:
        class Choice:
            class Message:
                content = '{"is_injection": true, "confidence": 0.88}'

            message = Message()

        choices = [Choice()]

    class FakeChatCompletions:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return FakeResponse()

    class FakeChat:
        completions = FakeChatCompletions()

    class FakeOpenAIClient:
        chat = FakeChat()

    return FakeOpenAIClient()


def test_openrouter_client_uses_chat_completion(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("SEMANTIC_FIREWALL_LLM_MODEL", raising=False)
    monkeypatch.delenv("SEMANTIC_FIREWALL_LLM_JSON_MODE", raising=False)

    captured = {}
    client = DetectorLLMClient()
    client._client = _fake_client(captured)
    response = client.complete("system prompt", "user prompt", max_tokens=33)

    assert captured["model"] == DEFAULT_LLM_MODEL
    assert captured["temperature"] == 0.0
    assert captured["messages"][0]["content"] == "system prompt"
    assert captured["messages"][1]["content"] == "user prompt"
    assert captured["max_tokens"] == 33
    assert "response_format" not in captured
    assert response.meta["llm_provider"] == "openrouter"
    assert response.meta["llm_model"] == DEFAULT_LLM_MODEL
    assert response.content == '{"is_injection": true, "confidence": 0.88}'


def test_json_mode_requests_json_object(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("SEMANTIC_FIREWALL_LLM_JSON_MODE", "1")

    captured = {}
    client = DetectorLLMClient()
    client._client = _fake_client(captured)
    client.complete("system prompt", "user prompt")

    assert captured["response_format"] == {"type": "json_object"}
