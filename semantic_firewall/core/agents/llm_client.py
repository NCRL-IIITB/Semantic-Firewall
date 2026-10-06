import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


# OpenRouter model ID of the LLM gate (Llama-3.3-70B-Instruct).
DEFAULT_LLM_MODEL = "meta-llama/llama-3.3-70b-instruct"


@dataclass
class LLMResponse:
    content: str
    meta: dict[str, Any]


class DetectorLLMClient:
    """Small provider wrapper for semantic detector LLM calls (OpenRouter, OpenAI-compatible)."""

    def __init__(self, default_model: str = DEFAULT_LLM_MODEL, model: str | None = None,
                 providers: list[str] | None = None):
        # `model` / `providers` override the environment (used for a secondary model with its own hosts).
        self.model = model or os.getenv("SEMANTIC_FIREWALL_LLM_MODEL", default_model).strip() or default_model
        self.provider = "openrouter"
        # JSON mode asks the provider to constrain decoding to a JSON object; off by
        # default because not every OpenRouter backend supports response_format.
        self.json_mode = os.getenv("SEMANTIC_FIREWALL_LLM_JSON_MODE", "0").strip().lower() in {"1", "true", "yes"}
        # Optional OpenRouter host pinning (comma-separated provider names, no fallback), so every call is
        # served by the same backend; hosts of the same model differ in quantisation and output quality.
        self.providers = providers if providers is not None else [
            p.strip() for p in os.getenv("SEMANTIC_FIREWALL_OPENROUTER_PROVIDERS", "").split(",") if p.strip()]
        timeout = float(os.getenv("SEMANTIC_FIREWALL_LLM_TIMEOUT_SEC", "30"))
        self.max_attempts = max(1, int(os.getenv("SEMANTIC_FIREWALL_LLM_MAX_ATTEMPTS", "4")))
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=timeout,
        ) if api_key else None

    @property
    def model_name(self) -> str:
        return self.model

    def availability_error(self) -> str | None:
        return None if self._client is not None else "missing_api_key"

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        return self._complete_openai(system_prompt, user_prompt, max_tokens)

    def _complete_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> LLMResponse:
        if self._client is None:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")

        extra = {"response_format": {"type": "json_object"}} if self.json_mode else {}
        if self.providers:
            # only the listed hosts, tried in order (fallback stays inside the list)
            extra["extra_body"] = {"provider": {"order": self.providers, "allow_fallbacks": False}}

        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens,
                    **extra,
                )
            except Exception as exc:  # rate limits (429) and transient server errors are retried
                status = getattr(exc, "status_code", None)
                if status is not None and status not in (408, 429, 500, 502, 503, 504):
                    raise
                last_error = exc
                time.sleep(min(2 ** attempt, 8) + random.random())
                continue
            content = (response.choices[0].message.content or "").strip()
            if content and not any(ch.isalnum() for ch in content):
                # degenerate output from a faulty host (e.g. "!!!!!!!!"): retry instead of parsing it
                last_error = RuntimeError(f"degenerate LLM output: {content[:20]!r}")
                time.sleep(min(2 ** attempt, 8) + random.random())
                continue
            host = (getattr(response, "model_extra", None) or {}).get("provider")
            return LLMResponse(
                content=content,
                meta={"llm_provider": self.provider, "llm_model": self.model, "llm_json_mode": self.json_mode,
                      "llm_host": host, "llm_attempts": attempt + 1},
            )
        raise RuntimeError(f"LLM call failed after {self.max_attempts} attempts: {last_error}")


def extract_json_object(raw: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as direct_error:
        payload = _extract_balanced_object(cleaned)
        if payload is None:
            return None, {
                "llm_parse_status": "invalid_json",
                "llm_parse_error": str(direct_error),
                "llm_raw_preview": raw[:120],
            }
    else:
        if not isinstance(payload, dict):
            return None, {
                "llm_parse_status": "invalid_shape",
                "llm_parse_error": f"Expected top-level object, got {type(payload).__name__}",
            }

    return payload, {"llm_parse_status": "ok"}


def _extract_balanced_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return payload if isinstance(payload, dict) else None

    return None
