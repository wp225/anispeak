from __future__ import annotations

import json
import logging
from typing import Any, Type

from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, APIError
from pydantic import ValidationError

from anispeakarchive.services.ask_ai_schema import AskAIModelOutput


LOGGER = logging.getLogger(__name__)


class AskAIUnavailableError(RuntimeError):
    pass


class AskAIResponseError(RuntimeError):
    pass


class AskAIClient:
    def __init__(self, ask_ai_cfg: Any) -> None:
        self._cfg = ask_ai_cfg
        self._client = OpenAI(
            base_url=str(ask_ai_cfg.base_url),
            api_key=str(ask_ai_cfg.api_key),
            timeout=120.0,
        )

    def _extract_content(self, response: Any) -> str:
        try:
            content = response.choices[0].message.content
        except Exception as exc:
            raise AskAIResponseError("LLM response missing message content") from exc
        if isinstance(content, list):
            text = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        else:
            text = str(content or "")
        text = text.strip()
        if not text:
            raise AskAIResponseError("LLM returned empty content")
        return text

    def _parse_output(self, raw_text: str) -> AskAIModelOutput:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json\n", "", 1).strip()
        return AskAIModelOutput.model_validate_json(cleaned)

    def _parse_model_output(self, schema_cls: Type[Any], raw_text: str) -> Any:
        cleaned = str(raw_text or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json\n", "", 1).strip()
        return schema_cls.model_validate_json(cleaned)

    def _create_completion(
        self,
        messages: list[dict[str, str]],
        force_json_fallback: bool = False,
        *,
        schema_cls: Type[Any] = AskAIModelOutput,
        max_tokens: int | None = None,
        temperature: float | None = None,
        structured_output: bool | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": str(self._cfg.model_name),
            "messages": messages,
            "temperature": float(self._cfg.temperature if temperature is None else temperature),
            "top_p": float(self._cfg.top_p),
            "max_tokens": int(self._cfg.max_new_tokens if max_tokens is None else max_tokens),
            "stream": False,
        }
        provider_name = str(getattr(self._cfg, "provider", "") or "").strip().lower()
        guided_json_supported = provider_name not in {"vllm_openai", "vllm"}
        use_structured_output = bool(self._cfg.structured_output if structured_output is None else structured_output)
        if use_structured_output and not force_json_fallback:
            if guided_json_supported:
                kwargs["extra_body"] = {
                    "guided_json": schema_cls.model_json_schema(),
                }
            else:
                LOGGER.info("Ask AI structured output falling back to prompt-only JSON for provider=%s", provider_name or "unknown")
        try:
            return self._client.chat.completions.create(**kwargs)
        except (APIConnectionError, APITimeoutError) as exc:
            raise AskAIUnavailableError("Ask AI model server is not available.") from exc
        except APIError as exc:
            raise AskAIResponseError(f"LLM API error: {exc}") from exc

    def _create_stream_completion(self, messages: list[dict[str, str]]) -> Any:
        kwargs: dict[str, Any] = {
            "model": str(self._cfg.model_name),
            "messages": messages,
            "temperature": float(self._cfg.temperature),
            "top_p": float(self._cfg.top_p),
            "max_tokens": int(self._cfg.max_new_tokens),
            "stream": True,
        }
        try:
            return self._client.chat.completions.create(**kwargs)
        except (APIConnectionError, APITimeoutError) as exc:
            raise AskAIUnavailableError("Ask AI model server is not available.") from exc
        except APIError as exc:
            raise AskAIResponseError(f"LLM API error: {exc}") from exc

    def generate(self, system_prompt: str, user_prompt: str) -> AskAIModelOutput:
        return self.generate_structured(AskAIModelOutput, system_prompt, user_prompt)

    def generate_structured(
        self,
        schema_cls: Type[Any],
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        structured_output: bool | None = None,
    ) -> Any:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for attempt in range(2):
            force_json_fallback = attempt > 0
            if force_json_fallback:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            user_prompt
                            + "\n\nReturn strict JSON only. No markdown fences. "
                            "Your response must validate against the required schema."
                        ),
                    },
                ]
            response = self._create_completion(
                messages,
                force_json_fallback=force_json_fallback,
                schema_cls=schema_cls,
                max_tokens=max_tokens,
                temperature=temperature,
                structured_output=structured_output,
            )
            raw_text = self._extract_content(response)
            try:
                return self._parse_model_output(schema_cls, raw_text)
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                LOGGER.warning("Ask AI output parse failed attempt=%s error=%s", attempt + 1, type(exc).__name__)
                if attempt == 1:
                    raise AskAIResponseError("Ask AI returned invalid structured output.") from exc
        raise AskAIResponseError("Ask AI returned invalid structured output.")

    def iter_stream_answer(self, system_prompt: str, user_prompt: str):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        stream = self._create_stream_completion(messages)
        try:
            for event in stream:
                try:
                    delta = event.choices[0].delta
                    content = getattr(delta, "content", None)
                except Exception:
                    content = None
                if not content:
                    continue
                yield str(content)
        except (APIConnectionError, APITimeoutError) as exc:
            raise AskAIUnavailableError("Ask AI model server is not available.") from exc
        except APIError as exc:
            raise AskAIResponseError(f"LLM API error: {exc}") from exc
