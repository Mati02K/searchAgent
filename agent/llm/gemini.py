from __future__ import annotations

import os
from typing import Any, Dict

from google import genai

from llm.base import LLMResponse
from logging_utils import get_logger

logger = get_logger(__name__)


class GeminiLLM:
    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None):
        self.model = model
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is not set.")

        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        logger.info("GeminiLLM initialized. model=%s", self.model)

    def generate(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        try:
            logger.info(
                "Gemini generate start. model=%s prompt_len=%d has_system=%s",
                self.model,
                len(prompt or ""),
                bool(system),
            )
            if system:
                contents = f"SYSTEM:\n{system}\n\nUSER:\n{prompt}"
            else:
                contents = prompt
            logger.info("Gemini generate payload:\n%s", contents)

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
            )

            text = getattr(response, "text", "") or ""
            raw: Dict[str, Any] = {
                "model": self.model,
            }
            logger.info("Gemini generate success. response_text_len=%d", len(text))
            return LLMResponse(text=text, raw=raw)
        except Exception as exc:
            logger.exception("Gemini generate failed: %s", exc)
            return LLMResponse(text="", raw={"error": str(exc)})
