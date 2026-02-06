from __future__ import annotations

import os
from typing import Optional, Dict, Any
from google import genai
from llm.base import LLMResponse
from google import genai

class GeminiLLM:
    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None):
        self.model = model
        self.api_key = api_key
        self.client = genai.Client(
            api_key=api_key,
        )

    def generate(self, prompt: str, *, system: str | None = None):
        if system:
            contents = f"SYSTEM:\n{system}\n\nUSER:\n{prompt}"
        else:
            contents = prompt

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
        )

        text = getattr(response, "text", "") or ""
        raw: Dict[str, Any] = {
            "model": self.model,
        }

        return LLMResponse(text=text, raw=raw)

