from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Protocol


@dataclass
class LLMResponse:
    text: str
    raw: Optional[Dict[str, Any]] = None


class LLM(Protocol):
    """
    Model-agnostic LLM interface.
    """
    def generate(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        ...
