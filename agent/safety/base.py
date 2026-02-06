from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class SafetyResult:
    allowed: bool
    reason: str
    matched_terms: list[str] = field(default_factory=list)
    classifier: str = "unknown"


class SafetyClassifier(Protocol):
    """Classifier contract for prompt-level safety checks."""

    def evaluate(self, prompt: str) -> SafetyResult:
        ...
