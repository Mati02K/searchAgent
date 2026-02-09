from __future__ import annotations

from functools import lru_cache

from nodes.safety.base import SafetyClassifier, SafetyResult
from nodes.safety.classifiers import get_model_classifier


class SafetyAgentService:
    """Strict safety gate. Model classifier is mandatory and singleton-backed."""

    def __init__(self, model_classifier: SafetyClassifier | None = None):
        self._model_classifier = model_classifier

    def _get_model_classifier(self) -> SafetyClassifier:
        if self._model_classifier is None:
            self._model_classifier = get_model_classifier()
        return self._model_classifier

    def evaluate(self, prompt: str) -> SafetyResult:
        model_classifier = self._get_model_classifier()
        return model_classifier.evaluate(prompt)


@lru_cache(maxsize=1)
def get_safety_agent_service() -> SafetyAgentService:
    return SafetyAgentService()
