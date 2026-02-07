from __future__ import annotations

from functools import lru_cache

from nodes.safety.base import SafetyClassifier, SafetyResult
from nodes.safety.classifiers import build_optional_model_classifier


class SafetyAgentService:
    """Policy gate for prompts before LLM access."""

    def __init__(
        self,
        model_classifier: SafetyClassifier | None = None,
    ):
        self._model_classifier = model_classifier
        self._model_init_attempted = model_classifier is not None

    def _get_model_classifier(self) -> SafetyClassifier | None:
        if self._model_init_attempted:
            return self._model_classifier

        self._model_init_attempted = True
        try:
            self._model_classifier = build_optional_model_classifier()
        except Exception:
            self._model_classifier = None
        return self._model_classifier

    def evaluate(self, prompt: str) -> SafetyResult:
        try:
            model_classifier = self._get_model_classifier()
            if model_classifier is not None:
                model_result = model_classifier.evaluate(prompt)
                return model_result

            return SafetyResult(
                allowed=True,
                reason="Prompt passed safety checks.",
                matched_terms=[],
                classifier="safety-agent",
            )
        except Exception as exc:
            return SafetyResult(
                allowed=False,
                reason=f"Safety service failure: {exc}",
                matched_terms=[],
                classifier="safety-agent",
            )


@lru_cache(maxsize=1)
def get_safety_agent_service() -> SafetyAgentService:
    return SafetyAgentService()
