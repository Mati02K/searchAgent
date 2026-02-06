from __future__ import annotations

import contextlib
import io
import os

from safety.base import SafetyClassifier, SafetyResult


class TransformersSafetyClassifier(SafetyClassifier):
    """
    Optional model-based safety classifier.

    Blocks when any configured toxicity label score is above threshold.
    """

    def __init__(
        self,
        model_name: str = "unitary/toxic-bert",
        threshold: float = 0.75,
    ):
        self.model_name = model_name
        self.threshold = threshold
        use_gpu = os.environ.get("SAFETY_USE_GPU", "false").strip().lower()
        self.device = 0 if use_gpu in {"1", "true", "yes", "on"} else -1
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        block_labels = os.environ.get(
            "SAFETY_BLOCK_LABELS",
            "toxic,severe_toxic,insult,threat,identity_hate,obscene,sexual_explicit",
        )
        self.block_labels = {
            label.strip().lower().replace(" ", "_")
            for label in block_labels.split(",")
            if label.strip()
        }

        from transformers import pipeline
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self._pipeline = pipeline(
                "text-classification",
                model=model_name,
                device=self.device,
            )

    def evaluate(self, prompt: str) -> SafetyResult:
        raw = self._pipeline(prompt, truncation=True, top_k=None)

        candidates: list[dict] = []
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            candidates = [item for item in raw[0] if isinstance(item, dict)]
        elif isinstance(raw, list):
            candidates = [item for item in raw if isinstance(item, dict)]
        elif isinstance(raw, dict):
            candidates = [raw]

        label_scores: dict[str, float] = {}
        for item in candidates:
            label = str(item.get("label", "")).strip().lower().replace(" ", "_")
            score = float(item.get("score", 0.0))
            if label:
                label_scores[label] = max(score, label_scores.get(label, 0.0))

        flagged = [
            (label, score)
            for label, score in label_scores.items()
            if label in self.block_labels and score >= self.threshold
        ]

        if flagged:
            details = ", ".join(f"{label}={score:.2f}" for label, score in sorted(flagged))
            return SafetyResult(
                allowed=False,
                reason=f"Prompt blocked by model classifier: {details}.",
                matched_terms=[],
                classifier=f"transformers:{self.model_name}",
            )

        if label_scores:
            top_label, top_score = max(label_scores.items(), key=lambda item: item[1])
            return SafetyResult(
                allowed=True,
                reason=f"Prompt passed model classifier: {top_label} ({top_score:.2f}).",
                matched_terms=[],
                classifier=f"transformers:{self.model_name}",
            )

        return SafetyResult(
            allowed=True,
            reason="Prompt passed model classifier: no label scores returned.",
            matched_terms=[],
            classifier=f"transformers:{self.model_name}",
        )


def build_optional_model_classifier() -> SafetyClassifier | None:
    provider = os.getenv("SAFETY_MODEL_PROVIDER", "transformers").lower()
    if provider in {"none", "disabled", "off"}:
        return None
    if provider != "transformers":
        return None

    model_name = os.getenv("SAFETY_MODEL_NAME", "unitary/toxic-bert")
    threshold = float(os.getenv("SAFETY_MODEL_THRESHOLD", "0.75"))
    try:
        return TransformersSafetyClassifier(model_name=model_name, threshold=threshold)
    except Exception:
        return None
