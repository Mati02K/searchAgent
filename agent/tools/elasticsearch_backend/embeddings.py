from __future__ import annotations

import os
from functools import lru_cache

from logging_utils import get_logger

logger = get_logger(__name__)

VECTOR_EMBED_MODEL = os.getenv(
    "VECTOR_EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
VECTOR_EMBED_BATCH_SIZE = max(1, int(os.getenv("VECTOR_EMBED_BATCH_SIZE", "32")))


@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model. model=%s", VECTOR_EMBED_MODEL)
    model = SentenceTransformer(VECTOR_EMBED_MODEL)
    logger.info("Embedding model ready. model=%s", VECTOR_EMBED_MODEL)
    return model


def embedding_dim() -> int:
    model = _get_embedder()
    return int(model.get_sentence_embedding_dimension())


def embed_text(text: str) -> list[float]:
    vectors = embed_texts([text])
    return vectors[0] if vectors else []


def embed_texts(texts: list[str]) -> list[list[float]]:
    cleaned = [" ".join((text or "").strip().split()) for text in texts]
    if not cleaned:
        return []

    model = _get_embedder()
    vectors = model.encode(
        cleaned,
        batch_size=VECTOR_EMBED_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]
