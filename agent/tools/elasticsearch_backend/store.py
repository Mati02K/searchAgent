from __future__ import annotations

import hashlib
import math
import os
import time
from datetime import datetime, timezone
from typing import Any

from logging_utils import get_logger
from tools.elasticsearch_backend.embeddings import embed_text, embed_texts, embedding_dim

ES_INDEX_NAME = os.getenv("ELASTICSEARCH_INDEX", "search_vectors")
SEMANTIC_CACHE_INDEX = os.getenv("SEMANTIC_CACHE_INDEX", "semantic_query_cache")
ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
VECTOR_NUM_CANDIDATES_FACTOR = max(
    1, int(os.getenv("VECTOR_NUM_CANDIDATES_FACTOR", "4"))
)
SEMANTIC_CACHE_NUM_CANDIDATES_FACTOR = max(
    1, int(os.getenv("SEMANTIC_CACHE_NUM_CANDIDATES_FACTOR", "4"))
)
logger = get_logger(__name__)


def _build_index_mapping() -> dict[str, Any]:
    dims = embedding_dim()
    return {
        "mappings": {
            "properties": {
                "source": {"type": "keyword"},
                "title": {"type": "text"},
                "content": {"type": "text"},
                "url": {"type": "keyword"},
                "domain_tags": {"type": "keyword"},
                "published": {
                    "type": "date",
                    "format": "strict_date_optional_time||epoch_millis",
                },
                "authors": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        }
    }


def _build_semantic_cache_mapping() -> dict[str, Any]:
    """Index mapping for semantic query cache records."""
    dims = embedding_dim()
    return {
        "mappings": {
            "properties": {
                "query_id": {"type": "keyword"},
                "query_text": {"type": "text"},
                "query_embedding": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                },
                "title": {"type": "text"},
                "content": {"type": "text"},
                "content_embedding": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                },
                "url": {"type": "keyword"},
                "source": {"type": "keyword"},
                "created_at": {
                    "type": "date",
                    "format": "strict_date_optional_time||epoch_millis",
                },
                "published": {
                    "type": "date",
                    "format": "strict_date_optional_time||epoch_millis",
                },
                "authors": {"type": "keyword"},
            }
        }
    }


def get_client():
    started_at = time.perf_counter()
    logger.info("Initializing Elasticsearch client. url=%s", ES_URL)
    try:
        from elasticsearch import Elasticsearch
    except Exception as exc:
        logger.error("Failed to import elasticsearch client: %s", exc)
        return None

    try:
        compat_headers = {
            "accept": "application/vnd.elasticsearch+json; compatible-with=8",
            "content-type": "application/vnd.elasticsearch+json; compatible-with=8",
        }
        client = Elasticsearch(ES_URL, request_timeout=10, headers=compat_headers)
        if not client.ping():
            logger.warning("Elasticsearch ping failed. url=%s", ES_URL)
            raise ConnectionError(f"Cannot connect to Elasticsearch at {ES_URL}")
        logger.info(
            "Elasticsearch client ready. elapsed_ms=%.2f",
            (time.perf_counter() - started_at) * 1000,
        )
        return client
    except Exception as exc:
        logger.exception(
            "Failed to initialize Elasticsearch client: %s elapsed_ms=%.2f",
            exc,
            (time.perf_counter() - started_at) * 1000,
        )
        return None


def _has_vector_mapping(client, index_name: str) -> bool:
    try:
        mapping_response = client.indices.get_mapping(index=index_name)
        index_mapping = mapping_response.get(index_name, {}).get("mappings", {})
        properties = index_mapping.get("properties", {})
        embedding_mapping = properties.get("embedding", {})
        return embedding_mapping.get("type") == "dense_vector"
    except Exception as exc:
        logger.warning("Failed to inspect index mapping. index=%s error=%s", index_name, exc)
        return False


def _has_semantic_cache_mapping(client, index_name: str) -> bool:
    try:
        mapping_response = client.indices.get_mapping(index=index_name)
        index_mapping = mapping_response.get(index_name, {}).get("mappings", {})
        properties = index_mapping.get("properties", {})
        query_embedding_mapping = properties.get("query_embedding", {})
        content_embedding_mapping = properties.get("content_embedding", {})
        return (
            query_embedding_mapping.get("type") == "dense_vector"
            and content_embedding_mapping.get("type") == "dense_vector"
        )
    except Exception as exc:
        logger.warning("Failed to inspect semantic cache mapping. index=%s error=%s", index_name, exc)
        return False


def ensure_index(client, index_name: str = ES_INDEX_NAME) -> bool:
    started_at = time.perf_counter()
    try:
        if client.indices.exists(index=index_name):
            if not _has_vector_mapping(client, index_name=index_name):
                logger.error(
                    "Existing index lacks vector mapping. index=%s elapsed_ms=%.2f",
                    index_name,
                    (time.perf_counter() - started_at) * 1000,
                )
                return False
            logger.info(
                "Elasticsearch vector index already exists. index=%s elapsed_ms=%.2f",
                index_name,
                (time.perf_counter() - started_at) * 1000,
            )
            return True

        client.indices.create(index=index_name, body=_build_index_mapping())
        logger.info(
            "Elasticsearch vector index created. index=%s elapsed_ms=%.2f",
            index_name,
            (time.perf_counter() - started_at) * 1000,
        )
        return True
    except Exception as exc:
        logger.exception(
            "Failed to ensure Elasticsearch index '%s': %s elapsed_ms=%.2f",
            index_name,
            exc,
            (time.perf_counter() - started_at) * 1000,
        )
        return False


def ensure_semantic_cache_index(client, index_name: str = SEMANTIC_CACHE_INDEX) -> bool:
    started_at = time.perf_counter()
    try:
        if client.indices.exists(index=index_name):
            if not _has_semantic_cache_mapping(client, index_name=index_name):
                logger.error(
                    "Existing semantic cache index lacks required vector mapping. index=%s elapsed_ms=%.2f",
                    index_name,
                    (time.perf_counter() - started_at) * 1000,
                )
                return False
            logger.info(
                "Semantic cache index already exists. index=%s elapsed_ms=%.2f",
                index_name,
                (time.perf_counter() - started_at) * 1000,
            )
            return True

        client.indices.create(index=index_name, body=_build_semantic_cache_mapping())
        logger.info(
            "Semantic cache index created. index=%s elapsed_ms=%.2f",
            index_name,
            (time.perf_counter() - started_at) * 1000,
        )
        return True
    except Exception as exc:
        logger.exception(
            "Failed to ensure semantic cache index '%s': %s elapsed_ms=%.2f",
            index_name,
            exc,
            (time.perf_counter() - started_at) * 1000,
        )
        return False


def _doc_id(doc: dict[str, Any]) -> str:
    raw = f"{doc.get('source', '')}::{doc.get('url', '')}::{doc.get('title', '')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _query_id(query_text: str) -> str:
    normalized = " ".join((query_text or "").strip().lower().split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _semantic_cache_doc_id(query_id: str, source: str, url: str, title: str) -> str:
    raw = f"{query_id}::{source}::{url}::{title}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _build_embedding_text(payload: dict[str, Any]) -> str:
    return " ".join(
        part for part in [payload.get("title", ""), payload.get("content", "")] if part
    ).strip()


def _cosine_similarity(query_vector: list[float], other_vector: list[float]) -> float:
    if not query_vector or not other_vector or len(query_vector) != len(other_vector):
        return 0.0
    dot = 0.0
    query_norm_sq = 0.0
    other_norm_sq = 0.0
    for query_value, other_value in zip(query_vector, other_vector, strict=False):
        query_float = float(query_value)
        other_float = float(other_value)
        dot += query_float * other_float
        query_norm_sq += query_float * query_float
        other_norm_sq += other_float * other_float
    if query_norm_sq <= 0.0 or other_norm_sq <= 0.0:
        return 0.0
    similarity = dot / (math.sqrt(query_norm_sq) * math.sqrt(other_norm_sq))
    return max(-1.0, min(1.0, similarity))


def index_documents(client, docs: list[dict[str, Any]], index_name: str = ES_INDEX_NAME) -> int:
    started_at = time.perf_counter()
    if not docs:
        logger.info(
            "index_documents called with empty docs list. elapsed_ms=%.2f",
            (time.perf_counter() - started_at) * 1000,
        )
        return 0

    logger.info("Indexing documents into Elasticsearch. count=%d index=%s", len(docs), index_name)
    payloads: list[dict[str, Any]] = []
    texts: list[str] = []
    for doc in docs:
        payload = {
            "source": str(doc.get("source", "")).strip(),
            "title": str(doc.get("title", "")).strip(),
            "content": str(doc.get("content", "")).strip(),
            "url": str(doc.get("url", "")).strip(),
            "domain_tags": doc.get("domain_tags") or [],
            "published": doc.get("published"),
            "authors": doc.get("authors"),
        }
        if not payload["title"] or not payload["url"]:
            continue
        payloads.append(payload)
        texts.append(_build_embedding_text(payload))

    if not payloads:
        logger.info(
            "No valid documents to index after validation. elapsed_ms=%.2f",
            (time.perf_counter() - started_at) * 1000,
        )
        return 0

    try:
        vectors = embed_texts(texts)
    except Exception as exc:
        logger.exception("Embedding generation failed during indexing. error=%s", exc)
        return 0

    if len(vectors) != len(payloads):
        logger.error(
            "Embedding output length mismatch. payloads=%d vectors=%d",
            len(payloads),
            len(vectors),
        )
        return 0

    indexed = 0
    for payload, vector in zip(payloads, vectors, strict=False):
        payload["embedding"] = vector
        try:
            client.index(index=index_name, id=_doc_id(payload), document=payload)
            indexed += 1
        except Exception as exc:
            logger.warning("Failed to index document url=%s error=%s", payload["url"], exc)

    logger.info(
        "Indexed documents complete. indexed=%d attempted=%d elapsed_ms=%.2f",
        indexed,
        len(payloads),
        (time.perf_counter() - started_at) * 1000,
    )
    return indexed


def index_semantic_cache_query(
    client,
    *,
    query_text: str,
    results: list[dict[str, Any]],
    index_name: str = SEMANTIC_CACHE_INDEX,
) -> int:
    """
    Insert one semantic cache record per retrieved result.
    Each record stores both query and content embeddings.
    """
    started_at = time.perf_counter()
    cleaned_query = " ".join((query_text or "").strip().split())
    if not cleaned_query or not results:
        logger.info(
            "index_semantic_cache_query skipped. query_present=%s results=%d elapsed_ms=%.2f",
            bool(cleaned_query),
            len(results),
            (time.perf_counter() - started_at) * 1000,
        )
        return 0

    if not ensure_semantic_cache_index(client, index_name=index_name):
        logger.warning("Semantic cache index unavailable. index=%s", index_name)
        return 0

    try:
        query_vector = embed_text(cleaned_query)
    except Exception as exc:
        logger.exception("Semantic cache query embedding failed. query=%s error=%s", cleaned_query, exc)
        return 0
    if not query_vector:
        logger.warning("Semantic cache query embedding empty. query=%s", cleaned_query)
        return 0

    payloads: list[dict[str, Any]] = []
    content_texts: list[str] = []
    query_id = _query_id(cleaned_query)
    created_at = datetime.now(timezone.utc).isoformat()
    for result in results:
        if not isinstance(result, dict):
            continue
        title = str(result.get("title", "")).strip()
        content = str(result.get("content") or result.get("summary") or "").strip()
        url = str(result.get("url", "")).strip()
        source = str(result.get("source", "")).strip()
        if not url or not source:
            continue
        payload = {
            "query_id": query_id,
            "query_text": cleaned_query,
            "query_embedding": query_vector,
            "title": title,
            "content": content,
            "url": url,
            "source": source,
            "created_at": created_at,
            "published": result.get("published"),
            "authors": result.get("authors"),
        }
        payloads.append(payload)
        content_texts.append(_build_embedding_text({"title": title, "content": content}))

    if not payloads:
        logger.info(
            "No valid semantic cache payloads generated. elapsed_ms=%.2f",
            (time.perf_counter() - started_at) * 1000,
        )
        return 0

    try:
        content_vectors = embed_texts(content_texts)
    except Exception as exc:
        logger.exception("Semantic cache content embeddings failed. error=%s", exc)
        return 0

    if len(content_vectors) != len(payloads):
        logger.error(
            "Semantic cache embedding length mismatch. payloads=%d vectors=%d",
            len(payloads),
            len(content_vectors),
        )
        return 0

    indexed = 0
    for payload, content_vector in zip(payloads, content_vectors, strict=False):
        payload["content_embedding"] = content_vector
        try:
            client.index(
                index=index_name,
                id=_semantic_cache_doc_id(
                    payload["query_id"],
                    payload["source"],
                    payload["url"],
                    payload["title"],
                ),
                document=payload,
            )
            indexed += 1
        except Exception as exc:
            logger.warning(
                "Semantic cache indexing failed. query_id=%s url=%s error=%s",
                payload["query_id"],
                payload["url"],
                exc,
            )

    logger.info(
        "Semantic cache indexing complete. query_id=%s indexed=%d attempted=%d elapsed_ms=%.2f",
        query_id,
        indexed,
        len(payloads),
        (time.perf_counter() - started_at) * 1000,
    )
    return indexed


def search_semantic_cache(
    client,
    *,
    query_text: str,
    size: int = 10,
    index_name: str = SEMANTIC_CACHE_INDEX,
) -> list[dict[str, Any]]:
    """
    Search semantically similar past queries using query_embedding kNN.
    """
    started_at = time.perf_counter()
    cleaned_query = " ".join((query_text or "").strip().split())
    if not cleaned_query:
        logger.info(
            "search_semantic_cache called with empty query. elapsed_ms=%.2f",
            (time.perf_counter() - started_at) * 1000,
        )
        return []

    if not ensure_semantic_cache_index(client, index_name=index_name):
        logger.warning("Semantic cache index unavailable during search. index=%s", index_name)
        return []

    try:
        query_vector = embed_text(cleaned_query)
    except Exception as exc:
        logger.exception("Semantic cache query embedding failed. query=%s error=%s", cleaned_query, exc)
        return []
    if not query_vector:
        logger.warning("Semantic cache query embedding empty. query=%s", cleaned_query)
        return []

    target_size = max(1, min(int(size), 50))
    num_candidates = max(
        target_size,
        min(500, target_size * SEMANTIC_CACHE_NUM_CANDIDATES_FACTOR),
    )
    body = {
        "size": target_size,
        "_source": [
            "query_id",
            "query_text",
            "query_embedding",
            "title",
            "content",
            "url",
            "source",
            "created_at",
            "published",
            "authors",
        ],
        "knn": {
            "field": "query_embedding",
            "query_vector": query_vector,
            "k": target_size,
            "num_candidates": num_candidates,
        },
    }

    logger.info(
        "Searching semantic cache. index=%s query=%s size=%d num_candidates=%d",
        index_name,
        cleaned_query,
        target_size,
        num_candidates,
    )
    try:
        response = client.search(index=index_name, body=body)
    except Exception as exc:
        logger.exception("Semantic cache search failed. query=%s error=%s", cleaned_query, exc)
        return []

    hits = response.get("hits", {}).get("hits", [])
    results: list[dict[str, Any]] = []
    for hit in hits:
        source = hit.get("_source") or {}
        if not isinstance(source, dict):
            continue
        cached_query_vector = source.get("query_embedding")
        similarity = 0.0
        if isinstance(cached_query_vector, list):
            similarity = _cosine_similarity(query_vector, cached_query_vector)
        results.append(
            {
                "query_id": source.get("query_id", ""),
                "query_text": source.get("query_text", ""),
                "title": source.get("title", ""),
                "content": source.get("content", ""),
                "url": source.get("url", ""),
                "source": source.get("source", ""),
                "created_at": source.get("created_at"),
                "published": source.get("published"),
                "authors": source.get("authors"),
                "query_similarity": float(similarity),
                "score": float(hit.get("_score") or 0.0),
            }
        )

    results.sort(key=lambda item: (-float(item.get("query_similarity", 0.0)), str(item.get("query_id", ""))))
    logger.info(
        "Semantic cache search complete. index=%s query=%s hits=%d elapsed_ms=%.2f",
        index_name,
        cleaned_query,
        len(results),
        (time.perf_counter() - started_at) * 1000,
    )
    return results


def get_cached_query_results(
    client,
    *,
    query_id: str,
    size: int = 30,
    index_name: str = SEMANTIC_CACHE_INDEX,
) -> list[dict[str, Any]]:
    """Fetch all cached results for one cached query id."""
    started_at = time.perf_counter()
    cleaned_query_id = str(query_id or "").strip()
    if not cleaned_query_id:
        return []

    body = {
        "size": max(1, min(int(size), 200)),
        "_source": [
            "query_id",
            "query_text",
            "title",
            "content",
            "url",
            "source",
            "created_at",
            "published",
            "authors",
        ],
        "query": {
            "term": {
                "query_id": cleaned_query_id,
            }
        },
    }
    try:
        response = client.search(index=index_name, body=body)
    except Exception as exc:
        logger.exception("Failed to fetch cached query results. query_id=%s error=%s", cleaned_query_id, exc)
        return []

    hits = response.get("hits", {}).get("hits", [])
    results: list[dict[str, Any]] = []
    for hit in hits:
        source = hit.get("_source") or {}
        if not isinstance(source, dict):
            continue
        results.append(
            {
                "query_id": source.get("query_id", ""),
                "query_text": source.get("query_text", ""),
                "title": source.get("title", ""),
                "content": source.get("content", ""),
                "url": source.get("url", ""),
                "source": source.get("source", ""),
                "created_at": source.get("created_at"),
                "published": source.get("published"),
                "authors": source.get("authors"),
                "score": float(hit.get("_score") or 0.0),
            }
        )
    logger.info(
        "Fetched cached query results. query_id=%s results=%d elapsed_ms=%.2f",
        cleaned_query_id,
        len(results),
        (time.perf_counter() - started_at) * 1000,
    )
    return results


def search_documents(
    client,
    query: str,
    *,
    size: int = 100,
    index_name: str = ES_INDEX_NAME,
) -> list[dict[str, Any]]:
    started_at = time.perf_counter()
    cleaned_query = " ".join((query or "").strip().split())
    if not cleaned_query:
        logger.info(
            "search_documents called with empty query. elapsed_ms=%.2f",
            (time.perf_counter() - started_at) * 1000,
        )
        return []

    try:
        query_vector = embed_text(cleaned_query)
    except Exception as exc:
        logger.exception(
            "Query embedding generation failed. query=%s error=%s elapsed_ms=%.2f",
            cleaned_query,
            exc,
            (time.perf_counter() - started_at) * 1000,
        )
        return []

    if not query_vector:
        logger.warning(
            "Query embedding is empty. query=%s elapsed_ms=%.2f",
            cleaned_query,
            (time.perf_counter() - started_at) * 1000,
        )
        return []

    target_size = max(1, min(int(size), 200))
    num_candidates = max(target_size, min(1000, target_size * VECTOR_NUM_CANDIDATES_FACTOR))
    body = {
        "size": target_size,
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": target_size,
            "num_candidates": num_candidates,
        },
    }

    logger.info(
        "Searching Elasticsearch (vector). index=%s query=%s size=%d num_candidates=%d",
        index_name,
        cleaned_query,
        target_size,
        num_candidates,
    )

    try:
        response = client.search(index=index_name, body=body)
    except Exception as exc:
        logger.exception(
            "Elasticsearch vector search failed. query=%s error=%s elapsed_ms=%.2f",
            cleaned_query,
            exc,
            (time.perf_counter() - started_at) * 1000,
        )
        return []

    hits = response.get("hits", {}).get("hits", [])
    results: list[dict[str, Any]] = []
    for hit in hits:
        source = hit.get("_source") or {}
        if not isinstance(source, dict):
            continue
        results.append(
            {
                "source": source.get("source", ""),
                "title": source.get("title", ""),
                "content": source.get("content", ""),
                "url": source.get("url", ""),
                "domain_tags": source.get("domain_tags", []),
                "published": source.get("published"),
                "authors": source.get("authors"),
                "score": float(hit.get("_score") or 0.0),
            }
        )

    logger.info(
        "Elasticsearch vector search complete. index=%s query=%s hits=%d elapsed_ms=%.2f",
        index_name,
        cleaned_query,
        len(results),
        (time.perf_counter() - started_at) * 1000,
    )
    return results
