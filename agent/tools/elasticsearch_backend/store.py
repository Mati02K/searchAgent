from __future__ import annotations

import hashlib
import os
import time
from typing import Any

from logging_utils import get_logger
from tools.elasticsearch_backend.embeddings import embed_text, embed_texts, embedding_dim

ES_INDEX_NAME = os.getenv("ELASTICSEARCH_INDEX", "search_vectors")
DEFAULT_ES_URL = "http://localhost:9200"
VECTOR_NUM_CANDIDATES_FACTOR = max(
    1, int(os.getenv("VECTOR_NUM_CANDIDATES_FACTOR", "4"))
)
logger = get_logger(__name__)


def _resolve_elasticsearch_url() -> str:
    configured_url = os.getenv("ELASTICSEARCH_URL", DEFAULT_ES_URL).strip() or DEFAULT_ES_URL
    running_in_docker = os.path.exists("/.dockerenv")
    localhost_hosts = ("localhost", "127.0.0.1", "0.0.0.0")
    if running_in_docker and any(host in configured_url for host in localhost_hosts):
        return "http://elasticsearch:9200"
    return configured_url


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


def get_client():
    started_at = time.perf_counter()
    es_url = _resolve_elasticsearch_url()
    logger.info("Initializing Elasticsearch client. url=%s", es_url)
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
        client = Elasticsearch(es_url, request_timeout=10, headers=compat_headers)
        if not client.ping():
            logger.warning("Elasticsearch ping failed. url=%s", es_url)
            raise ConnectionError(f"Cannot connect to Elasticsearch at {es_url}")
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


def ensure_index(client, index_name: str = ES_INDEX_NAME) -> bool:
    """
    Making sure index is present and has the correct mapping. If index exists but lacks vector mapping, logs an error and returns False.
    right now we have only one index, but in the future, I wanna add category based indexes.
    """
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


def _doc_id(doc: dict[str, Any]) -> str:
    raw = f"{doc.get('source', '')}::{doc.get('url', '')}::{doc.get('title', '')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _build_embedding_text(payload: dict[str, Any]) -> str:
    return " ".join(
        part for part in [payload.get("title", ""), payload.get("content", "")] if part
    ).strip()


def index_documents(client, docs: list[dict[str, Any]], index_name: str = ES_INDEX_NAME) -> int:
    # write to elasticsearch
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
