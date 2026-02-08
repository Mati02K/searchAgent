from __future__ import annotations

import hashlib
import os
import time
from typing import Any

from logging_utils import get_logger

ES_INDEX_NAME = os.getenv("ELASTICSEARCH_INDEX", "search_documents")
ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
logger = get_logger(__name__)

INDEX_MAPPING: dict[str, Any] = {
    "mappings": {
        "properties": {
            "source": {"type": "keyword"},
            "title": {"type": "text"},
            "content": {"type": "text"},
            "url": {"type": "keyword"},
            "domain_tags": {"type": "keyword"},
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
        # Keep client/server wire compatibility on ES 8.x even if an ES 9 python client
        # is installed in the environment.
        compat_headers = {
            "accept": "application/vnd.elasticsearch+json; compatible-with=8",
            "content-type": "application/vnd.elasticsearch+json; compatible-with=8",
        }
        client = Elasticsearch(ES_URL, request_timeout=10, headers=compat_headers)
        if not client.ping():
            logger.warning("Elasticsearch ping failed. url=%s", ES_URL)
            raise ConnectionError(f"Cannot connect to Elasticsearch at {ES_URL}")
        logger.info("Elasticsearch client ready. elapsed_ms=%.2f", (time.perf_counter() - started_at) * 1000)
        return client
    except Exception as exc:
        logger.exception(
            "Failed to initialize Elasticsearch client: %s elapsed_ms=%.2f",
            exc,
            (time.perf_counter() - started_at) * 1000,
        )
        return None


def ensure_index(client, index_name: str = ES_INDEX_NAME) -> bool:
    started_at = time.perf_counter()
    try:
        if client.indices.exists(index=index_name):
            logger.info(
                "Elasticsearch index already exists. index=%s elapsed_ms=%.2f",
                index_name,
                (time.perf_counter() - started_at) * 1000,
            )
            return True
        client.indices.create(index=index_name, body=INDEX_MAPPING)
        logger.info(
            "Elasticsearch index created. index=%s elapsed_ms=%.2f",
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
    raw = f"{doc.get('source','')}::{doc.get('url','')}::{doc.get('title','')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def index_documents(client, docs: list[dict[str, Any]], index_name: str = ES_INDEX_NAME) -> int:
    started_at = time.perf_counter()
    if not docs:
        logger.info(
            "index_documents called with empty docs list. elapsed_ms=%.2f",
            (time.perf_counter() - started_at) * 1000,
        )
        return 0
    logger.info("Indexing documents into Elasticsearch. count=%d index=%s", len(docs), index_name)
    indexed = 0
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
        try:
            client.index(index=index_name, id=_doc_id(payload), document=payload)
            indexed += 1
        except Exception as exc:
            logger.warning("Failed to index document url=%s error=%s", payload["url"], exc)
            continue
    logger.info(
        "Indexed documents complete. indexed=%d attempted=%d elapsed_ms=%.2f",
        indexed,
        len(docs),
        (time.perf_counter() - started_at) * 1000,
    )
    return indexed


def search_documents(
    client,
    query: str,
    *,
    size: int = 100,
    index_name: str = ES_INDEX_NAME,
    must_include: list[str] | None = None,
    must_exclude: list[str] | None = None,
) -> list[dict[str, Any]]:
    started_at = time.perf_counter()
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        logger.info(
            "search_documents called with empty query. elapsed_ms=%.2f",
            (time.perf_counter() - started_at) * 1000,
        )
        return []

    target_size = max(1, min(int(size), 200))
    must: list[dict[str, Any]] = [
        {
            "multi_match": {
                "query": cleaned_query,
                "fields": ["title^3", "content^2", "domain_tags^2"],
            }
        }
    ]
    should: list[dict[str, Any]] = []
    must_not: list[dict[str, Any]] = []

    include_terms = [str(term or "").strip() for term in (must_include or []) if str(term or "").strip()]
    for term in include_terms:
        normalized = (term or "").strip()
        if not normalized:
            continue
        # keep must_include soft at retrieval stage; hard enforcement is done in reranker node
        should.append(
            {
                "multi_match": {
                    "query": normalized,
                    "fields": ["title^2", "content", "domain_tags^2"],
                }
            }
        )

    for term in must_exclude or []:
        normalized = (term or "").strip()
        if not normalized:
            continue
        must_not.append(
            {
                "multi_match": {
                    "query": normalized,
                    "fields": ["title^2", "content", "domain_tags^2"],
                }
            }
        )

    bool_query: dict[str, Any] = {
        "must": must,
        "must_not": must_not,
    }
    if should:
        bool_query["should"] = should
        bool_query["minimum_should_match"] = 1

    body = {
        "size": target_size,
        "query": {
            "bool": bool_query
        },
    }
    logger.info(
        "Searching Elasticsearch. index=%s query=%s size=%d must_include=%d must_exclude=%d",
        index_name,
        cleaned_query,
        target_size,
        len(include_terms),
        len(must_exclude or []),
    )

    try:
        response = client.search(index=index_name, body=body)
    except Exception as exc:
        logger.exception(
            "Elasticsearch search failed. query=%s error=%s elapsed_ms=%.2f",
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
        "Elasticsearch search complete. index=%s query=%s hits=%d elapsed_ms=%.2f",
        index_name,
        cleaned_query,
        len(results),
        (time.perf_counter() - started_at) * 1000,
    )
    return results
