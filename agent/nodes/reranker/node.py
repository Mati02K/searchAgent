from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import os
import re
import time

from logging_utils import get_logger
from nodes.state import AgentState
from tools.arxiv.tool import search_arxiv
from tools.elasticsearch_backend import (
    ensure_index,
    get_client,
    index_documents,
    search_documents,
)
from tools.wikipedia.tool import search_wikipedia

DEFAULT_CANDIDATE_LIMIT = 80
DEFAULT_TOP_K = 8
DEFAULT_VECTOR_MIN_SIMILARITY = 0.45
DEFAULT_MIN_SOURCES_FOR_LLM = 6
ES_INDEX_PREFIX = os.getenv(
    "ELASTICSEARCH_INDEX_PREFIX",
    os.getenv("ELASTICSEARCH_INDEX", "search_vectors"),
)
BACKFILL_MAX_QUERIES = max(1, min(8, int(os.getenv("BACKFILL_MAX_QUERIES", "4"))))
WIKIPEDIA_BACKFILL_WORKERS = max(
    1, min(8, int(os.getenv("WIKIPEDIA_BACKFILL_WORKERS", "3")))
)
ARXIV_BACKFILL_WORKERS = max(1, min(4, int(os.getenv("ARXIV_BACKFILL_WORKERS", "2"))))
logger = get_logger(__name__)
ARXIV_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "focus",
}


def _normalize_doc(raw: dict) -> dict:
    return {
        "source": str(raw.get("source", "")).strip(),
        "title": str(raw.get("title", "")).strip(),
        "summary": str(raw.get("summary") or raw.get("content") or "").strip(),
        "content": str(raw.get("content") or raw.get("summary") or "").strip(),
        "url": str(raw.get("url", "")).strip(),
        "published": raw.get("published"),
        "authors": raw.get("authors"),
        "domain_tags": raw.get("domain_tags") or [],
        "score": float(raw.get("score") or 0.0),
    }


def _score_to_similarity(score: float) -> float:
    """
    returns similarity score
    """
    return max(0.0, min(1.0, float(score or 0.0)))


def _apply_min_similarity_filter(docs: list[dict], min_similarity: float) -> list[dict]:
    filtered: list[dict] = []
    for doc in docs:
        similarity = _score_to_similarity(float(doc.get("score", 0.0)))
        doc["similarity"] = similarity
        logger.info(
            "Doc similarity computed. title='%s' url='%s' score=%.4f similarity=%.4f",
            doc.get("title", ""),
            doc.get("url", ""),
            float(doc.get("score", 0.0)),
            similarity,
        )
        if similarity >= min_similarity:
            filtered.append(doc)
    return filtered


def _sort_docs_by_score(docs: list[dict]) -> list[dict]:
    docs.sort(
        key=lambda doc: (
            -float(doc.get("similarity", _score_to_similarity(float(doc.get("score", 0.0))))),
            str(doc.get("title", "")),
            str(doc.get("url", "")),
        )
    )
    return docs


def _dedupe_docs(docs: list[dict], limit: int) -> list[dict]:
    deduped: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for doc in docs:
        key = (
            str(doc.get("source", "")),
            str(doc.get("url", "")),
            str(doc.get("title", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
        if len(deduped) >= limit:
            break
    return deduped


def _shorten_for_arxiv(query: str) -> str:
    text = " ".join((query or "").strip().lower().split())
    if not text:
        return ""

    tokens = re.findall(r"[a-z0-9_]+", text)
    keep: list[str] = []
    for token in tokens:
        if token in ARXIV_STOPWORDS:
            continue
        if len(token) < 4:
            continue
        if token not in keep:
            keep.append(token)

    if "large language model" in text and "large language model" not in keep:
        keep.insert(0, "large language model")
    elif "llm" in text and "llm" not in keep:
        keep.insert(0, "llm")

    priority = [
        "synthetic",
        "data",
        "bias",
        "evaluation",
        "quality",
        "fine",
        "tune",
        "training",
        "consensus",
        "autonomous",
    ]
    ordered: list[str] = []
    for term in priority:
        if term in keep and term not in ordered:
            ordered.append(term)
    for term in keep:
        if term not in ordered:
            ordered.append(term)

    short_query = " ".join(ordered[:8]).strip()
    return short_query or text[:120]


def _fetch_from_apis(queries: list[str], per_query_limit: int = 8) -> list[dict]:
    started_at = time.perf_counter()
    selected_queries = [query for query in queries[:BACKFILL_MAX_QUERIES] if query.strip()]
    logger.info(
        "API backfill start. query_count=%d selected_query_count=%d per_query_limit=%d wiki_workers=%d arxiv_workers=%d",
        len(queries),
        len(selected_queries),
        per_query_limit,
        WIKIPEDIA_BACKFILL_WORKERS,
        ARXIV_BACKFILL_WORKERS,
    )
    if not selected_queries:
        logger.info(
            "API backfill complete. fetched=0 normalized=0 elapsed_ms=%.2f",
            (time.perf_counter() - started_at) * 1000,
        )
        return []

    queued_tasks: list[tuple[int, int, str, str, Future]] = []
    with ThreadPoolExecutor(
        max_workers=WIKIPEDIA_BACKFILL_WORKERS,
        thread_name_prefix="wiki-backfill",
    ) as wiki_pool, ThreadPoolExecutor(
        max_workers=ARXIV_BACKFILL_WORKERS,
        thread_name_prefix="arxiv-backfill",
    ) as arxiv_pool:
        for query_idx, query in enumerate(selected_queries):
            logger.info("API backfill queue wikipedia query[%d]='%s'", query_idx, query)
            wiki_future = wiki_pool.submit(
                search_wikipedia,
                query=query,
                limit=per_query_limit,
            )
            queued_tasks.append((query_idx, 0, "wikipedia", query, wiki_future))

            arxiv_query = _shorten_for_arxiv(query)
            logger.info(
                "API backfill queue arXiv query[%d]='%s' (from='%s')",
                query_idx,
                arxiv_query,
                query,
            )
            arxiv_future = arxiv_pool.submit(
                search_arxiv,
                query=arxiv_query,
                limit=per_query_limit,
            )
            queued_tasks.append((query_idx, 1, "arxiv", arxiv_query, arxiv_future))

        fetched_count_by_tool = {"wikipedia": 0, "arxiv": 0}
        normalized_count_by_tool = {"wikipedia": 0, "arxiv": 0}
        normalized_chunks: list[tuple[int, int, str, list[dict]]] = []
        for query_idx, tool_rank, tool_name, emitted_query, future in queued_tasks:
            try:
                raw_results = future.result()
            except Exception as exc:
                logger.exception(
                    "API backfill task failed. tool=%s query_idx=%d query='%s' error=%s",
                    tool_name,
                    query_idx,
                    emitted_query,
                    exc,
                )
                raw_results = []

            safe_raw = raw_results if isinstance(raw_results, list) else []
            normalized = [_normalize_doc(item) for item in safe_raw if isinstance(item, dict)]
            fetched_count_by_tool[tool_name] += len(safe_raw)
            normalized_count_by_tool[tool_name] += len(normalized)
            normalized_chunks.append((query_idx, tool_rank, tool_name, normalized))

    normalized_chunks.sort(key=lambda item: (item[0], item[1], item[2]))
    normalized: list[dict] = []
    for _, _, _, chunk in normalized_chunks:
        normalized.extend(chunk)

    fetched_total = sum(fetched_count_by_tool.values())
    logger.info(
        "API backfill complete. fetched=%d normalized=%d wiki_fetched=%d wiki_normalized=%d arxiv_fetched=%d arxiv_normalized=%d elapsed_ms=%.2f",
        fetched_total,
        len(normalized),
        fetched_count_by_tool["wikipedia"],
        normalized_count_by_tool["wikipedia"],
        fetched_count_by_tool["arxiv"],
        normalized_count_by_tool["arxiv"],
        (time.perf_counter() - started_at) * 1000,
    )
    return normalized


def _retrieve_candidates_from_es(
    queries: list[str],
    target_limit: int,
    primary_index: str,
) -> tuple[list[dict], list[str]]:
    started_at = time.perf_counter()
    errors: list[str] = []
    logger.info(
        "Vector retrieval start. index=%s target_limit=%d query_count=%d",
        primary_index,
        target_limit,
        len(queries),
    )

    client = get_client()
    if client is None:
        errors.append("Elasticsearch client unavailable.")
        logger.warning(
            "Elasticsearch client unavailable. elapsed_ms=%.2f",
            (time.perf_counter() - started_at) * 1000,
        )
        return [], errors

    if not ensure_index(client, index_name=primary_index):
        errors.append(f"Elasticsearch index unavailable: {primary_index}.")
        logger.warning(
            "Elasticsearch index unavailable. index=%s elapsed_ms=%.2f",
            primary_index,
            (time.perf_counter() - started_at) * 1000,
        )
        return [], errors

    candidates: list[dict] = []
    per_query_size = max(10, min(50, target_limit))
    for query in queries[:6]:
        results = search_documents(
            client,
            query,
            size=per_query_size,
            index_name=primary_index,
        )
        candidates.extend(_normalize_doc(item) for item in results)

    candidates = _sort_docs_by_score(_dedupe_docs(candidates, limit=target_limit))
    logger.info(
        "Vector initial retrieval complete. candidates=%d elapsed_ms=%.2f",
        len(candidates),
        (time.perf_counter() - started_at) * 1000,
    )
    logger.info(
        "Vector retrieval complete. elapsed_ms=%.2f",
        (time.perf_counter() - started_at) * 1000,
    )
    return candidates, errors


def _backfill_index_from_apis(
    *,
    queries: list[str],
    primary_index: str,
) -> tuple[list[dict], list[str]]:
    """
    Call external tools and index returned documents into Elasticsearch.
    """
    started_at = time.perf_counter()
    errors: list[str] = []

    fetched = _fetch_from_apis(queries)
    if not fetched:
        errors.append("API backfill returned no results.")
        logger.warning("API backfill returned no results.")
        return [], errors

    client = get_client()
    if client is None:
        errors.append("Elasticsearch client unavailable during backfill indexing.")
        logger.warning("Elasticsearch client unavailable during backfill indexing.")
        return fetched, errors

    if not ensure_index(client, index_name=primary_index):
        errors.append(f"Elasticsearch index unavailable during backfill: {primary_index}.")
        logger.warning("Elasticsearch index unavailable during backfill. index=%s", primary_index)
        return fetched, errors

    indexed = index_documents(
        client,
        docs=[
            {
                "source": item.get("source", ""),
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "url": item.get("url", ""),
                "domain_tags": item.get("domain_tags", []),
                "published": item.get("published"),
                "authors": item.get("authors"),
            }
            for item in fetched
        ],
        index_name=primary_index,
    )
    if indexed <= 0:
        errors.append("Failed to index backfilled documents into Elasticsearch.")
        logger.warning("Backfill indexing failed. indexed=%d", indexed)
    else:
        logger.info(
            "Backfill indexing complete. indexed=%d elapsed_ms=%.2f",
            indexed,
            (time.perf_counter() - started_at) * 1000,
        )

    return fetched, errors


def _retrieve_and_filter_candidates(
    *,
    queries: list[str],
    target_limit: int,
    primary_index: str,
    min_similarity: float,
) -> tuple[list[dict], list[str], int]:
    """
    Run ES retrieval followed by min-similarity filtering.

    Returns:
    - filtered candidates
    - retrieval errors
    - original candidate count before filtering
    """
    candidates, retrieval_errors = _retrieve_candidates_from_es(
        queries=queries,
        target_limit=target_limit,
        primary_index=primary_index,
    )
    original_candidates = list(candidates)
    filtered = _apply_min_similarity_filter(
        docs=original_candidates,
        min_similarity=min_similarity,
    )
    logger.info(
        "Min-similarity filter applied. before=%d after=%d min_similarity=%.2f",
        len(original_candidates),
        len(filtered),
        min_similarity,
    )
    return filtered, retrieval_errors, len(original_candidates)


def reranker_node(state: AgentState) -> dict:
    """
    Vector retrieval node.

    Planner emits deterministic control object; this node performs:
    1) Vector retrieval from Elasticsearch
    2) API backfill + index when vector index is sparse
    3) Vector requery and top-k selection
    """
    started_at = time.perf_counter()
    errors = list(state.get("errors", []))
    prompt = (state.get("prompt") or "").strip()
    queries = [str(item).strip() for item in state.get("queries", []) if str(item).strip()]
    primary_index = f"{ES_INDEX_PREFIX}_general"

    candidate_limit = max(
        50,
        min(
            100,
            int(
                os.getenv(
                    "VECTOR_CANDIDATE_LIMIT",
                    os.getenv("RERANK_CANDIDATE_LIMIT", str(DEFAULT_CANDIDATE_LIMIT)),
                )
            ),
        ),
    )
    top_k = max(
        1,
        min(
            20,
            int(os.getenv("VECTOR_TOP_K", os.getenv("RERANK_TOP_K", str(DEFAULT_TOP_K)))),
        ),
    )
    min_similarity = max(
        0.0,
        min(1.0, float(os.getenv("VECTOR_MIN_SIMILARITY", str(DEFAULT_VECTOR_MIN_SIMILARITY)))),
    )
    min_sources_for_llm = max(
        1,
        min(
            top_k,
            int(os.getenv("MIN_SOURCES_FOR_LLM", str(DEFAULT_MIN_SOURCES_FOR_LLM))),
        ),
    )

    if not queries and prompt:
        queries = [prompt]
    logger.info(
        "Vector node start. trace_id=%s prompt_len=%d queries=%d index=%s",
        state.get("trace_id", ""),
        len(prompt),
        len(queries),
        primary_index,
    )

    filtered, retrieval_errors, _ = _retrieve_and_filter_candidates(
        queries=queries,
        target_limit=candidate_limit,
        primary_index=primary_index,
        min_similarity=min_similarity,
    )
    errors.extend(retrieval_errors)

    # Backfill is triggered only when filtered results are below minimum.
    if len(filtered) < min_sources_for_llm:
        logger.info(
            "Filtered candidates below minimum for synthesis. filtered=%d min_sources_for_llm=%d. Running API backfill and using tool results directly.",
            len(filtered),
            min_sources_for_llm,
        )
        backfill_docs, backfill_errors = _backfill_index_from_apis(
            queries=queries,
            primary_index=primary_index,
        )
        errors.extend(backfill_errors)

        if backfill_docs:
            filtered = _sort_docs_by_score(
                _dedupe_docs(
                    docs=[_normalize_doc(doc) for doc in backfill_docs if isinstance(doc, dict)],
                    limit=candidate_limit,
                )
            )
            logger.info(
                "Using tool results directly after backfill. candidates=%d",
                len(filtered),
            )

    filtered = _sort_docs_by_score(filtered)

    top_docs = filtered[:top_k]
    logger.info(
        "Vector node complete. candidates=%d top_docs=%d min_sources_for_llm=%d min_similarity=%.2f errors=%d elapsed_ms=%.2f",
        len(filtered),
        len(top_docs),
        min_sources_for_llm,
        min_similarity,
        len(errors),
        (time.perf_counter() - started_at) * 1000,
    )

    if not top_docs:
        logger.warning(
            "No top docs produced. prompt='%s' queries=%s",
            prompt,
            queries,
        )

    sources = [
        {
            "title": doc.get("title", ""),
            "summary": doc.get("summary", ""),
            "url": doc.get("url", ""),
            "source": doc.get("source", ""),
            "published": doc.get("published"),
            "authors": doc.get("authors"),
        }
        for doc in top_docs
    ]
    evidence = [
        {
            "statement": source.get("summary", ""),
            "url": source.get("url", ""),
            "source": source.get("source", ""),
        }
        for source in sources
    ]

    return {
        "candidates": filtered,
        "sources": sources,
        "evidence": evidence,
        "errors": errors,
    }
