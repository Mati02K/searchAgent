from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import os
import re
import time
from functools import lru_cache

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
MIN_CANDIDATES_BEFORE_BACKFILL = 20
DEFAULT_TOP_K = 8
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
ES_INDEX_PREFIX = os.getenv("ELASTICSEARCH_INDEX_PREFIX", os.getenv("ELASTICSEARCH_INDEX", "search_documents"))
BACKFILL_MAX_QUERIES = max(1, min(8, int(os.getenv("BACKFILL_MAX_QUERIES", "4"))))
WIKIPEDIA_BACKFILL_WORKERS = max(
    1, min(8, int(os.getenv("WIKIPEDIA_BACKFILL_WORKERS", "3")))
)
ARXIV_BACKFILL_WORKERS = max(
    1, min(4, int(os.getenv("ARXIV_BACKFILL_WORKERS", "2")))
)
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
    }


def _rules_from_state(state: AgentState) -> tuple[list[str], list[str]]:
    control = state.get("planner_control", {})
    rules = control.get("relevance_rules", {}) if isinstance(control, dict) else {}
    must_include = [str(item).strip().lower() for item in rules.get("must_include", []) if str(item).strip()]
    must_exclude = [str(item).strip().lower() for item in rules.get("must_exclude", []) if str(item).strip()]
    return must_include, must_exclude


def _index_names_from_state(state: AgentState) -> tuple[str, list[str]]:
    control = state.get("planner_control", {})
    target_key = "general"
    if isinstance(control, dict):
        target_key = str(control.get("target_index_key", "general")).strip().lower() or "general"
    primary_index = f"{ES_INDEX_PREFIX}_{target_key}"
    fallback_indexes: list[str] = []
    general_index = f"{ES_INDEX_PREFIX}_general"
    if primary_index != general_index:
        fallback_indexes.append(general_index)
    return primary_index, fallback_indexes


def _apply_rule_filters(docs: list[dict], must_include: list[str], must_exclude: list[str]) -> list[dict]:
    filtered: list[dict] = []
    for doc in docs:
        text = f"{doc.get('title', '')} {doc.get('content', '')}".lower()
        if any(term in text for term in must_exclude):
            continue
        if must_include and not any(term in text for term in must_include):
            continue
        filtered.append(doc)
    return filtered


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
            wiki_future = wiki_pool.submit(search_wikipedia, query=query, limit=per_query_limit)
            queued_tasks.append((query_idx, 0, "wikipedia", query, wiki_future))

            arxiv_query = _shorten_for_arxiv(query)
            logger.info(
                "API backfill queue arXiv query[%d]='%s' (from='%s')",
                query_idx,
                arxiv_query,
                query,
            )
            arxiv_future = arxiv_pool.submit(search_arxiv, query=arxiv_query, limit=per_query_limit)
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
    must_include: list[str],
    must_exclude: list[str],
    target_limit: int,
    primary_index: str,
    fallback_indexes: list[str],
) -> tuple[list[dict], list[str]]:
    started_at = time.perf_counter()
    errors: list[str] = []
    logger.info(
        "ES retrieval start. primary_index=%s fallback_indexes=%s target_limit=%d query_count=%d must_include=%d must_exclude=%d",
        primary_index,
        fallback_indexes,
        target_limit,
        len(queries),
        len(must_include),
        len(must_exclude),
    )
    client = get_client()
    if client is None:
        errors.append("Elasticsearch client unavailable. Falling back to API hydration only.")
        logger.warning(
            "Elasticsearch client unavailable. elapsed_ms=%.2f",
            (time.perf_counter() - started_at) * 1000,
        )
        return _fetch_from_apis(queries), errors

    if not ensure_index(client, index_name=primary_index):
        errors.append(f"Elasticsearch index unavailable: {primary_index}.")
        logger.warning(
            "Elasticsearch index unavailable. index=%s elapsed_ms=%.2f",
            primary_index,
            (time.perf_counter() - started_at) * 1000,
        )
        return _fetch_from_apis(queries), errors
    for fallback_index in fallback_indexes:
        ensure_index(client, index_name=fallback_index)

    candidates: list[dict] = []
    per_query_size = max(10, min(50, target_limit))
    index_names = [primary_index, *fallback_indexes]
    for query in queries[:6]:
        for index_name in index_names:
            results = search_documents(
                client,
                query,
                size=per_query_size,
                index_name=index_name,
                must_include=must_include,
                must_exclude=must_exclude,
            )
            candidates.extend(_normalize_doc(item) for item in results)

    candidates = _dedupe_docs(candidates, limit=target_limit)
    logger.info(
        "ES initial retrieval complete. candidates=%d elapsed_ms=%.2f",
        len(candidates),
        (time.perf_counter() - started_at) * 1000,
    )
    if len(candidates) >= MIN_CANDIDATES_BEFORE_BACKFILL:
        logger.info(
            "ES retrieval complete without backfill. elapsed_ms=%.2f",
            (time.perf_counter() - started_at) * 1000,
        )
        return candidates, errors

    fetched = _fetch_from_apis(queries)
    if fetched:
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
            errors.append("Failed to index hydrated documents into Elasticsearch.")
            logger.warning("Backfill indexing failed. indexed=%d", indexed)
        else:
            logger.info("Backfill indexing complete. indexed=%d", indexed)

        requeried: list[dict] = []
        for query in queries[:6]:
            for index_name in index_names:
                results = search_documents(
                    client,
                    query,
                    size=per_query_size,
                    index_name=index_name,
                    must_include=must_include,
                    must_exclude=must_exclude,
                )
                requeried.extend(_normalize_doc(item) for item in results)
        if requeried:
            candidates = _dedupe_docs(requeried, limit=target_limit)
            logger.info("Requery after backfill complete. candidates=%d", len(candidates))
        else:
            candidates = _dedupe_docs(fetched, limit=target_limit)
            logger.info("Using fetched fallback docs without ES requery. candidates=%d", len(candidates))

    logger.info(
        "ES retrieval complete with backfill. elapsed_ms=%.2f",
        (time.perf_counter() - started_at) * 1000,
    )
    return candidates, errors


def _lexical_score(query: str, doc: dict) -> float:
    query_terms = {term for term in query.lower().split() if term}
    text_terms = set(f"{doc.get('title', '')} {doc.get('content', '')}".lower().split())
    if not query_terms:
        return 0.0
    return len(query_terms & text_terms) / max(1, len(query_terms))


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL_NAME)


def _rerank(query: str, docs: list[dict], errors: list[str]) -> list[dict]:
    started_at = time.perf_counter()
    if not docs:
        logger.warning("Rerank skipped: no docs.")
        return []

    try:
        logger.info("Cross-encoder rerank start. model=%s docs=%d", RERANK_MODEL_NAME, len(docs))
        reranker = _get_reranker()
        pairs = [(query, f"{doc.get('title', '')} {doc.get('content', '')}") for doc in docs]
        scores = reranker.predict(pairs)
        for idx, score in enumerate(scores):
            docs[idx]["rerank_score"] = float(score)
        logger.info(
            "Cross-encoder rerank complete. elapsed_ms=%.2f",
            (time.perf_counter() - started_at) * 1000,
        )
    except Exception as exc:
        errors.append(f"Cross-encoder unavailable: {exc}")
        logger.exception("Cross-encoder unavailable; falling back to lexical scoring: %s", exc)
        for doc in docs:
            doc["rerank_score"] = _lexical_score(query, doc)

    docs.sort(
        key=lambda doc: (
            -float(doc.get("rerank_score", 0.0)),
            str(doc.get("title", "")),
            str(doc.get("url", "")),
        )
    )
    logger.info("Rerank ordering complete. elapsed_ms=%.2f", (time.perf_counter() - started_at) * 1000)
    return docs


def reranker_node(state: AgentState) -> dict:
    """
    Retrieval + ML reranking node.

    Planner emits deterministic control object; this node performs:
    1) ES retrieval
    2) API backfill + index when ES is sparse
    3) Cross-encoder reranking
    """
    started_at = time.perf_counter()
    errors = list(state.get("errors", []))
    prompt = (state.get("prompt") or "").strip()
    queries = [str(item).strip() for item in state.get("queries", []) if str(item).strip()]
    must_include, must_exclude = _rules_from_state(state)
    primary_index, fallback_indexes = _index_names_from_state(state)
    candidate_limit = max(50, min(100, int(os.getenv("RERANK_CANDIDATE_LIMIT", str(DEFAULT_CANDIDATE_LIMIT)))))
    top_k = max(6, min(10, int(os.getenv("RERANK_TOP_K", str(DEFAULT_TOP_K)))))

    if not queries and prompt:
        queries = [prompt]
    logger.info(
        "Reranker node start. trace_id=%s prompt_len=%d queries=%d primary_index=%s fallback_indexes=%s",
        state.get("trace_id", ""),
        len(prompt),
        len(queries),
        primary_index,
        fallback_indexes,
    )

    candidates, retrieval_errors = _retrieve_candidates_from_es(
        queries=queries,
        must_include=must_include,
        must_exclude=must_exclude,
        target_limit=candidate_limit,
        primary_index=primary_index,
        fallback_indexes=fallback_indexes,
    )
    errors.extend(retrieval_errors)
    original_candidates = list(candidates)
    prefilter_count = len(original_candidates)
    candidates = _apply_rule_filters(original_candidates, must_include, must_exclude)
    logger.info(
        "Rule filter applied. before=%d after=%d must_include=%s must_exclude=%s",
        prefilter_count,
        len(candidates),
        must_include,
        must_exclude,
    )
    if prefilter_count > 0 and not candidates:
        # Avoid complete collapse from over-strict include rules.
        logger.warning("Rule filters removed all candidates; relaxing include filter.")
        candidates = _apply_rule_filters(
            docs=original_candidates,
            must_include=[],
            must_exclude=must_exclude,
        )

    ranked = _rerank(prompt, candidates, errors)
    top_docs = ranked[:top_k]
    logger.info(
        "Reranker complete. candidates=%d top_docs=%d errors=%d elapsed_ms=%.2f",
        len(candidates),
        len(top_docs),
        len(errors),
        (time.perf_counter() - started_at) * 1000,
    )
    if not top_docs:
        logger.warning(
            "No top docs produced. prompt='%s' queries=%s must_include=%s must_exclude=%s",
            prompt,
            queries,
            must_include,
            must_exclude,
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
        "candidates": candidates,
        "sources": sources,
        "evidence": evidence,
        "errors": errors,
    }
