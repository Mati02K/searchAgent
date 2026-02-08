from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any

from logging_utils import get_logger

logger = get_logger(__name__)

WIKIPEDIA_REST_SEARCH_ENDPOINTS = tuple(
    endpoint.strip()
    for endpoint in os.getenv(
        "WIKIPEDIA_REST_SEARCH_ENDPOINTS",
        "https://en.wikipedia.org/w/rest.php/v1/search/title,https://en.wikipedia.org/w/rest.php/v1/search/page",
    ).split(",")
    if endpoint.strip()
)
WIKIPEDIA_SUMMARY_ENDPOINT = os.getenv(
    "WIKIPEDIA_SUMMARY_ENDPOINT",
    "https://en.wikipedia.org/api/rest_v1/page/summary",
).rstrip("/")


def _http_get_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "SearchAgent/0.1 (MCP Wikipedia Tool)",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _extract_search_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("pages", "results", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _get_page_summary(title: str) -> dict[str, Any] | None:
    started_at = time.perf_counter()
    encoded_title = urllib.parse.quote(title, safe="")
    url = f"{WIKIPEDIA_SUMMARY_ENDPOINT}/{encoded_title}"
    try:
        payload = _http_get_json(url)
    except Exception as exc:
        logger.warning(
            "Wikipedia summary fetch failed. title=%s error=%s elapsed_ms=%.2f",
            title,
            exc,
            (time.perf_counter() - started_at) * 1000,
        )
        return None

    page_title = str(payload.get("title") or title).strip()
    summary = str(payload.get("extract") or "").strip()
    url_value = ""
    content_urls = payload.get("content_urls")
    if isinstance(content_urls, dict):
        desktop = content_urls.get("desktop")
        if isinstance(desktop, dict):
            url_value = str(desktop.get("page") or "").strip()
    if not url_value:
        url_value = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page_title.replace(' ', '_'))}"

    result = {
        "title": page_title,
        "summary": summary,
        "url": url_value,
        "source": "wikipedia",
        "published": None,
        "authors": None,
    }
    logger.info(
        "Wikipedia summary fetched. title=%s elapsed_ms=%.2f",
        page_title,
        (time.perf_counter() - started_at) * 1000,
    )
    return result


def search_wikipedia(query: str, limit: int = 5) -> list[dict]:
    """
    Search Wikipedia using official MediaWiki APIs and return page summaries.

    Returns structured results with schema:
    title, summary, url, source, published, authors.
    """
    started_at = time.perf_counter()
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        logger.info(
            "Wikipedia search called with empty query. elapsed_ms=%.2f",
            (time.perf_counter() - started_at) * 1000,
        )
        return []

    try:
        max_results = max(1, min(int(limit), 20))
    except Exception:
        max_results = 5

    search_items: list[dict[str, Any]] = []
    logger.info("Wikipedia search start. query=%s limit=%d", cleaned_query, max_results)
    for endpoint in WIKIPEDIA_REST_SEARCH_ENDPOINTS:
        search_url = (
            f"{endpoint}?q={urllib.parse.quote(cleaned_query)}&limit={max_results}"
        )
        try:
            payload = _http_get_json(search_url)
            search_items = _extract_search_items(payload)
            if search_items:
                break
        except Exception as exc:
            logger.warning("Wikipedia search failed. endpoint=%s error=%s", endpoint, exc)

    if not search_items:
        logger.info(
            "Wikipedia search returned no search items. query=%s elapsed_ms=%.2f",
            cleaned_query,
            (time.perf_counter() - started_at) * 1000,
        )
        return []

    titles: list[str] = []
    for item in search_items:
        title = str(item.get("title") or "").strip()
        if title and title not in titles:
            titles.append(title)
        if len(titles) >= max_results:
            break

    results: list[dict] = []
    for title in titles:
        summary_result = _get_page_summary(title)
        if summary_result is not None:
            results.append(summary_result)
        if len(results) >= max_results:
            break

    logger.info(
        "Wikipedia search complete. query=%s results=%d elapsed_ms=%.2f",
        cleaned_query,
        len(results),
        (time.perf_counter() - started_at) * 1000,
    )
    return results
