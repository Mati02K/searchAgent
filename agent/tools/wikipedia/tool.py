from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any


WIKIPEDIA_REST_SEARCH_ENDPOINTS = (
    "https://en.wikipedia.org/w/rest.php/v1/search/title",
    "https://en.wikipedia.org/w/rest.php/v1/search/page",
)
WIKIPEDIA_SUMMARY_ENDPOINT = "https://en.wikipedia.org/api/rest_v1/page/summary"


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
    encoded_title = urllib.parse.quote(title, safe="")
    url = f"{WIKIPEDIA_SUMMARY_ENDPOINT}/{encoded_title}"
    try:
        payload = _http_get_json(url)
    except Exception as exc:
        print(f"[wikipedia] summary fetch failed for '{title}': {exc}")
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

    return {
        "title": page_title,
        "summary": summary,
        "url": url_value,
        "source": "wikipedia",
        "published": None,
        "authors": None,
    }


def search_wikipedia(query: str, limit: int = 5) -> list[dict]:
    """
    Search Wikipedia using official MediaWiki APIs and return page summaries.

    Returns structured results with schema:
    title, summary, url, source, published, authors.
    """
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return []

    try:
        max_results = max(1, min(int(limit), 20))
    except Exception:
        max_results = 5

    search_items: list[dict[str, Any]] = []
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
            print(f"[wikipedia] search failed via '{endpoint}': {exc}")

    if not search_items:
        # Fallback to official MediaWiki Action API if REST search is unavailable.
        fallback_url = (
            "https://en.wikipedia.org/w/api.php?"
            f"action=query&list=search&format=json&srlimit={max_results}&srsearch={urllib.parse.quote(cleaned_query)}"
        )
        try:
            payload = _http_get_json(fallback_url)
            query_block = payload.get("query")
            if isinstance(query_block, dict):
                values = query_block.get("search")
                if isinstance(values, list):
                    search_items = [item for item in values if isinstance(item, dict)]
        except Exception as exc:
            print(f"[wikipedia] fallback search failed: {exc}")

    if not search_items:
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

    return results
