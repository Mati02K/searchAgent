from __future__ import annotations

import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from logging_utils import get_logger

logger = get_logger(__name__)

ARXIV_API_ENDPOINT = os.getenv(
    "ARXIV_API_ENDPOINT", "https://export.arxiv.org/api/query"
)
ARXIV_API_ENDPOINTS = tuple(
    endpoint.strip()
    for endpoint in os.getenv(
        "ARXIV_API_ENDPOINTS",
        ",".join(
            [
                ARXIV_API_ENDPOINT,
                "https://arxiv.org/api/query",
                "http://export.arxiv.org/api/query",
            ]
        ),
    ).split(",")
    if endpoint.strip()
)
ARXIV_ATOM_NAMESPACE = os.getenv("ARXIV_ATOM_NAMESPACE", "http://www.w3.org/2005/Atom")
ATOM_NS = {"atom": ARXIV_ATOM_NAMESPACE}
ARXIV_REQUEST_DELAY_SECONDS = float(os.getenv("ARXIV_REQUEST_DELAY_SECONDS", "0.0"))
ARXIV_HTTP_TIMEOUT_SECONDS = float(os.getenv("ARXIV_HTTP_TIMEOUT_SECONDS", "10.0"))


def _http_get_text(url: str, timeout: float = ARXIV_HTTP_TIMEOUT_SECONDS) -> str:
    started_at = time.perf_counter()
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "SearchAgent/0.1 (MCP arXiv Tool)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
            logger.info(
                "arXiv HTTP fetch success. attempt=1 status=%s elapsed_ms=%.2f",
                getattr(response, "status", "unknown"),
                (time.perf_counter() - started_at) * 1000,
            )
            return text
    except urllib.error.HTTPError as exc:
        body_snippet = ""
        try:
            body_snippet = exc.read(512).decode("utf-8", errors="replace")
        except Exception:
            body_snippet = ""
        logger.error(
            "arXiv HTTP error. status=%s reason=%s url=%s body_snippet=%r elapsed_ms=%.2f",
            exc.code,
            exc.reason,
            url,
            body_snippet,
            (time.perf_counter() - started_at) * 1000,
        )
        raise
    except urllib.error.URLError as exc:
        logger.error(
            "arXiv URL error. reason=%s url=%s elapsed_ms=%.2f",
            exc.reason,
            url,
            (time.perf_counter() - started_at) * 1000,
        )
        raise
    except TimeoutError as exc:
        logger.error(
            "arXiv timeout error. timeout=%s url=%s error=%s elapsed_ms=%.2f",
            timeout,
            url,
            exc,
            (time.perf_counter() - started_at) * 1000,
        )
        raise
    except socket.timeout as exc:
        logger.error(
            "arXiv socket timeout. timeout=%s url=%s error=%s elapsed_ms=%.2f",
            timeout,
            url,
            exc,
            (time.perf_counter() - started_at) * 1000,
        )
        raise
    except Exception as exc:
        logger.exception(
            "arXiv HTTP fetch unexpected failure. url=%s error=%s elapsed_ms=%.2f",
            url,
            exc,
            (time.perf_counter() - started_at) * 1000,
        )
        raise


def _find_best_url(entry: ET.Element, fallback: str) -> str:
    for link in entry.findall("atom:link", ATOM_NS):
        href = link.attrib.get("href", "").strip()
        title = link.attrib.get("title", "").strip().lower()
        if href and title == "pdf":
            return href
    for link in entry.findall("atom:link", ATOM_NS):
        href = link.attrib.get("href", "").strip()
        rel = link.attrib.get("rel", "").strip().lower()
        if href and rel == "alternate":
            return href
    return fallback


def _build_search_url(endpoint: str, query: str, max_results: int) -> str:
    params = {
        "search_query": f"all:{query}",
        "start": "0",
        "max_results": str(max_results),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    return f"{endpoint}?{urllib.parse.urlencode(params)}"


def search_arxiv(query: str, limit: int = 5) -> list[dict]:
    """
    Search arXiv via official Atom API and return structured paper metadata.

    Returns schema:
    title, summary, url, source, published, authors.
    """
    started_at = time.perf_counter()
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        logger.info(
            "arXiv search called with empty query. elapsed_ms=%.2f",
            (time.perf_counter() - started_at) * 1000,
        )
        return []

    try:
        max_results = max(1, min(int(limit), 20))
    except Exception:
        max_results = 5

    logger.info(
        "arXiv search start. query=%s limit=%d endpoints=%s timeout=%.1fs",
        cleaned_query,
        max_results,
        list(ARXIV_API_ENDPOINTS),
        ARXIV_HTTP_TIMEOUT_SECONDS,
    )
    if ARXIV_REQUEST_DELAY_SECONDS > 0:
        time.sleep(ARXIV_REQUEST_DELAY_SECONDS)

    root: ET.Element | None = None
    last_error: Exception | None = None
    selected_url = ""
    for endpoint in ARXIV_API_ENDPOINTS:
        selected_url = _build_search_url(endpoint, cleaned_query, max_results)
        try:
            xml_text = _http_get_text(selected_url, timeout=ARXIV_HTTP_TIMEOUT_SECONDS)
            root = ET.fromstring(xml_text)
            logger.info("arXiv endpoint success. endpoint=%s", endpoint)
            break
        except ET.ParseError as exc:
            last_error = exc
            logger.error(
                "arXiv XML parse failure. query=%s endpoint=%s error=%s elapsed_ms=%.2f",
                cleaned_query,
                endpoint,
                exc,
                (time.perf_counter() - started_at) * 1000,
            )
        except Exception as exc:
            last_error = exc
            logger.warning(
                "arXiv endpoint failed. query=%s endpoint=%s error=%s elapsed_ms=%.2f",
                cleaned_query,
                endpoint,
                exc,
                (time.perf_counter() - started_at) * 1000,
            )

    if root is None:
        logger.error(
            "arXiv search failed (all endpoints, fail-silent). query=%s last_url=%s error=%s elapsed_ms=%.2f",
            cleaned_query,
            selected_url,
            last_error,
            (time.perf_counter() - started_at) * 1000,
        )
        return []

    results: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
        summary = (
            entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or ""
        ).strip()
        published = (
            entry.findtext("atom:published", default="", namespaces=ATOM_NS) or ""
        ).strip()
        entry_id = (entry.findtext("atom:id", default="", namespaces=ATOM_NS) or "").strip()

        authors: list[str] = []
        for author_el in entry.findall("atom:author", ATOM_NS):
            name = (
                author_el.findtext("atom:name", default="", namespaces=ATOM_NS) or ""
            ).strip()
            if name:
                authors.append(name)

        result = {
            "title": title,
            "summary": summary,
            "url": _find_best_url(entry, entry_id),
            "source": "arxiv",
            "published": published or None,
            "authors": authors or None,
        }
        results.append(result)
        if len(results) >= max_results:
            break

    logger.info(
        "arXiv search complete. query=%s results=%d elapsed_ms=%.2f",
        cleaned_query,
        len(results),
        (time.perf_counter() - started_at) * 1000,
    )
    return results
