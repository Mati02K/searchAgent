from __future__ import annotations

import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from logging_utils import get_logger

logger = get_logger(__name__)

ARXIV_API_ENDPOINT = os.getenv(
    "ARXIV_API_ENDPOINT", "https://export.arxiv.org/api/query"
)
ARXIV_ATOM_NAMESPACE = os.getenv("ARXIV_ATOM_NAMESPACE", "http://www.w3.org/2005/Atom")
ATOM_NS = {"atom": ARXIV_ATOM_NAMESPACE}
ARXIV_RETRY_ATTEMPTS = max(1, int(os.getenv("ARXIV_RETRY_ATTEMPTS", "4")))
ARXIV_BACKOFF_BASE_SECONDS = float(os.getenv("ARXIV_BACKOFF_BASE_SECONDS", "1.0"))
ARXIV_REQUEST_DELAY_SECONDS = float(os.getenv("ARXIV_REQUEST_DELAY_SECONDS", "1.0"))


def _http_get_text(url: str, timeout: float = 10.0) -> str:
    started_at = time.perf_counter()
    last_exc: Exception | None = None
    for attempt in range(1, ARXIV_RETRY_ATTEMPTS + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "SearchAgent/0.1 (MCP arXiv Tool)"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                text = response.read().decode("utf-8", errors="replace")
                logger.info(
                    "arXiv HTTP fetch success. attempt=%d elapsed_ms=%.2f",
                    attempt,
                    (time.perf_counter() - started_at) * 1000,
                )
                return text
        except urllib.error.HTTPError as exc:
            last_exc = exc
            status = int(getattr(exc, "code", 0))
            if status == 429 or status >= 500:
                backoff = ARXIV_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "arXiv request throttled/failed. status=%s attempt=%d/%d backoff=%.2fs",
                    status,
                    attempt,
                    ARXIV_RETRY_ATTEMPTS,
                    backoff,
                )
                time.sleep(backoff)
                continue
            raise
        except Exception as exc:
            last_exc = exc
            backoff = ARXIV_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "arXiv request error. attempt=%d/%d backoff=%.2fs error=%s",
                attempt,
                ARXIV_RETRY_ATTEMPTS,
                backoff,
                exc,
            )
            time.sleep(backoff)
    if last_exc is not None:
        logger.error(
            "arXiv HTTP fetch failed after retries. attempts=%d elapsed_ms=%.2f",
            ARXIV_RETRY_ATTEMPTS,
            (time.perf_counter() - started_at) * 1000,
        )
        raise last_exc
    raise RuntimeError("arXiv request failed without exception details.")


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

    params = {
        "search_query": f"all:{cleaned_query}",
        "start": "0",
        "max_results": str(max_results),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API_ENDPOINT}?{urllib.parse.urlencode(params)}"
    logger.info("arXiv search start. query=%s limit=%d", cleaned_query, max_results)
    if ARXIV_REQUEST_DELAY_SECONDS > 0:
        time.sleep(ARXIV_REQUEST_DELAY_SECONDS)

    try:
        xml_text = _http_get_text(url)
        root = ET.fromstring(xml_text)
    except Exception as exc:
        logger.exception(
            "arXiv search failed. query=%s error=%s elapsed_ms=%.2f",
            cleaned_query,
            exc,
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
