from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any


ARXIV_API_ENDPOINT = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _http_get_text(url: str, timeout: float = 10.0) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "SearchAgent/0.1 (MCP arXiv Tool)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


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
    cleaned_query = (query or "").strip()
    if not cleaned_query:
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

    try:
        xml_text = _http_get_text(url)
        root = ET.fromstring(xml_text)
    except Exception as exc:
        print(f"[arxiv] search failed: {exc}")
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

    return results
