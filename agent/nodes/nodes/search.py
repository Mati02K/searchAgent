from __future__ import annotations

from nodes.state import AgentState
from tools.search_web import search_web


def search_sources(state: AgentState) -> dict:
    """Collect source snippets using the local deterministic search stub."""
    if (state.get("report") or "").startswith("## Error"):
        return {}

    errors = list(state.get("errors", []))
    queries = list(state.get("queries", []))
    if not queries:
        prompt = state.get("prompt", "").strip()
        if prompt:
            queries = [prompt]

    sources: list[dict] = []
    evidence: list[dict] = []

    for query in queries:
        try:
            results = search_web(query)
            for item in results:
                source_item = {
                    "query": query,
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("snippet", ""),
                    "source": item.get("source", ""),
                }
                sources.append(source_item)
                evidence.append(
                    {
                        "query": query,
                        "statement": source_item["snippet"],
                        "url": source_item["url"],
                        "source": source_item["source"],
                    }
                )
        except Exception as exc:
            errors.append(f"Search failed for query '{query}': {exc}")

    unique_sources: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in sources:
        key = (item.get("url", ""), item.get("title", ""))
        if key in seen:
            continue
        seen.add(key)
        unique_sources.append(item)

    return {
        "sources": unique_sources,
        "evidence": evidence,
        "errors": errors,
    }
