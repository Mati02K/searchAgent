from __future__ import annotations

import hashlib
import re


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "topic"


def search_web(query: str) -> list[dict]:
    """Deterministic local web search stub."""
    normalized = " ".join((query or "").strip().split()).lower()
    slug = _slugify(normalized)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    token = digest[:8]

    return [
        {
            "title": f"{query.strip().title()} overview",
            "url": f"https://example.org/{slug}/{token}-overview",
            "snippet": f"Background context and key definitions related to {query.strip()}.",
            "source": "Example Research Digest",
        },
        {
            "title": f"Benefits of {query.strip()}",
            "url": f"https://example.net/{slug}/{token}-benefits",
            "snippet": f"Reported advantages, opportunities, and use-cases for {query.strip()}.",
            "source": "Open Knowledge Review",
        },
        {
            "title": f"Risks and limitations of {query.strip()}",
            "url": f"https://example.com/{slug}/{token}-risks",
            "snippet": f"Documented risks, limitations, and mitigation patterns for {query.strip()}.",
            "source": "Policy Lab Notes",
        },
    ]
