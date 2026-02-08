from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200").rstrip("/")
ELASTICSEARCH_INDEX = os.getenv("ELASTICSEARCH_INDEX", "search_documents")

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "source": {"type": "keyword"},
            "title": {"type": "text"},
            "content": {"type": "text"},
            "url": {"type": "keyword"},
            "domain_tags": {"type": "keyword"},
            "published": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
            "authors": {"type": "keyword"},
        }
    }
}


def _request(method: str, url: str, payload: dict | None = None) -> tuple[int, str]:
    body = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, method=method, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def main() -> None:
    exists_code, _ = _request("HEAD", f"{ELASTICSEARCH_URL}/{ELASTICSEARCH_INDEX}")
    if exists_code == 200:
        print(f"[es-init] index '{ELASTICSEARCH_INDEX}' already exists")
        return

    create_code, create_body = _request(
        "PUT",
        f"{ELASTICSEARCH_URL}/{ELASTICSEARCH_INDEX}",
        payload=INDEX_MAPPING,
    )
    if create_code in {200, 201}:
        print(f"[es-init] created index '{ELASTICSEARCH_INDEX}'")
        return

    print(f"[es-init] failed to create index '{ELASTICSEARCH_INDEX}': {create_code}")
    print(create_body)


if __name__ == "__main__":
    main()
