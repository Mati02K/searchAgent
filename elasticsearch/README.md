# SearchAgent Elasticsearch

This folder provides local Elasticsearch infrastructure for SearchAgent retrieval.

## What it does

- Runs Elasticsearch as a separate Docker container.
- Initializes an index named `search_documents`.
- Stores arXiv/Wikipedia documents with fields:
  - `source`
  - `title`
  - `content`
  - `url`
  - `domain_tags`

## Start

```bash
cd elasticsearch
docker compose up -d
```

The `es-init` service creates the index automatically after Elasticsearch is healthy.

## Stop

```bash
docker compose down
```

## Environment (app-side)

Set these for the Python app:

```bash
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_INDEX=search_documents
```

## Verify

```bash
curl -s http://localhost:9200/search_documents | jq
```
