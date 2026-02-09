# SearchAgent

SearchAgent is a Dockerized research pipeline with:
- `safety` node
- `planner` node
- `reranker` (Elasticsearch + MCP tools)
- `search` synthesis node (final Gemini call)

## Services

`docker-compose.yml` runs 3 services:
1. `elasticsearch` (vector index store)
2. `mcp-server` (Wikipedia + arXiv tools over JSON-RPC HTTP)
3. `search-agent` (FastAPI app)

## API

- Endpoint: `POST /v1/research`
- URL: `http://localhost:8000/v1/research`
- Body:

```json
{
  "prompt": "What are the risks and benefits of synthetic data for LLM training?"
}
```

## Bring Up

```bash
docker compose down
docker compose up -d --build
docker compose ps
```

## Logs

```bash
docker compose logs -f search-agent mcp-server elasticsearch
```

App file logs are also written to:
- `logs/search_agent.log`

## Test the System

### 1) Direct API call

```bash
curl -s -X POST "http://localhost:8000/v1/research" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What are recent developments in consensus for autonomous systems?"}'
```

### 2) Interactive test client

```bash
python3 tests/test.py
```

Type prompts in loop. Type `exit` to stop.

## Elasticsearch Cleanup Commands

### Delete indexed documents (default index)

```bash
curl -X DELETE "http://localhost:9200/search_vectors_general?pretty"
```

### Verify indices

```bash
curl -s "http://localhost:9200/_cat/indices?v"
```

### Optional full reset (removes ES volume data)

```bash
docker compose down -v
docker compose up -d --build
```

`search-agent` recreates required index mappings at runtime (`ensure_index(...)`).
