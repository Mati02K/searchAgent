from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure `agent/` modules are importable when running from repo root.
_AGENT_ROOT = Path(__file__).resolve().parents[1] / "agent"
if str(_AGENT_ROOT) not in sys.path:
    sys.path.append(str(_AGENT_ROOT))

from logging_utils import get_logger
from nodes.graph import run_graph

logger = get_logger(__name__)

app = FastAPI(
    title="SearchAgent API",
    version="1.0.0",
    description="Ask any research questions backed by Wikipedia and arXiv for reference.",
)


class ResearchRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=20000)


class ResearchResponse(BaseModel):
    status: str
    trace_id: str
    report: str
    sources: list[dict[str, Any]]
    errors: list[str]
    duration_ms: float


@app.post("/v1/research", response_model=ResearchResponse)
def research(request: ResearchRequest) -> ResearchResponse:
    started_at = time.perf_counter()
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt must not be empty.")

    logger.info("API research request start. prompt_len=%d", len(prompt))
    state = run_graph(prompt)
    duration_ms = (time.perf_counter() - started_at) * 1000

    report = str(state.get("report", ""))
    sources = state.get("sources", [])
    errors = state.get("errors", [])
    trace_id = str(state.get("trace_id", ""))

    status = "ok" if report else "error"
    logger.info(
        "API research request complete. status=%s trace_id=%s report_len=%d sources=%d errors=%d elapsed_ms=%.2f",
        status,
        trace_id,
        len(report),
        len(sources),
        len(errors),
        duration_ms,
    )

    return ResearchResponse(
        status=status,
        trace_id=trace_id,
        report=report,
        sources=sources,
        errors=errors,
        duration_ms=round(duration_ms, 2),
    )
