from __future__ import annotations

from nodes.prompt import SEARCH_SYSTEM_PROMPT, build_search_user_prompt
from nodes.state import AgentState


def search_node(state: AgentState) -> dict:
    """Synthesize final markdown report from planner-produced sources."""
    errors = list(state.get("errors", []))
    try:
        unique_sources: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for item in state.get("sources", []):
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("source", "")),
                str(item.get("url", "")),
                str(item.get("title", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            unique_sources.append(item)

        evidence = [
            {
                "statement": item.get("summary", ""),
                "url": item.get("url", ""),
                "source": item.get("source", ""),
            }
            for item in unique_sources
        ]

        report = ""
        try:
            from llm.factory import get_llm

            llm = get_llm()
            prompt = state.get("prompt", "")
            llm_prompt = build_search_user_prompt(
                prompt=prompt,
                plan=state.get("plan", []),
                sources=unique_sources,
            )
            print(f"Search synthesis LLM prompt: {llm_prompt}")
            response = llm.generate(llm_prompt, system=SEARCH_SYSTEM_PROMPT)
            report = (response.text or "").strip()
            if not report:
                errors.append("Search synthesis LLM returned empty output.")
        except Exception as exc:
            errors.append(f"Search synthesis LLM error: {exc}")

        return {
            "sources": unique_sources,
            "evidence": evidence,
            "report": report,
            "errors": errors,
        }
    except Exception as exc:
        errors.append(f"Search node failure: {exc}")
        return {
            "sources": state.get("sources", []),
            "evidence": state.get("evidence", []),
            "report": "",
            "errors": errors,
        }
