from __future__ import annotations

from collections.abc import Sequence


SAFETY_BLOCK_REPORT = "## Unable to Answer\n\nI can't answer this question."


FINAL_SYNTHESIS_SYSTEM_PROMPT = """
You are the final synthesis agent for a research system.

You must produce a concise markdown report with citations and no fabricated facts.

Critical constraints:
- Use only the provided source documents.
- Do not add outside knowledge.
- If evidence is missing, state that explicitly.
- Follow the exact section order provided by the caller.
- Cite links in findings.
- In the final `## Citations` section, list all provided source URLs even if minimally used.

Output must be markdown only.
""".strip()


def build_final_synthesis_prompt(
    prompt: str,
    sections: Sequence[str],
    sources: Sequence[dict],
) -> str:
    section_template = "\n".join(f"- {section.title()}" for section in sections) or "- Definition"
    source_lines = "\n".join(
        f"- {item.get('title', '')} | {item.get('source', '')} | {item.get('url', '')} | {item.get('summary', '')}"
        for item in sources[:10]
    ) or "- No sources retrieved."

    return (
        f"User question:\n{(prompt or '').strip()}\n\n"
        "Write the answer using these sections:\n"
        f"{section_template}\n\n"
        "Only use the provided sources. If evidence is missing, say so explicitly.\n\n"
        "Citation policy: In `## Citations`, list every provided source URL, "
        "even if a source contributed minimally.\n\n"
        f"Sources:\n{source_lines}\n\n"
        "Format requirements:\n"
        "# Research Report\n"
        "## Problem Framing\n"
        "## <Section 1>\n"
        "## <Section 2>\n"
        "...\n"
        "## Citations\n"
    )
