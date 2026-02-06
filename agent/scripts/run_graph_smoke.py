from __future__ import annotations

import argparse

from nodes.planner_agent.graph import run_planner_agent


DEFAULT_PROMPT = (
    "Assess the risks and benefits of using synthetic data in healthcare ML systems."
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run SearchAgent planner-agent smoke test.")
    parser.add_argument("prompt", nargs="*", help="Optional prompt text.")
    args = parser.parse_args()

    prompt = " ".join(args.prompt).strip() if args.prompt else DEFAULT_PROMPT
    state = run_planner_agent(prompt)
    print(state["report"])


if __name__ == "__main__":
    main()
