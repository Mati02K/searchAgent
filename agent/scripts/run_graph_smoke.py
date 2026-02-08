from __future__ import annotations

import argparse
import os
from pathlib import Path

from logging_utils import get_logger
from nodes.graph import run_graph


def _load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()
logger = get_logger(__name__)

EXIT_COMMANDS = {"exit"}


def _run_once(prompt: str, debug: bool) -> None:
    logger.info("Smoke run start. prompt_len=%d", len(prompt))
    state = run_graph(prompt)
    logger.info(
        "Smoke run complete. report_len=%d sources=%d errors=%d",
        len(state.get("report", "")),
        len(state.get("sources", [])),
        len(state.get("errors", [])),
    )
    print(state["report"])
    if debug:
        log_file = os.getenv("SEARCH_AGENT_LOG_FILE", "logs/search_agent.log")
        print("\n[debug] errors:", state.get("errors", []))
        print(
            f"[debug] log_file: {Path.cwd() / log_file if not Path(log_file).is_absolute() else Path(log_file)}"
        )

def main() -> None:
    parser = argparse.ArgumentParser(description="Run SearchAgent graph smoke test loop.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print errors and log file path for debugging.",
    )
    args = parser.parse_args()

    print("SearchAgent smoke loop started. Type 'exit' to quit.")
    while True:
        try:
            prompt = input("\nPrompt> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting smoke loop.")
            break

        if not prompt:
            print("Prompt is empty. Enter a prompt or type 'exit'.")
            continue
        if prompt.lower() in EXIT_COMMANDS:
            print("Exiting smoke loop.")
            break

        _run_once(prompt, args.debug)


if __name__ == "__main__":
    main()
