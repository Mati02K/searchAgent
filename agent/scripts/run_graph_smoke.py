from __future__ import annotations

import argparse
import os
from pathlib import Path

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


DEFAULT_PROMPT = (
    "Assess the risks and benefits of using synthetic data in healthcare ML systems."
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run SearchAgent graph smoke test.")
    parser.add_argument("prompt", nargs="*", help="Optional prompt text.")
    args = parser.parse_args()

    prompt = " ".join(args.prompt).strip() if args.prompt else DEFAULT_PROMPT
    state = run_graph(prompt)
    print(state["report"])


if __name__ == "__main__":
    main()
