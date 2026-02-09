from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

API_URL = os.getenv("SEARCH_AGENT_API_URL", "http://localhost:8000/v1/research")
EXIT_COMMANDS = {"exit", "quit"}


def call_research_api(prompt: str) -> dict:
    payload = {"prompt": prompt}
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        body = response.read().decode("utf-8", errors="replace")
    data = json.loads(body) if body else {}
    return data if isinstance(data, dict) else {}


def main() -> None:
    print(f"Test client started. endpoint={API_URL}")
    print("Type 'exit' to quit.")
    while True:
        try:
            prompt = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting test client.")
            break

        if not prompt:
            print("Prompt is empty.")
            continue
        if prompt.lower() in EXIT_COMMANDS:
            print("Exiting test client.")
            break

        try:
            result = call_research_api(prompt)
            report = str(result.get("report", ""))
            errors = result.get("errors", [])
            print(report or "[no report]")
            if errors:
                print("\n[errors]", errors)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"HTTP error: status={exc.code} body={detail}")
        except Exception as exc:
            print(f"Request failed: {exc}")


if __name__ == "__main__":
    sys.exit(main())
