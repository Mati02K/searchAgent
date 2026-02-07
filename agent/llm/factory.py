import os
from pathlib import Path


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


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    api_key = os.getenv("GOOGLE_API_KEY")

    if provider == "gemini":
        from llm.gemini import GeminiLLM

        return GeminiLLM(model=model, api_key=api_key)

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
