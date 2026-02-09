import os
from functools import lru_cache

@lru_cache(maxsize=1)
def get_llm():
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    api_key = os.getenv("GOOGLE_API_KEY")

    if provider == "gemini":
        from llm.gemini import GeminiLLM

        return GeminiLLM(model=model, api_key=api_key)

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
