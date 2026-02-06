import os


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    model = os.getenv("LLM_MODEL", "gemini-2.0-flash")

    if provider == "gemini":
        from llm.gemini import GeminiLLM

        return GeminiLLM(model=model)

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
