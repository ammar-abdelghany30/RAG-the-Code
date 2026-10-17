from langchain_openai import ChatOpenAI

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME


def get_llm(temperature: float = 0.0, streaming: bool = False) -> ChatOpenAI:
    """One place that builds the OpenRouter-backed chat model."""
    if not LLM_API_KEY or not LLM_MODEL_NAME:
        raise RuntimeError("Set LLM_API_KEY and LLM_MODEL_NAME in your .env file.")
    return ChatOpenAI(
        model=LLM_MODEL_NAME,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=temperature,
        streaming=streaming,
    )