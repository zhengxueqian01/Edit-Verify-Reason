from __future__ import annotations

from typing import Any

from chart_agent.config import ModelConfig


def build_llm(model_config: ModelConfig) -> Any:
    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "langchain_openai is required for the OpenAI-compatible LLM provider."
        ) from exc

    return ChatOpenAI(
        model=model_config.model,
        api_key=model_config.api_key,
        base_url=model_config.base_url,
        temperature=model_config.temperature,
    )
