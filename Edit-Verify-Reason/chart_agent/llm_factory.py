from __future__ import annotations

import json
from dataclasses import dataclass
from collections import deque
import threading
import time
from typing import Any

from chart_agent.config import ModelConfig, get_model_config


@dataclass
class MockResponse:
    content: str


class MockLLM:
    def invoke(self, prompt: str | dict[str, Any]) -> MockResponse:
        payload = {
            "chart_type": "unknown",
            "confidence": 0.3,
            "parsed_data": {},
            "rationale": "MockLLM fallback response.",
        }
        return MockResponse(content=json.dumps(payload))


class _SlidingWindowRateLimiter:
    def __init__(self, *, rpm_limit: int | None, tpm_limit: int | None) -> None:
        self.rpm_limit = rpm_limit if isinstance(rpm_limit, int) and rpm_limit > 0 else None
        self.tpm_limit = tpm_limit if isinstance(tpm_limit, int) and tpm_limit > 0 else None
        self._request_times: deque[float] = deque()
        self._token_events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def acquire(self, estimated_tokens: int) -> None:
        token_cost = max(1, int(estimated_tokens))
        while True:
            wait_seconds = 0.0
            now = time.monotonic()
            with self._lock:
                self._prune(now)
                if self.rpm_limit is not None and len(self._request_times) >= self.rpm_limit:
                    wait_seconds = max(wait_seconds, 60.0 - (now - self._request_times[0]))
                if self.tpm_limit is not None:
                    used_tokens = sum(tokens for _, tokens in self._token_events)
                    if used_tokens + token_cost > self.tpm_limit and self._token_events:
                        wait_seconds = max(wait_seconds, 60.0 - (now - self._token_events[0][0]))
                if wait_seconds <= 0.0:
                    self._request_times.append(now)
                    self._token_events.append((now, token_cost))
                    return
            time.sleep(min(wait_seconds, 60.0))

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        while self._request_times and self._request_times[0] <= cutoff:
            self._request_times.popleft()
        while self._token_events and self._token_events[0][0] <= cutoff:
            self._token_events.popleft()


_RATE_LIMITERS: dict[str, _SlidingWindowRateLimiter] = {}
_RATE_LIMITERS_LOCK = threading.Lock()


class RateLimitedLLM:
    def __init__(
        self,
        inner: Any,
        *,
        model_config: ModelConfig,
        sleep_fn: Any | None = None,
        clock_fn: Any | None = None,
        limiter: _SlidingWindowRateLimiter | None = None,
    ) -> None:
        self._inner = inner
        self._config = model_config
        self._sleep_fn = sleep_fn or time.sleep
        self._clock_fn = clock_fn or time.monotonic
        self._limiter = limiter or _get_rate_limiter(model_config)

    def invoke(self, prompt: str | dict[str, Any]) -> Any:
        estimated_tokens = _estimate_prompt_tokens(prompt, self._config.tpm_limit)
        attempts = max(0, int(self._config.rate_limit_retries))
        for attempt in range(attempts + 1):
            self._limiter.acquire(estimated_tokens)
            try:
                return self._inner.invoke(prompt)
            except Exception as exc:
                if attempt >= attempts or not _is_rate_limit_error(exc):
                    raise
                backoff = min(60.0, 5.0 * (attempt + 1))
                self._sleep_fn(backoff)
        return self._inner.invoke(prompt)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _get_rate_limiter(model_config: ModelConfig) -> _SlidingWindowRateLimiter:
    key = f"{model_config.name}:{model_config.model}:{model_config.rpm_limit}:{model_config.tpm_limit}"
    with _RATE_LIMITERS_LOCK:
        limiter = _RATE_LIMITERS.get(key)
        if limiter is None:
            limiter = _SlidingWindowRateLimiter(
                rpm_limit=model_config.rpm_limit,
                tpm_limit=model_config.tpm_limit,
            )
            _RATE_LIMITERS[key] = limiter
        return limiter


def _estimate_prompt_tokens(prompt: Any, tpm_limit: int | None) -> int:
    text_chars = _estimate_text_chars(prompt)
    image_count = _count_prompt_images(prompt)
    estimated = max(1, (text_chars + 3) // 4)
    estimated += image_count * 4000
    if tpm_limit is not None and tpm_limit > 0:
        return min(max(1, estimated), tpm_limit)
    return estimated


def _estimate_text_chars(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(_estimate_text_chars(k) + _estimate_text_chars(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return sum(_estimate_text_chars(item) for item in value)
    return len(str(value))


def _count_prompt_images(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, dict):
        count = 0
        if "image_url" in value:
            count += 1
        for item in value.values():
            count += _count_prompt_images(item)
        return count
    if isinstance(value, (list, tuple)):
        return sum(_count_prompt_images(item) for item in value)
    return 0


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return any(
        token in text
        for token in (
            "rate limit",
            "too many requests",
            "429",
            "rpm",
            "tpm",
            "requests per minute",
            "tokens per minute",
        )
    )


def make_llm(model_config: ModelConfig | None = None) -> Any:
    config = model_config or get_model_config()
    if not config.api_key:
        return MockLLM()

    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "langchain_openai is required for OpenAI-compatible LLM usage."
        ) from exc

    llm = ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
    )
    if config.rpm_limit or config.tpm_limit:
        return RateLimitedLLM(llm, model_config=config)
    return llm
