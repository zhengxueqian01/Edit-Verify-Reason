from __future__ import annotations

from collections import deque

from chart_agent.config import ModelConfig
from chart_agent.llm_factory import (
    RateLimitedLLM,
    _SlidingWindowRateLimiter,
    _estimate_prompt_tokens,
)


class _FakeInnerLLM:
    def __init__(self) -> None:
        self.prompts: list[object] = []
        self.failures_remaining = 0

    def invoke(self, prompt: object) -> dict[str, object]:
        self.prompts.append(prompt)
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise RuntimeError("429 rate limit exceeded")
        return {"ok": True}


class _TestLimiter(_SlidingWindowRateLimiter):
    def __init__(self, waits: deque[float]) -> None:
        super().__init__(rpm_limit=1, tpm_limit=100)
        self.waits = waits
        self.acquired: list[int] = []

    def acquire(self, estimated_tokens: int) -> None:
        self.acquired.append(estimated_tokens)
        if self.waits:
            self.waits.popleft()


def test_estimate_prompt_tokens_adds_image_budget() -> None:
    prompt = {
        "messages": [
            {"type": "text", "text": "hello world"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
    }

    estimated = _estimate_prompt_tokens(prompt, 15000)

    assert estimated >= 4002


def test_rate_limited_llm_acquires_before_invoke() -> None:
    inner = _FakeInnerLLM()
    limiter = _TestLimiter(deque())
    config = ModelConfig(
        name="qwen",
        model="test-qwen",
        api_key="key",
        base_url="https://example.com/v1",
        temperature=0.0,
        rpm_limit=1200,
        tpm_limit=15000,
        rate_limit_retries=0,
    )
    llm = RateLimitedLLM(inner, model_config=config, limiter=limiter)

    result = llm.invoke("short prompt")

    assert result == {"ok": True}
    assert len(inner.prompts) == 1
    assert len(limiter.acquired) == 1
    assert limiter.acquired[0] >= 1


def test_rate_limited_llm_retries_rate_limit_errors() -> None:
    inner = _FakeInnerLLM()
    inner.failures_remaining = 1
    limiter = _TestLimiter(deque())
    sleeps: list[float] = []
    config = ModelConfig(
        name="qwen",
        model="test-qwen",
        api_key="key",
        base_url="https://example.com/v1",
        temperature=0.0,
        rpm_limit=1200,
        tpm_limit=15000,
        rate_limit_retries=2,
    )
    llm = RateLimitedLLM(
        inner,
        model_config=config,
        limiter=limiter,
        sleep_fn=sleeps.append,
    )

    result = llm.invoke("prompt")

    assert result == {"ok": True}
    assert len(inner.prompts) == 2
    assert sleeps == [5.0]
