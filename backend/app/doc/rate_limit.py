"""Retry a generation call when the provider rate-limits us.

The observed failure: 12 shots fired concurrently at SiliconFlow's free Kolors
endpoint. The first few succeed, then every remaining call returns 429:

    seg_1-sh07 … sh11   文生图被限流
    seg_2-sh01, sh06…sh09  文生图被限流

The shots are not bad and the prompts are not wrong — the free tier simply allows
a few images per minute. Failing them outright wastes the whole segment for a
condition that clears itself.

What this does:

  · waits and retries only on a rate-limit error (429 / "限流" / "rate limit")
  · waits long enough to matter — a free tier's window is a minute, so 20 s
    retries would all fail again
  · jitters the wait, so several concurrent shots do not retry in lockstep and
    immediately collide again
  · gives up after a bounded number of attempts, reporting the last error

A terminal error (empty balance) is never retried: waiting cannot create credit,
and the pipeline has to stop anyway.
"""

from __future__ import annotations

import asyncio
import random
from typing import Awaitable, TypeVar

T = TypeVar("T")

#: Error fragments meaning "too many requests right now". Matched against the
#: diagnosed message and the raw provider body alike.
RATE_LIMIT_MARKERS = (
    "429",
    "限流",
    "rate limit",
    "too many requests",
    "free limit",
    "免费额度",
)

#: Attempts in total, including the first. Three means two retries.
MAX_ATTEMPTS = 3

#: Base wait per attempt. A free tier's window is a minute, so 20 s is not enough
#: — the second attempt would land inside the same window and fail again.
BASE_WAIT = 45.0


def is_rate_limited(error_text: str) -> bool:
    """Whether this error is a rate limit rather than a real failure."""
    text = (error_text or "").lower()
    # Balance exhaustion also mentions 额度, so it is excluded explicitly: waiting
    # cannot fix it, and the terminal check has already handled it upstream.
    if any(k in text for k in ("余额不足", "账户余额", "insufficient balance", "请充值")):
        return False
    return any(marker in text for marker in RATE_LIMIT_MARKERS)


def wait_for(attempt: int) -> float:
    """How long to wait before attempt number `attempt + 1`.

    Grows linearly and jitters by up to 30 %, because several shots hit the limit
    at once and an identical delay would make them all retry together.
    """
    base = BASE_WAIT * (attempt + 1)
    return base + random.uniform(0, base * 0.3)


async def with_rate_limit_retry(
    call,
    *,
    label: str = "",
    on_wait=None,
    max_attempts: int = MAX_ATTEMPTS,
):
    """Run `call()`, retrying only when the failure is a rate limit.

    `on_wait(seconds, attempt, error)` is awaited before each retry, so a caller
    can record progress on the trace rather than appearing frozen.

    Returns whatever `call()` returns. Re-raises the last error when attempts run
    out, or immediately for a non-rate-limit failure.
    """
    last: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await call()
        except Exception as exc:  # noqa: BLE001 - the classification decides
            if not is_rate_limited(str(exc)):
                raise
            last = exc
            if attempt == max_attempts - 1:
                break
            delay = wait_for(attempt)
            if on_wait is not None:
                await on_wait(delay, attempt + 1, exc)
            else:
                await asyncio.sleep(delay)
    if last is not None:
        raise last
    raise RuntimeError(f"{label} 未执行")
