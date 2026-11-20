"""Streaming chat completion, for progressive output.

Why this exists separately from `chat()`:

`chat()` waits for the whole response, so a stage that thinks for 40 seconds shows
nothing until it finishes. The requirement is to see the reasoning as it happens,
which means consuming the SSE stream and emitting deltas.

Two design points that matter:

  · **Reasoning has to be streamed too.** Models like glm-5 emit
    `reasoning_content` deltas before the answer. Showing only the final answer
    would reproduce the problem being solved.
  · **Streaming must not break JSON parsing.** Agents need an object, not prose.
    The stream is therefore accumulated into a buffer while being emitted; the
    caller parses the buffer once the stream ends, so a partially-streamed object
    is never parsed mid-way.

The delta callback is optional: a caller that only wants the final value passes
`on_delta=None` and gets the same behaviour as `chat()`.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

import httpx
from .provider_network import provider_trust_env


#: Receives (channel, text) where channel is "reasoning" or "content".
DeltaSink = Callable[[str, str], Awaitable[None]]


async def stream_chat(
    config: Any,
    *,
    system: str,
    user: str,
    images: list[str] | None = None,
    max_tokens: int = 2000,
    json_mode: bool = False,
    on_delta: DeltaSink | None = None,
    timeout: float | None = None,
    thinking: bool = True,
) -> Any:
    """One streaming chat completion, returning the same LLMResult as `chat()`.

    Raises `LLMError` on a transport or protocol failure, matching `chat()` so a
    caller can swap between them without changing its error handling.
    """
    from .llm import LLMError, LLMResult, _extract_error_message, _friendly_error, _strip_reasoning

    # Same reasoning as `chat()`: ask for the specific reason, because `configured`
    # passes a placeholder key and the generic message sends a UI-configured user
    # to an environment variable they never used.
    reason = config.missing_reason() if hasattr(config, "missing_reason") else ""
    if reason:
        raise LLMError(reason)
    if not config.configured:
        raise LLMError(
            "尚未配置 LLM：请在「设置 → 供应商与 Key」里填入 API Key 并保存"
            "（也可以设置环境变量 STUDIO_LLM_API_KEY）"
        )

    content: Any = user
    if images:
        parts: list[dict[str, Any]] = [{"type": "text", "text": user}]
        for url in images:
            parts.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
        content = parts

    model = config.effective_vision_model if images else config.model
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "temperature": config.temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    # Cap the reasoning phase here as well. `chat()` already does it, but
    # `agent_ask` prefers streaming, so the uncapped path was the one every agent
    # actually used — measured: glm-5 thought for 12079 characters and returned 2,
    # leaving the storyboard empty.
    from .llm import THINKING_BUDGET

    payload["thinking"] = (
        {
            "type": "enabled",
            "budget_tokens": max(256, min(THINKING_BUDGET, max_tokens // 3)),
        }
        if thinking
        else {"type": "disabled"}
    )

    url = config.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    last_error = ""
    #: The provider's stop reason from the last chunk that carried one. `length`
    #: means the reply was cut, which the caller must not treat as a whole answer.
    finish_reason = ""

    try:
        async with httpx.AsyncClient(timeout=timeout or config.timeout_seconds, trust_env=provider_trust_env(url)) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    body = ""
                    try:
                        body = await resp.aread()
                    except Exception:  # noqa: BLE001
                        pass
                    raise LLMError(
                        f"LLM 拒绝了请求（HTTP {resp.status_code}）："
                        f"{_extract_error_message(body.decode('utf-8', 'replace') if isinstance(body, bytes) else str(body)) or '未提供原因'}",
                        status=resp.status_code,
                        detail=(body.decode('utf-8', 'replace') if isinstance(body, bytes) else str(body))[:1000],
                        error_type="http_error",
                    )

                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk_raw = line[5:].strip()
                    if chunk_raw == "[DONE]":
                        break
                    try:
                        chunk = json.loads(chunk_raw)
                    except json.JSONDecodeError:
                        continue

                    # Some relays send an error event with no choices. It must
                    # not be discarded as a usage-only event or accepted as a
                    # successful partial answer.
                    if chunk.get("error"):
                        raise LLMError(
                            f"上游流式返回错误：{_extract_error_message(chunk_raw)}",
                            detail=chunk_raw[:1000], error_type="http_error",
                        )

                    choices = chunk.get("choices") or []
                    if not choices:
                        # Usage-only chunks arrive with no choices; ignore them.
                        continue
                    delta = choices[0].get("delta") or {}

                    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if reasoning:
                        reasoning_parts.append(reasoning)
                        if on_delta is not None:
                            await on_delta("reasoning", reasoning)

                    piece = delta.get("content") or ""
                    if piece:
                        text_parts.append(piece)
                        if on_delta is not None:
                            await on_delta("content", piece)

                    if choices[0].get("finish_reason"):
                        finish_reason = str(choices[0]["finish_reason"])
                        if chunk.get("error"):
                            last_error = str(chunk.get("error"))
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalised the same way as chat()
        raise _friendly_error(exc, config) from exc

    if last_error:
        raise LLMError(f"流式返回错误：{last_error[:160]}")

    text = "".join(text_parts)
    reasoning = "".join(reasoning_parts).strip()
    if not text and not reasoning:
        raise LLMError("流式返回为空（没有 content，也没有 reasoning_content）")

    return LLMResult(
        text=_strip_reasoning(text),
        raw={"streamed": True, "model": model},
        model=model,
        reasoning=reasoning,
        finish_reason=finish_reason,
    )
