"""让一个 agent 真的能用工具：调一次 → 执行 → 把结果还回去 → 再调。

现在的 agent 都是一次问答：system + user 进，一段文本出。但「导演自己上网找材料」
不可能一次问答完成——它得先搜、看结果、发现不够再抓一页、最后才下结论。这中间
每一步都取决于上一步的结果，只能由模型自己决定，不能由代码写死顺序。

所以这里有三个约定：

1. **步数有限**。默认 4 步。工具是手段不是目的，一个判断用了四轮还没结论，多半
   是问题本身不清楚，再给十轮也不会更清楚，而用户已经在等。
2. **工具失败不中断**。某一页抓不下来、某个搜索结果为空，把「失败」当结果还回去，
   让模型自己换一条路——这正是它比代码写死流程强的地方。整条判断失败才中断。
3. **最后一步不给工具**。步数用完还在调工具，就关掉工具再问一次，逼它给结论；
   否则调用方拿到的是一串 tool_calls，等于什么都没判断。
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from ..tools import llm as llm_tools

#: 一次工具循环最多几轮模型调用。判断用不到两轮，留一倍余量给「没查到再换说法」。
DEFAULT_MAX_STEPS = 4

#: 单个工具结果回填的上限。一页正文可能上万字，全塞回对话会挤掉前面的工具调用
#: 和原始请求，模型反而忘了自己在判断什么。
MAX_TOOL_RESULT_CHARS = 8000

Executor = Callable[[str, dict[str, Any]], Awaitable[Any]]


async def run_tool_loop(
    config: Any,
    *,
    system: str,
    user: str,
    tools: list[dict[str, Any]],
    executor: Executor,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_tokens: int = 1600,
    on_step: Callable[[str, dict[str, Any], Any], None] | None = None,
) -> dict[str, Any]:
    """Run the loop and return `{"content", "steps", "tool_calls", "messages"}`.

    `content` is the model's final answer after tools are taken away — the caller
    parses that, not any intermediate turn.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    used: list[dict[str, Any]] = []

    for step in range(max(1, max_steps)):
        reply = await llm_tools.chat_messages(
            config, messages=messages, tools=tools, max_tokens=max_tokens
        )
        calls = reply.get("tool_calls") or []
        if not calls:
            return {
                "content": reply.get("content") or "",
                "steps": step + 1,
                "tool_calls": used,
                "messages": messages,
            }

        messages.append(
            {
                "role": "assistant",
                "content": reply.get("content") or "",
                "tool_calls": calls,
            }
        )
        for call in calls:
            name = str(((call.get("function") or {}).get("name")) or "")
            raw_args = (call.get("function") or {}).get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except (ValueError, TypeError):
                args = {}
            try:
                result = await executor(name, args)
                failed = False
            except Exception as exc:  # noqa: BLE001 - 失败也是结果，交给模型判断
                result = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                failed = True
            used.append({"name": name, "args": args, "failed": failed})
            if on_step is not None:
                try:
                    on_step(name, args, result)
                except Exception:  # noqa: BLE001 - 记录失败不能影响判断
                    pass
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id") or "",
                    "content": json.dumps(result, ensure_ascii=False)[:MAX_TOOL_RESULT_CHARS],
                }
            )

    # 步数用尽仍在调工具：关掉工具，逼一次结论。
    messages.append(
        {
            "role": "user",
            "content": "工具调用次数已用完。现在只根据已获得的信息给出结论，不要再调用工具。",
        }
    )
    reply = await llm_tools.chat_messages(
        config, messages=messages, tools=None, max_tokens=max_tokens
    )
    return {
        "content": reply.get("content") or "",
        "steps": max_steps + 1,
        "tool_calls": used,
        "messages": messages,
    }
