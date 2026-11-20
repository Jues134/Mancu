"""把一次模型调用失败翻成项目统一的错误契约。

`tool_feedback.failure()` 已经把「任务影响 / 下一步 / 能不能重试」定好了，MCP 工具
和检索都在用它。模型调用一直绕开这套契约，直接把自己的 `message` 打到屏幕上——
于是同一个界面里，飞书失败能说清下一步做什么，模型失败只有一句「HTTP 400」。

这个模块只做翻译，不做判断：分类由 `LLMError.error_type` 给出，处置由
`tool_feedback` 的默认表给出。两处都不在这里重写，否则又会出现「同一个问题两套说法」。
"""

from __future__ import annotations

from typing import Any

from .tool_feedback import failure, user_message

#: 失败发生在某一个分段里，所以「任务影响」说的都是**这一段**，不是整部片子。
#:
#: 以前审核类写的是「这一段已跳过」——现在不跳过了。跳过让界面看起来什么都没缺，
#: 而实际交付的是一段没做过那项校验的片；说成「没完成」至少和事实一致。
_IMPACT_DEFAULT = "这一段没有完成，其余分段继续；已完成的阶段保留，不需要从头再跑"
_IMPACT_MODERATION = (
    "这一段没有完成，其余分段继续；已完成的剧本和分镜保留，"
    "改掉这处描述后可以只重跑这一段"
)


def failure_from(
    exc: BaseException,
    *,
    stage: str,
    model: str = "",
    provider: str = "模型服务",
    impact: str = "",
) -> dict[str, Any]:
    """一次 LLMError → 错误契约。`stage` 是给人看的中文阶段名（如「一致性」）。"""
    error_type = str(getattr(exc, "error_type", "") or "")
    if not error_type:
        # 老调用点可能抛的是裸 RuntimeError："内容审核" 这类短语仍能认出来。
        from ..tools.llm import _classify_message  # noqa: PLC2701 - 分类只此一份

        error_type = _classify_message(str(exc))

    is_moderation = error_type in ("moderation_in", "moderation_out")
    detail = str(getattr(exc, "detail", "") or "")
    message = str(exc) or "模型调用失败"

    return failure(
        provider=provider,
        tool=f"chat:{model}" if model else "chat",
        stage=stage,
        message=message,
        error_type=error_type,
        impact=impact or (_IMPACT_MODERATION if is_moderation else _IMPACT_DEFAULT),
        # 审核拦下的原因在供应商返回体里，界面上折叠显示：整段报错对用户是噪音，
        # 但排查时必须能看到供应商原话。
        detail=detail[:500],
    )


def message_for(
    exc: BaseException, *, stage: str, model: str = "", provider: str = "模型服务"
) -> tuple[dict[str, Any], str]:
    """(错误契约, 三段式文案)。"""
    error = failure_from(exc, stage=stage, model=model, provider=provider)
    return error, user_message(error)
