"""One error contract for every external MCP/tool connector.

Connectors can have very different native errors.  The pipeline and UI should
not need provider-specific parsing to answer the user's five questions: what
failed, where, why, whether retrying is useful, and what to do next.
"""

from __future__ import annotations

from typing import Any


def classify_error(message: str, fallback: str = "external_tool_error") -> str:
    text = str(message or "").lower()
    if any(mark in text for mark in ("尚未配置", "not configured", "app id", "app secret")):
        return "not_configured"
    if any(mark in text for mark in ("尚未授权", "尚未连接", "not logged", "not connected")):
        return "not_connected"
    if any(mark in text for mark in ("授权已经过期", "token expire", "expired", "99991677")):
        return "auth_expired"
    if any(mark in text for mark in ("权限", "permission", "forbidden", "http 403", "99991672", "99991679")):
        return "permission_denied"
    if any(mark in text for mark in ("限流", "rate limit", "too many", "http 429")):
        return "rate_limited"
    if any(mark in text for mark in ("超时", "timeout", "timed out")):
        return "timeout"
    if any(mark in text for mark in ("无法连接", "connecterror", "connection", "dns", "代理")):
        return "connection_error"
    if any(mark in text for mark in ("http 404", "not found", "不存在")):
        return "not_found"
    if any(mark in text for mark in ("空文档", "空页面", "没有正文", "empty")):
        return "empty_content"
    if any(mark in text for mark in ("不支持", "unsupported")):
        return "unsupported_resource"
    return fallback


#: 这些错误说明「工具本身没跑通」，不是「查了没有」。界面上两者的处置完全不同：
#: 前者要让用户去看配置和额度，后者才是「换一个来源试试」。写在一处，是因为
#: 判定卡和问卡都要用同一个判断——各写一份，迟早一处说「没找到」一处说「坏了」。
SYSTEM_FAILURES = frozenset({
    "local_error", "tool_error", "external_tool_error", "tool_disabled",
    "balance", "unauthorized", "rate_limited", "auth_expired", "not_configured",
})

_DEFAULTS: dict[str, tuple[bool, str]] = {
    "tool_disabled": (False, "请先在“设置 → MCP 工具”中启用此服务"),
    "not_configured": (False, "请先在设置中完成此工具的应用配置"),
    "not_connected": (False, "请先连接并授权此工具"),
    "auth_expired": (False, "请重新授权后重试"),
    "permission_denied": (False, "请检查应用权限和当前账号对资源的访问权限"),
    "rate_limited": (True, "请稍后重试"),
    "timeout": (True, "请检查网络后重试"),
    "connection_error": (True, "请检查网络、代理或外部服务状态后重试"),
    "not_found": (False, "请更换关键词或直接提供资源链接"),
    "no_results": (False, "请更换关键词或直接提供资源链接"),
    "empty_content": (False, "请确认资源包含正文且当前账号有读取权限"),
    "unsupported_resource": (False, "请改用当前工具支持的资源类型"),
    "invalid_result": (True, "请重试；若仍失败，请检查外部工具版本"),
    "external_tool_error": (True, "请重试；若仍失败，请检查外部工具连接"),
    # 本地语义检索没跑通（没配向量化模型、额度用尽、网络失败）。它和
    # `no_results` 必须分开：「库里没有」只能换来源，「检索坏了」要去看配置和额度。
    "local_error": (True, "请检查向量化模型的配置与额度后重试"),
    # ---- 模型调用失败 ----
    #
    # 审核类刻意是 retryable=False。这不是保守，是实测结论：同一句话原样再发一次，
    # 结果一定是同一个 400。告诉用户可以「重试」等于把一件修不好的事说成能修好，
    # 而用户会在三次尝试之后得出「这个系统坏了」的结论。
    "moderation_in": (False, "请改写这一段描述（换一种说法）后重试，或换一个审核更宽松的模型"),
    "moderation_out": (False, "请改写出问题的这一段后重试，或换一个审核更宽松的模型"),
    "invalid_request": (False, "请检查提示词长度、图片格式与模型参数"),
    "auth": (False, "请在「设置 → 供应商与 Key」里检查 API Key 与模型权限"),
    "balance": (False, "请到供应商控制台充值，或换成免费模型"),
    "rate_limit": (True, "请稍后重试，或降低并发数"),
    "transport": (True, "请重试；反复发生请检查网络出口或改用流式输出"),
    "unreachable": (True, "请检查网络、代理与接口地址"),
    "http_error": (True, "请重试；若仍失败，请检查供应商状态页"),
    "bad_response": (True, "请重试；若仍失败，请更换模型"),
    "truncated": (True, "请重试，或把该供应商的最大输出长度调大"),
    "empty_reply": (True, "请重试；若反复为空，请关闭该模型的思考模式"),
    "unparseable": (True, "请重试，或换用输出更稳定的模型"),
}


def failure(
    *,
    provider: str,
    tool: str,
    stage: str,
    message: str,
    error_type: str = "",
    retryable: bool | None = None,
    user_action: str = "",
    impact: str = "已在生成前暂停，不会产生后续媒体生成费用",
    **extra: Any,
) -> dict[str, Any]:
    kind = error_type or classify_error(message)
    default_retry, default_action = _DEFAULTS.get(kind, _DEFAULTS["external_tool_error"])
    action = user_action or default_action
    return {
        "ok": False,
        "provider": provider,
        "tool": tool,
        "stage": stage,
        "error_type": kind,
        "message": str(message or "外部工具调用失败")[:1000],
        "retryable": default_retry if retryable is None else bool(retryable),
        "user_action": action[:500],
        "impact": impact[:500],
        # Compatibility aliases for existing traces and clients.
        "reason": str(message or "外部工具调用失败")[:1000],
        "action": action[:500],
        **extra,
    }


def normalize_failure(
    value: Any,
    *,
    provider: str,
    tool: str,
    stage: str,
    fallback_message: str,
    fallback_type: str = "external_tool_error",
    **extra: Any,
) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    message = str(raw.get("message") or raw.get("reason") or raw.get("error") or fallback_message)
    error_type = str(raw.get("error_type") or classify_error(message, fallback_type))
    return failure(
        provider=str(raw.get("provider") or provider),
        tool=str(raw.get("tool") or tool),
        stage=str(raw.get("stage") or stage),
        message=message,
        error_type=error_type,
        retryable=raw.get("retryable") if isinstance(raw.get("retryable"), bool) else None,
        user_action=str(raw.get("user_action") or raw.get("action") or ""),
        impact=str(raw.get("impact") or "已在生成前暂停，不会产生后续媒体生成费用"),
        **extra,
    )


def user_message(error: dict[str, Any]) -> str:
    provider = str(error.get("provider") or "外部工具")
    stage = str(error.get("stage") or "调用")
    message = str(error.get("message") or error.get("reason") or "调用失败")
    action = str(error.get("user_action") or error.get("action") or "")
    impact = str(error.get("impact") or "")
    retry = "可以重试" if error.get("retryable") else "需要处理后再试"
    return "\n".join(
        part for part in (
            f"{provider}在“{stage}”阶段失败：{message}",
            f"任务影响：{impact}" if impact else "",
            f"建议：{action}（{retry}）" if action else retry,
        ) if part
    )
