"""LLM client built on httpx.

Why not the `openai` SDK: every endpoint we target (DeepSeek, Qwen, Atlas,
Ollama, OpenAI) speaks the same `/chat/completions` JSON contract, so one small
httpx client covers all of them while keeping the dependency surface small.

The client adds the three things a bare HTTP call lacks:
  1. JSON-mode extraction with a tolerant parser (models wrap JSON in prose
     despite instructions, and reasoning models emit <think> blocks).
  2. One retry when the reply is unparseable — most failures are transient
     formatting, not a broken prompt.
  3. Bilingual, actionable error messages (a bare 401 sends people hunting).
"""

from __future__ import annotations

import asyncio
import json
import contextvars
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from .provider_network import provider_trust_env

from .. import logs
from ..config import LLMConfig

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised with a human-actionable message; routes surface it verbatim.

    `error_type` 是一台机器可读的分类，`message` 是给人看的一句话。两者都要：
    一个数字段同时失败时，界面要能把「审核拦了」和「超时了」分成两种提示，
    而按中文句子做字符串匹配来决定这件事，第一次改文案就会失效。
    取值见模块末尾的 `ERROR_TYPES`。
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        detail: str = "",
        error_type: str = "",
    ) -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail
        self.error_type = error_type

    @property
    def kind(self) -> str:
        """分类，缺省时从文案里猜一个，便于老调用点无需逐个补参数。"""
        return self.error_type or _classify_message(str(self))


@dataclass
class LLMResult:
    text: str
    raw: dict[str, Any]
    model: str
    #: The model's own planning text, when the provider exposes it. Empty for
    #: models without a reasoning mode.
    reasoning: str = ""
    #: Why generation stopped: "stop", "length", "content_filter", …
    #:
    #: Carried because `length` is the difference between "the model finished" and
    #: "the model was cut off mid-sentence". Without it a truncated JSON reply was
    #: handed to the repair step, which closed the braces and returned a smaller
    #: object — the stage then ran on data the model never produced, and the loss
    #: only surfaced as an unrelated finding two stages later.
    finish_reason: str = ""
    #: 这次回答为了让审核放行做过什么：「as_is」原样、「neutralize」换了命中的
    #: 短语、「soften」还在整句层面降了调。空字符串表示没走这条路径。
    #:
    #: 带出去是因为成片会被悄悄改变调子：镜头描述里「民国时期的旗帜」是系统改的，
    #: 不是模型本来要写的，用户有权知道，否则他看到的画面和他说的话对不上，
    #: 而界面上没有任何地方提示过这件事。
    moderation_plan: str = ""
    #: 被替换掉的短语原文，用于在界面上指名道姓地说"这三个词被改写了"。
    moderation_terms: list[str] = field(default_factory=list)

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


#: How many tokens the reasoning phase may consume before the model
#: must start answering. Without a cap, a reasoning model spends the
#: entire budget thinking and returns an empty body.
THINKING_BUDGET = 1200

#: How many times a 429 is retried before the stage gives up. Three attempts with
#: the delays below span roughly 10 s — about how long a per-key concurrency limit
#: takes to clear. More would only delay an error that is going to happen anyway.
RATE_LIMIT_ATTEMPTS = 3

#: Transport failures that are worth one more attempt.
#:
#: A storyboard call reasons for 100+ seconds before the first byte arrives, and a
#: connection can be dropped in that window by a router, a corporate gateway or a
#: TLS terminator reclaiming an "idle" socket. That is indistinguishable, at the
#: call site, from "the provider is down" — but it is transient: the identical
#: request succeeded a minute later in a measured run. Without this, one
#: interrupted stream ended the whole storyboard stage, the segment was retried,
#: and the run reported "无法连接 LLM 服务" as if the address were wrong.
#:
#: `ReadTimeout` is deliberately absent: a 420-second wait that already timed out
#: would simply be repeated, doubling how long a failing run stays silent before
#: it tells the user anything.
TRANSIENT_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
)


def _network_retry_delay(attempt: int) -> float:
    """Short backoff between transport retries.

    Kept to a few seconds: the failure is usually a single dropped socket, and the
    retry re-sends a request the provider may already have billed, so waiting
    longer buys nothing.
    """
    return min(8.0, 2.0 * (attempt + 1)) + random.uniform(0, 0.5)

#: Shown when the provider's quota is exhausted. The wording deliberately repeats
#: the provider's own phrase (余额不足 / 请充值): `free_models.is_terminal()` scans
#: for those markers to stop the run, and a message that only *describes* the
#: problem in our own words would be retried forever.
_BALANCE_MESSAGE = (
    "账户余额或额度不足：供应商返回「余额不足或无可用资源包，请充值」。"
    "请到控制台充值，或在设置里换成仍可用的模型（免费档如 glm-4-flash 不消耗资源包）"
)

#: Providers signal an empty balance with 429 as well as 402/payment errors, so the
#: status code alone cannot tell "too fast" from "no money". Zhipu uses code 1113.
_BALANCE_MARKERS = (
    "余额不足",
    "额度不足",
    "无可用资源包",
    "请充值",
    "insufficient balance",
    "insufficient_quota",
    "payment required",
)


def _is_balance_error(body: str) -> bool:
    """Whether a provider error body means "out of quota" rather than "too fast"."""
    text = (body or "").lower()
    compact = text.replace(" ", "")
    if '"code":"1113"' in compact or '"code":1113' in compact:
        return True
    return any(marker.lower() in text for marker in _BALANCE_MARKERS)


#: 国内模型的内容审核会直接 400 的表述，以及它们的替代表述。
#:
#: 实测（智谱 glm-4.5，2026-11）：「青天白日旗」「青天白日满地红旗」必定返回
#: 400「系统检测到输入或生成内容可能包含不安全或敏感内容」，而同一条提示词里的
#: 「国民党」「蒋介石」「中华民国国旗」「反革命政变」都能正常通过。也就是说这是
#: 审核词表命中，不是这段历史本身不能写。
#:
#: 处理方式是**换说法，不是删内容**：「会场悬挂青天白日满地红旗」改成「会场悬挂
#: 民国时期的旗帜」，画面里仍然有一面旗，拍摄指令的字面信息没有丢，只是不再
#: 撞审核词表。替换只在真的被拒之后才做，所以对不触发审核的供应商（DeepSeek、
#: Qwen、本地模型）提示词保持原样。
_MODERATION_TERMS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"青天白日满地红[旗]?"), "民国时期的旗帜"),
    (re.compile(r"青天白日[旗]"), "民国时期的旗帜"),
    (re.compile(r"青天白日(徽|勋章|国徽)"), "民国时期的徽记"),
    (re.compile(r"青天白日"), "民国时期的标志"),
)

#: 供应商在 400 里表示"内容被拦"的说法。命中这些词才值得重发中性化版本；
#: 参数错误（比如不支持的字段）同样返回 400，重发一次只是白白再失败一次。
_MODERATION_MARKERS = (
    "敏感", "不安全", "不合规", "违规",
    "content_filter", "content filter", "safety", "sensitive", "risk",
)

#: 词表可扩充，但**默认值只留在这里**。写进配置文件当默认值会让同一份词表出现
#: 两个来源，改了其中一份而另一份没改时，被拦的词会忽而能过忽而不能过。
#: 配置文件只做「追加/覆盖」：`STUDIO_MODERATION_TERMS_FILE` 指向一个
#: `{"短语": "替代说法"}` 的 JSON，重启后生效。
_EXTRA_TERMS: tuple[tuple[re.Pattern[str], str], ...] | None = None


def _extra_terms() -> tuple[tuple[re.Pattern[str], str], ...]:
    global _EXTRA_TERMS
    if _EXTRA_TERMS is not None:
        return _EXTRA_TERMS

    import os

    path = (os.environ.get("STUDIO_MODERATION_TERMS_FILE") or "").strip()
    pairs: list[tuple[re.Pattern[str], str]] = []
    if path:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            table = data.get("terms") if isinstance(data, dict) else data
            if isinstance(table, dict):
                # 长的放前面：先替换「青天白日满地红旗」，否则它会被更短的
                # 「青天白日」先吃掉一半，剩下的「满地红旗」留在句子里。
                for phrase, replacement in sorted(
                    table.items(), key=lambda kv: len(str(kv[0])), reverse=True
                ):
                    if str(phrase).strip():
                        pairs.append((re.compile(re.escape(str(phrase))), str(replacement)))
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            logger.warning("审核词表配置读不出来，已忽略（%s）：%s", path, exc)
    _EXTRA_TERMS = tuple(pairs)
    return _EXTRA_TERMS


def _terms() -> tuple[tuple[re.Pattern[str], str], ...]:
    """生效的词表：配置文件在前，内置的在后面兜底。"""
    return _extra_terms() + _MODERATION_TERMS


def _moderation_hits(*texts: str) -> list[str]:
    """提示词里会被审核拦下的字面，用于日志和重发时的说明。

    短的命中会被长的包含（「青天白日满地红旗」里也有「青天白日」），日志里
    只留最长的那一个：报「替换了 A、B、C 三个词」而实际是同一条短语，读起来
    像改了三处，会让人以为提示词被动了大手术。
    """
    hits: list[str] = []
    for text in texts:
        for pattern, _ in _terms():
            for match in pattern.finditer(text or ""):
                hits.append(match.group(0))
    longest: list[str] = []
    for hit in sorted(set(hits), key=len, reverse=True):
        if not any(hit in kept for kept in longest):
            longest.append(hit)
    return longest


def _neutralize(text: str) -> str:
    out = text or ""
    for pattern, replacement in _terms():
        out = pattern.sub(replacement, out)
    return out


def _is_moderation_rejection(exc: Exception) -> bool:
    """400 + 审核措辞。缺任一条都按普通错误处理。"""
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    if exc.response.status_code != 400:
        return False
    try:
        body = (exc.response.text or "").lower()
    except Exception:  # noqa: BLE001 - body already consumed / not readable
        return False
    return any(marker in body for marker in _MODERATION_MARKERS)


#: The machine-readable vocabulary every LLM failure is filed under.
#:
#: 界面按这套分类决定「能不能重试、下一步做什么」，所以它是契约而不是日志标签：
#: 加一个值就要同时在 `platform/tool_feedback.py` 里给默认处置。
ERROR_TYPES = {
    "not_configured": "尚未配置模型",
    "moderation_in": "提示词被内容审核拦截",
    "moderation_out": "生成的内容被审核拦截",
    "invalid_request": "请求参数被拒",
    "auth": "API Key 无效或无权限",
    "not_found": "模型或接口地址不存在",
    "rate_limit": "被限流",
    "balance": "余额或额度不足",
    "timeout": "请求超时",
    "transport": "连接中途被断开",
    "unreachable": "无法连接服务",
    "http_error": "接口返回其他错误",
    "bad_response": "返回结构异常",
    "truncated": "输出被截断",
    "empty_reply": "模型没有作答",
    "unparseable": "输出无法解析",
}

#: 出向审核的 finish_reason。HTTP 200，正文为空或半截——模型自己把答案撤回了。
_CONTENT_FILTER = "content_filter"

#: 出向审核的第二级手段：词表里没有对应短语时，只能要求改写表述本身。
#:
#: 「换掉命中的那个词」只对已知词有效。出向拦截往往是模型自己写出了那个词，
#: 提示词里根本没有它——这一级是唯一还能加力的地方，因此必须保留拍摄要素，
#: 否则「降调」会变成「把这段拍的东西删掉」。
_SOFTEN_INSTRUCTION = (
    "\n\n【表述要求】同一段内容可以有多种说法。请用中性、书面化的方式描述，"
    "避开可能触发内容审核的表述；但必须保留人物、时间、地点、动作与镜头安排，"
    "不要删减情节，也不要省略镜头。"
)

#: 提示词改写可用到的强度，按顺序尝试。原样先发一次：大多数任务根本不会触发
#: 审核，无差别地先降调会让所有成片都退化成一种四平八稳的腔调。
_MODERATION_PLANS = ("as_is", "neutralize", "soften")


def _moderation_plan(hits: list[str]) -> tuple[str, ...]:
    """按命中情况裁剪强度表。

    没命中已知词时，「只换词」和「什么都没改」是同一件事，跳过它——一次多余的
    一模一样的重发既浪费时间，也会让日志里出现两条相同错误。
    """
    if hits:
        return _MODERATION_PLANS
    return ("as_is", "soften")


def _apply_plan(plan: str, system: str, user: str) -> tuple[str, str]:
    if plan == "as_is":
        return system, user
    text_system, text_user = _neutralize(system), _neutralize(user)
    if plan == "soften":
        text_system = text_system + _SOFTEN_INSTRUCTION
    return text_system, text_user


def _finish_reason(data: Any) -> str:
    """read finish_reason defensively — a missing choices list must not mask a 200."""
    try:
        choice = (data or {}).get("choices")[0]
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""
    return str(choice.get("finish_reason") or "") if isinstance(choice, dict) else ""


def _classify_message(message: str) -> str:
    """Best-effort classification for an LLMError raised without a type.

    老代码里还有些 `raise LLMError("…")` 没带 error_type。与其逐个去补（漏一个就
    落回默认处置），不如从文案反推一次——文案本身就在这个文件里，改文案时测试会
    先把这里撞出来。
    """
    text = str(message or "")
    if "内容审核" in text or ("敏感" in text and "400" in text):
        return "moderation_out" if "finish_reason" in text else "moderation_in"
    if "余额" in text or "额度" in text:
        return "balance"
    if "限流" in text:
        return "rate_limit"
    if "超时" in text:
        return "timeout"
    if "API Key" in text or "无权限" in text:
        return "auth"
    if "不存在" in text:
        return "not_found"
    if "尚未配置" in text:
        return "not_configured"
    if "无法连接" in text:
        return "unreachable"
    if "中断" in text:
        return "transport"
    if "无法解析" in text or "合法 JSON" in text:
        return "unparseable"
    if "截断" in text:
        return "truncated"
    if "结构异常" in text:
        return "bad_response"
    return "http_error"


def _retry_after(resp: "httpx.Response", attempt: int) -> float:
    """Delay before retrying a rate-limited call.

    `Retry-After` wins when the provider sends it; otherwise exponential backoff
    (2 s, then 4 s), capped so a stuck run does not look frozen. The jitter keeps
    two concurrent segments from retrying in lockstep and tripping the same limit
    again — which is exactly how the first 429 turned into two failed stages.
    """
    raw = (resp.headers.get("Retry-After") or "").strip()
    if raw:
        try:
            return max(1.0, min(30.0, float(raw)))
        except ValueError:
            pass
    return min(20.0, 2.0 * (2**attempt)) + random.uniform(0, 0.75)


def _strip_reasoning(text: str) -> str:
    """Remove reasoning traces that would otherwise break JSON parsing."""
    text = re.sub(r"<think[^>]*>[\s\S]*?<｜end▁of▁thinking｜>", "", text, flags=re.IGNORECASE)
    return text.strip()


def looks_truncated(text: str) -> bool:
    """Whether a reply stops before its JSON closes.

    `finish_reason` is the reliable signal, but not every provider sends it, and the
    repair step is happy to close the braces for us: `{"characters": [` became
    `{"characters": []}` — a plausible-looking empty list that no consumer can tell
    apart from a real answer. Checking the last character catches that class without
    guessing at semantics.

    A trailing newline or code fence is stripped first, so a normal reply is not
    mistaken for a cut one; only a body ending mid-token is flagged.
    """
    cleaned = _strip_reasoning(text).strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, flags=re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    return bool(cleaned) and cleaned[-1] not in "}]"


def extract_json(text: str) -> Any:
    """Pull the first complete JSON object/array out of a model reply.

    Tolerant by design: models routinely prepend "Here is the JSON:" or wrap
    output in ```json fences even when told not to.
    """
    cleaned = _strip_reasoning(text)

    # Strip code fences first — they are the most common wrapper.
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, flags=re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fall back to brace matching, respecting strings and escapes.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        escape = False
        for idx in range(start, len(cleaned)):
            ch = cleaned[idx]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start : idx + 1])
                    except json.JSONDecodeError:
                        break

    # Last resort: mechanical repair. The observed defects are all cheap to fix
    # (trailing commas, smart quotes, a truncated tail), and repairing is much
    # better than a second billed call plus a failed stage. The repair never
    # invents content — a cut-off array is closed, not completed.
    try:
        from .json_repair import repair

        fixed = repair(cleaned)
        if fixed is not None:
            value, _note = fixed
            return value
    except Exception:  # noqa: BLE001 - repair must never mask the real error
        pass

    raise LLMError(
        "模型返回的内容不是合法 JSON，无法解析",
        detail=cleaned[:400],
        error_type="unparseable",
    )


def _extract_error_message(body: str) -> str:
    """Pull the human-readable reason out of a provider error body.

    Providers disagree on shape: OpenAI-style `{"error":{"message":...}}`,
    Zhipu's `{"error":{"code":...,"message":...}}`, and plain text. Returning the
    first non-empty string found means a 400 report always carries a reason
    instead of only a status code.
    """
    import json

    text = (body or "").strip()
    if not text:
        return ""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text[:200]

    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            for key in ("message", "msg", "detail"):
                if err.get(key):
                    return str(err[key])[:200]
            if err.get("code"):
                return f"code {err['code']}"
        for key in ("message", "msg", "detail"):
            if data.get(key):
                return str(data[key])[:200]
    return text[:200]

def _friendly_error(exc: Exception, config: LLMConfig) -> LLMError:
    # A classified error is already the answer — re-wrapping it would bury the
    # message the caller (and `is_terminal`) matches on under "LLM 调用失败".
    if isinstance(exc, LLMError):
        return exc
    if isinstance(exc, httpx.TimeoutException):
        return LLMError(
            f"LLM 请求超时（{config.base_url}）：{config.timeout_seconds:.0f} 秒内没有等到响应。"
            "分镜这类一万字以上的长输出，思考型模型单步常需 5–10 分钟；"
            "在「设置 → 模型供应商」里把该供应商的超时调大（不只是 STUDIO_LLM_TIMEOUT）",
            detail=str(exc),
            error_type="timeout",
        )
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        body = ""
        try:
            body = exc.response.text[:300]
        except Exception:  # noqa: BLE001 - best effort
            pass
        if status in (401, 403):
            return LLMError(
                f"API Key 无效或无权限：请检查 STUDIO_LLM_API_KEY 与模型「{config.model}」是否可用",
                status=status,
                detail=body,
                error_type="auth",
            )
        if status == 404:
            return LLMError(
                f"模型或接口地址不存在：请确认 STUDIO_LLM_BASE_URL（当前 {config.base_url}）"
                f"与 STUDIO_LLM_MODEL（当前 {config.model}）",
                status=status,
                detail=body,
                error_type="not_found",
            )
        if status == 429:
            # Zhipu answers 429 for a drained balance too (code 1113), not only for
            # concurrency. Calling that "请求过于频繁" sends the user to reduce
            # parallelism for a problem only a recharge fixes.
            if _is_balance_error(body):
                return LLMError(_BALANCE_MESSAGE, status=status, detail=body, error_type="balance")
            return LLMError(
                "触发限流：请求过于频繁，请稍后重试或降低并发",
                status=status, detail=body, error_type="rate_limit",
            )
        # Surface the provider's own message for 4xx-other, not just the code.
        # A bare "HTTP 400" is undiagnosable: the response body is the only place
        # that says *which* field or content was rejected, and dropping it meant a
        # failure could only be investigated by replaying the request by hand.
        provider_msg = _extract_error_message(body)
        if status == 400:
            # 审核拦截和参数错误都回 400，处置却相反：前者换说法有用，后者换说法
            # 只是把同一个错误再发一次。分类在这里定，调用方不必解析中文。
            rejected = any(marker in (body or "").lower() for marker in _MODERATION_MARKERS)
            return LLMError(
                f"LLM 拒绝了请求（HTTP 400）：{provider_msg or '未提供原因'}。"
                "常见原因：内容安全策略拦截、参数不合法、或模型不支持该输入形式",
                status=status,
                detail=body,
                error_type="moderation_in" if rejected else "invalid_request",
            )
        return LLMError(
            f"LLM 接口返回 HTTP {status}" + (f"：{provider_msg}" if provider_msg else ""),
            status=status,
            detail=body,
            error_type="http_error",
        )
    if isinstance(exc, httpx.RequestError):
        # One blanket sentence used to cover every transport failure, so a
        # connection dropped mid-response read exactly like a wrong `base_url`
        # and sent people to check an address that was correct. The exception
        # type is the only place the distinction lives, so it is spelled out.
        detail = str(exc) or exc.__class__.__name__
        if isinstance(exc, (httpx.ReadError, httpx.RemoteProtocolError, httpx.WriteError)):
            return LLMError(
                f"与 LLM 服务的连接在响应中途被中断（{config.base_url}）：已经连上并发出请求，"
                "但对端或中间设备（路由器、公司网关、安全设备）提前断开了连接。"
                "这类中断通常是瞬时抖动，重试常常就能成功；若反复发生，"
                "请在「设置 → 调用方式」打开流式输出，让连接持续有数据流动，"
                "或换一个网络出口",
                detail=f"{exc.__class__.__name__}: {detail}",
                error_type="transport",
            )
        if isinstance(exc, (httpx.ConnectTimeout, httpx.ConnectError)):
            return LLMError(
                f"无法连接 LLM 服务（{config.base_url}）：{exc.__class__.__name__}，"
                "地址不可达或被拦截。请检查网络、代理与地址拼写",
                detail=f"{exc.__class__.__name__}: {detail}",
                error_type="unreachable",
            )
        return LLMError(
            f"无法连接 LLM 服务（{config.base_url}）：{exc.__class__.__name__}",
            detail=f"{exc.__class__.__name__}: {detail}",
            error_type="unreachable",
        )
    return LLMError(f"LLM 调用失败：{exc}", detail=str(exc), error_type="http_error")


async def _post_chat(
    url: str, payload: dict[str, Any], headers: dict[str, str], config: LLMConfig
) -> dict[str, Any]:
    """发出一次请求，处理传输重试与 429，返回解析好的响应体。

    从 `chat()` 里抽出来，是因为内容审核被拒之后需要用换过措辞的提示词
    重发一次——把重试循环复制一份是最容易让两版逻辑分叉的写法。

    **不在这里把异常翻译成 `LLMError`**：调用方需要先看原始 HTTP 状态码来
    判断这是不是「内容审核拒了」，翻译过的错误只剩一句中文，判断不了。
    """
    model = str(payload.get("model") or "")
    async with httpx.AsyncClient(timeout=config.timeout_seconds, trust_env=provider_trust_env(url)) as client:
        for attempt in range(RATE_LIMIT_ATTEMPTS):
            started = time.perf_counter()

            def elapsed_ms() -> int:
                return int((time.perf_counter() - started) * 1000)

            try:
                resp = await client.post(url, json=payload, headers=headers)
            except TRANSIENT_TRANSPORT_ERRORS as exc:
                # Record the drop before retrying. Without this line a call that
                # failed twice and succeeded on the third attempt leaves no trace
                # at all, and "it works now" becomes impossible to confirm or
                # explain.
                logs.event(
                    "llm.transport", level="warn", model=model, attempt=attempt + 1,
                    ms=elapsed_ms(), error=f"{type(exc).__name__}: {exc}",
                )
                # A dropped socket is retried here rather than surfaced: the
                # request never reached a verdict, so reporting it as "cannot
                # reach the endpoint" both misdiagnoses it and throws away a
                # call the user has already waited minutes for. A real outage
                # still ends after the last attempt.
                if attempt < RATE_LIMIT_ATTEMPTS - 1:
                    await asyncio.sleep(_network_retry_delay(attempt))
                    continue
                raise
            # 429 is not a failure of this request, it is "not now". Two
            # segments run concurrently, so the same stage can fire two or
            # three calls within a second and trip a per-key concurrency
            # limit — the run then failed with "触发限流" even though a
            # two-second wait would have succeeded.
            if resp.status_code == 429:
                body = resp.text or ""
                logs.event(
                    "llm.rate_limit", level="warn", model=model, attempt=attempt + 1,
                    ms=elapsed_ms(), balance=_is_balance_error(body), body=body,
                )
                # "Out of quota" arrives as 429 too. Waiting cannot fix it, and
                # retrying turns one clear error into ten seconds of silence
                # followed by the same message — so it is raised immediately.
                if _is_balance_error(body):
                    raise LLMError(_BALANCE_MESSAGE, status=429, detail=body[:300], error_type="balance")
                if attempt < RATE_LIMIT_ATTEMPTS - 1:
                    await asyncio.sleep(_retry_after(resp, attempt))
                    continue
            if not resp.is_success:
                # The status code and the provider's own words are the only two
                # things that make a 4xx diagnosable; everything downstream keeps
                # at most a translated Chinese sentence.
                logs.event(
                    "llm.http", level="warn", model=model, attempt=attempt + 1,
                    status=resp.status_code, ms=elapsed_ms(), body=resp.text,
                )
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage") if isinstance(data, dict) else None
            logs.event(
                "llm.call", model=model, attempt=attempt + 1, status=resp.status_code,
                ms=elapsed_ms(), usage=usage,
                chars=len(str((data.get("choices") or [{}])[0].get("message", {}).get("content") or "")),
            )
            return data
    raise LLMError("LLM 请求未返回结果（重试次数已用尽）", error_type="rate_limit")


async def _post_with_moderation(
    url: str,
    headers: dict[str, str],
    config: LLMConfig,
    *,
    build: Any,
    system: str,
    user: str,
) -> tuple[dict[str, Any], str, list[str]]:
    """发出请求；被审核拦下就降低措辞强度重发。

    审核有两个方向，两条路都要走，因为它们的表现完全不同：

      · 入向 —— HTTP 400 + 「敏感/不安全」类措辞。提示词里的某个词撞了词表；
      · 出向 —— HTTP 200 + `finish_reason=content_filter`。模型自己把答案撤回了，
        正文是空的或半截。此前它落到 `extract_json` 的「不是合法 JSON」分支，
        用户拿着那句提示永远修不好：提示词没问题，被拦的是生成的内容。

    强度按 `_moderation_plan()` 递增，每一级都是**换说法不是删内容**。返回
    (响应体, 用到的强度, 被替换的短语)，最后一项是给用户看的：镜头描述被改写过，
    这件事得让人知道，而不是让成片悄悄变了调子。
    """
    hits = _moderation_hits(system, user)
    plans = _moderation_plan(hits)
    last_error: LLMError | None = None
    direction = ""

    for plan in plans:
        text_system, text_user = _apply_plan(plan, system, user)
        try:
            data = await _post_chat(url, build(text_system, text_user), headers, config)
        except Exception as exc:  # noqa: BLE001 - 分类后重抛
            if not _is_moderation_rejection(exc):
                raise _friendly_error(exc, config) from exc
            last_error = _friendly_error(exc, config)
            direction = "in"
            continue
        if _finish_reason(data) == _CONTENT_FILTER:
            last_error = LLMError(
                f"模型撤回了这次回答（finish_reason=content_filter，{config.model}）："
                "生成的内容被内容安全策略拦下，正文为空或只剩半截",
                error_type="moderation_out",
            )
            direction = "out"
            continue
        if plan != "as_is":
            logger.warning(
                "内容审核拦截（%s）：已按「%s」重发并成功。改写的短语：%s",
                config.model, plan, "、".join(hits) or "（提示词未命中已知词，只加了降调要求）",
            )
        return data, plan, hits

    # 所有强度都过不去。这时候不该再建议「重试」——重试得到的是同一个结果。
    where = "生成的内容" if direction == "out" else "提示词"
    status = getattr(last_error, "status", None)
    raise LLMError(
        f"内容审核连续拦截了 {len(plans)} 次（{config.model}"
        f"{f'，HTTP {status}' if status else ''}）：{where}被内容安全策略判为敏感，"
        "已经试过替换已知敏感短语、以及整句降低表述强度两种改写。"
        "说明这一段里有一个审核词表内没有等价说法的表述。"
        "请改写这一段描述后重试，或在设置里换一个审核更宽松的模型",
        detail=str(last_error or "")[:400],
        error_type="moderation_out" if direction == "out" else "moderation_in",
    )


async def chat(
    config: LLMConfig,
    *,
    system: str,
    user: str,
    images: list[str] | None = None,
    json_mode: bool = True,
    max_tokens: int = 2000,
    thinking: bool = True,
) -> LLMResult:
    """One chat completion against any OpenAI-compatible endpoint.

    `images` are data URLs (data:image/...;base64,...) for vision models.
    """
    # Ask for the reason rather than testing `configured` first: `configured` only
    # means "a key is present", so a placeholder passes it and the run would proceed
    # to a 401 instead of a clear explanation.
    reason = config.missing_reason() if hasattr(config, "missing_reason") else ""
    if not config.configured:
        reason = reason or (
            "尚未配置 LLM：请在「设置 → 供应商与 Key」里填入 API Key 并保存"
            "（也可以设置环境变量 STUDIO_LLM_API_KEY）"
        )
    if reason:
        raise LLMError(reason, error_type="not_configured")

    model = config.effective_vision_model if images else config.model

    def build(system_text: str, user_text: str) -> dict[str, Any]:
        """组请求体。抽成闭包是为了「被审核拒了之后换一版提示词重发」。"""
        body: Any = user_text
        if images:
            body = [{"type": "text", "text": user_text}] + [
                {"type": "image_url", "image_url": {"url": u, "detail": "high"}} for u in images
            ]
        built: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": body},
            ],
            "temperature": config.temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            built["response_format"] = {"type": "json_object"}
        # Cap the reasoning phase so the answer is guaranteed room. Measured on
        # glm-5: without a cap it spent 5325 characters thinking and returned an
        # empty body, so a JSON caller saw "无法解析" — a budget allocation problem,
        # not a formatting one.
        built["thinking"] = (
            {
                "type": "enabled",
                "budget_tokens": max(256, min(THINKING_BUDGET, max_tokens // 3)),
            }
            if thinking
            else {"type": "disabled"}
        )
        return built

    payload = build(system, user)
    url = config.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    # 审核拦截在这里处理，不在 `chat()` 里现写一遍重发：入向 400 和出向
    # `content_filter` 要共用同一套「换说法 → 整句降调」的强度表，两处各写一份
    # 迟早会分叉，而分叉的表现是「某个方向永远兜不住」——正是这次要修的 bug。
    try:
        data, moderation_plan, moderation_terms = await _post_with_moderation(
            url, headers, config, build=build, system=system, user=user
        )
    except LLMError as exc:
        if exc.status not in (502, 504):
            raise
        # A relay may time out while buffering a complete long answer even
        # though the model accepts the same request via SSE. Retry once using
        # streaming transport; callers still receive a complete LLMResult.
        # Do not change the user's persistent streaming/display preference.
        from .llm_stream import stream_chat

        logger.warning("非流式请求收到 HTTP %s，尝试一次流式传输：%s model=%s", exc.status, url, model)
        try:
            result = await stream_chat(
                config, system=system, user=user, images=images,
                json_mode=json_mode, max_tokens=max_tokens, thinking=thinking,
            )
        except LLMError as stream_exc:
            raise LLMError(
                f"非流式请求收到上游 HTTP {exc.status}；改用流式传输后仍失败：{stream_exc}",
                status=stream_exc.status,
                detail=f"地址：{url}\n模型：{model}\n非流式：{exc.detail}\n流式：{stream_exc.detail}",
                error_type=stream_exc.kind,
            ) from stream_exc
        result.raw["transport_fallback"] = {"from": "nonstream", "status": exc.status}
        return result

    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(
            "LLM 返回结构异常（缺少 choices[0].message.content）",
            detail=str(data)[:300],
            error_type="bad_response",
        ) from exc

    # Reasoning models put their plan in a separate field. It must be read before
    # the content is returned, because the field is the only place the model's
    # actual thinking survives: `_strip_reasoning` removes any inline trace to
    # keep JSON parseable, so without this the reasoning would be discarded
    # entirely and the UI could only show a summary.
    reasoning = ""
    for field_name in ("reasoning_content", "reasoning", "thinking"):
        value = message.get(field_name)
        if isinstance(value, str) and value.strip():
            reasoning = value.strip()
            break

    text = message.get("content") or ""
    finish = _finish_reason(data)

    if not text and reasoning:
        # The model thought without answering. Retry once with thinking off: a
        # truncated think is already billed, and returning nothing wastes it.
        #
        # 提示词用**实际生效的那一版**重建：如果这一轮是靠换说法才通过的，拿原始
        # 提示词重发等于把刚绕过去的审核再撞一次。
        payload = build(*_apply_plan(moderation_plan, system, user))
        payload["thinking"] = {"type": "disabled"}  # thinking_disabled_retry
        async with httpx.AsyncClient(timeout=config.timeout_seconds, trust_env=provider_trust_env(url)) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code == 200:
                retry_data = resp.json()
                message = retry_data["choices"][0]["message"]
                if message.get("content"):
                    result = LLMResult(
                        text=message["content"], raw=retry_data, model=model,
                        reasoning=reasoning,
                        finish_reason=_finish_reason(retry_data),
                        moderation_plan=moderation_plan,
                        moderation_terms=moderation_terms,
                    )
                    _publish_moderation(moderation_plan, moderation_terms, model=model)
                    return result
    result = LLMResult(
        text=text, raw=data, model=model, reasoning=reasoning, finish_reason=finish,
        moderation_plan=moderation_plan, moderation_terms=moderation_terms,
    )
    _publish_moderation(moderation_plan, moderation_terms, model=model)
    return result


async def chat_messages(
    config: LLMConfig,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int = 2000,
    json_mode: bool = False,
    tool_choice: str = "auto",
) -> dict[str, Any]:
    """一次完整 messages 的调用，可选带上工具。

    与 `chat()` 分开，而不是给它加参数：`chat()` 的约定是「system + user 进，
    一段文本出」，所有 agent 都建立在这个约定上。工具循环需要拿到原始 message
    （含 `tool_calls`），把它追加回对话再问一次——输出形状不一样，混在一个函数
    里会让每个调用方都要判断自己拿到的是哪种。

    返回 {"content", "tool_calls", "finish_reason", "raw"}。
    """
    reason = config.missing_reason() if hasattr(config, "missing_reason") else ""
    if not config.configured:
        reason = reason or "尚未配置 LLM：请在「设置 → 供应商与 Key」里填入 API Key 并保存"
    if reason:
        raise LLMError(reason, error_type="not_configured")

    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": max_tokens,
    }
    # 思考会吃掉工具调用要用的预算：实测开着思考问一次搜索，2500 token 全用在
    # reasoning 上，content 只剩一个「我」字。工具循环一律关思考。
    payload["thinking"] = {"type": "disabled"}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    if json_mode and not tools:
        payload["response_format"] = {"type": "json_object"}

    url = config.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    try:
        async with httpx.AsyncClient(timeout=config.timeout_seconds, trust_env=provider_trust_env(url)) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # noqa: BLE001 - normalised below
        raise _friendly_error(exc, config) from exc

    try:
        choice = data["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("LLM 返回结构异常", detail=str(data)[:300]) from exc

    return {
        "content": str(message.get("content") or ""),
        "tool_calls": message.get("tool_calls") or [],
        "finish_reason": str(choice.get("finish_reason") or ""),
        "raw": data,
    }


#: The reasoning text of the most recent call, keyed by the current asyncio task.
#:
#: A module-level variable would be wrong with concurrent agents: four segments
#: running at once would interleave their reasoning. Task-local storage keeps each
#: chain of execution isolated.
_reasoning_sink: "contextvars.ContextVar[list[str]]" = contextvars.ContextVar(
    "llm_reasoning", default=[]
)


def _publish_reasoning(reasoning: str, *, model: str) -> None:
    """Record the model's own plan for the caller to pick up."""
    try:
        _reasoning_sink.set([reasoning])
    except Exception:  # noqa: BLE001
        pass


def take_reasoning() -> str:
    """Read and clear the reasoning recorded since the last call."""
    try:
        items = _reasoning_sink.get()
        _reasoning_sink.set([])
        return "\n\n".join(items)
    except Exception:  # noqa: BLE001
        return ""


#: 最近一次成功调用为了让审核放行做了什么，键随 asyncio 任务走。
#:
#: 与 reasoning 用同一套机制（而不是在多处层层加返回值）：`chat_json` 的契约是
#: 「进两个字符串、出一个对象」，为了带一个旁注去改所有 agent 的调用签名，会让
#: 每个调用点都要处理一个它们多半不关心的值。
_moderation_sink: "contextvars.ContextVar[list[dict[str, Any]]]" = contextvars.ContextVar(
    "llm_moderation", default=[]
)


def _publish_moderation(plan: str, terms: list[str], *, model: str) -> None:
    if plan in ("", "as_is"):
        return
    try:
        current = list(_moderation_sink.get())
        current.append({"plan": plan, "terms": list(terms), "model": model})
        _moderation_sink.set(current)
    except Exception:  # noqa: BLE001
        pass


def take_moderation() -> list[dict[str, Any]]:
    """读取并清空自上次调用以来，为通过审核而做过的改写。"""
    try:
        items = _moderation_sink.get()
        _moderation_sink.set([])
        return items
    except Exception:  # noqa: BLE001
        return []

async def chat_json(
    config: LLMConfig,
    *,
    system: str,
    user: str,
    images: list[str] | None = None,
    max_tokens: int = 2000,
    retries: int = 1,
    thinking: bool = True,
) -> Any:
    """chat() + JSON extraction, with one repair retry.

    The retry exists because the dominant failure is formatting (prose wrapper,
    truncated output), not a bad instruction — replaying the same request with a
    stricter reminder usually succeeds. It never re-sends a *billed* action, only
    a text completion, so it cannot duplicate a paid media action.

    Truncation is handled separately from malformed prose. A reasoning model spends
    most of `max_tokens` thinking, the answer is cut at `finish_reason=length`, and
    the repair step happily closes the braces — so the caller receives a *smaller*
    object instead of an error, and the missing half only shows up two stages later
    as "关键内容缺失". Here a cut-off reply is retried with twice the budget, and
    the retry's prompt says why.
    """
    last_error: Exception | None = None
    budget = max_tokens
    for attempt in range(retries + 1):
        prompt = user
        if attempt > 0:
            prompt += (
                "\n\n【重要】上一次回复无法解析为 JSON。请只输出一个合法 JSON 对象，"
                "不要任何解释文字、不要 markdown 代码块。"
            )
        result = await chat(
            config, system=system, user=prompt, images=images,
            json_mode=True, max_tokens=budget, thinking=thinking,
        )
        # Surface the model's own plan before parsing it away. `chat_json` returns
        # only the object, so the reasoning would otherwise be lost between the
        # HTTP call and the caller — which is why the trace could only ever show a
        # summary instead of the model's actual thinking.
        if result.reasoning:
            _publish_reasoning(result.reasoning, model=result.model)

        if result.truncated or looks_truncated(result.text):
            # The reply was cut, so whatever parses out of it is incomplete by
            # construction. `looks_truncated` covers providers that do not report
            # `finish_reason`: a body not ending in `}` / `]` was closed by the
            # repair step, not by the model, and a silently emptied array is worse
            # than an error because nothing downstream can notice it.
            last_error = LLMError(
                f"输出被截断（finish_reason={result.finish_reason or '未提供'}，"
                f"max_tokens={budget}）",
                # `stop` 却截断，说明截断不是本次预算造成的：供应商侧有更低的
                # 长度上限，或思考已把预算吃光。写进 detail，界面上那句 headline
                # 保持不变，但排查时能一眼看出「调大 max_tokens 没用」。
                detail=(
                    ""
                    if result.finish_reason == "length"
                    else "上游未报 length（返回 stop），截断来自供应商侧的长度/内容限制，"
                    "调大 max_tokens 通常无效；应改为拆分任务或换模型。"
                ),
                error_type="truncated",
            )
            budget = min(16000, budget * 2)
            continue

        try:
            value = extract_json(result.text)
        except LLMError as exc:
            last_error = exc
            continue
        # An empty object from a non-empty reply is the repair step's way of saying
        # "nothing survived". Accepting it silently is how a stage proceeded with
        # zero characters / zero shots, so treat it as a failed attempt.
        if isinstance(value, dict) and not value and result.text.strip():
            last_error = LLMError("解析结果为空对象，按失败处理", error_type="empty_reply")
            continue
        return value
    # 截断和「解析不出来」是两件事：前者是预算不够（调大 max_tokens 有用），后者是
    # 格式问题（换模型有用）。把它们说成同一句，用户只能两样都试一遍。
    if isinstance(last_error, LLMError) and last_error.error_type == "truncated":
        raise last_error
    if isinstance(last_error, LLMError) and last_error.error_type == "moderation_out":
        raise last_error
    raise LLMError(
        "模型连续返回无法解析的内容，建议换模型或缩短输出长度",
        detail=str(last_error) if last_error else "",
        error_type=getattr(last_error, "error_type", "") or "unparseable",
    )
