"""Agent base contract.

Every agent is a small, single-purpose unit with the same shape:

    async def run(ctx: AgentContext) -> AgentResult

Why this shape and not "an agent is a chat loop": the failure mode of multi-agent
systems is that agents talk to each other in prose and nobody can tell which one
produced a bad output. Here each agent declares typed input expectations and
returns a typed payload, the orchestrator persists both, and the UI shows them.
That makes the pipeline inspectable and each failure traceable to its source.

Agents are stateless; all state lives in the run record, so any step can be
retried in isolation.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings, settings as default_settings
from ..schemas import AgentResult
from ..tools import llm as llm_tool


#: 模型调用失败时跟在原始报错后面的人话解释，按 `tools/llm.py` 的错误分类。
#: 错误原文保留（排查需要），解释回答两件事：这错是什么意思、程序接下来怎么办。
MODEL_ERROR_HINTS = {
    "timeout": "意思是模型在规定时间内没有答完（长输出常见）——程序会自动重试，仍失败会如实报出这一步",
    "transport": "意思是连接中途被断开——程序会自动重试",
    "unreachable": "意思是连不上模型服务——请检查网关地址是否在线、本机网络是否可达",
    "rate_limit": "意思是请求太密集被限流——程序会稍等后自动重试",
    "http_error": "收到上游 HTTP 错误；请求参数、传输方式或网关内部转发均可能导致失败，需要结合响应详情判断",
    "moderation_in": "意思是发给模型的提示词被内容审核拦截——程序会改用更中性的表述重发",
    "moderation_out": "意思是模型写出的内容被审核拦截——程序会要求模型换一种说法重写",
    "truncated": "意思是输出被截断（思考太长顶到上限）——程序会自动重试",
    "empty_reply": "意思是模型这一次没有作答——程序会自动重试",
    "bad_response": "意思是模型返回的结构异常——程序会自动重试",
    "unparseable": "意思是输出无法解析成约定的格式——程序会要求模型重答",
    "auth": "意思是 API Key 无效或没有权限——不会自动重试，请检查 Key",
    "balance": "意思是账户余额或额度不足——不会自动重试，请充值或更换服务商",
    "not_found": "意思是模型名或接口地址不存在——不会自动重试，请检查设置里的模型配置",
    "invalid_request": "意思是请求参数被服务商拒绝——请检查设置里的模型配置",
    "not_configured": "意思是还没有配置模型——请先在设置里完成配置",
}


def agent_label(agent_id: str) -> str:
    """Agent id → 用户认识的中文名（「script_adaptor」→「内容改编」）。

    找不到映射时原样返回 id：宁可看到英文代号，也不要显示一个错的岗位名。
    """
    try:
        from ..models_config import AGENT_IDS_LABEL

        return AGENT_IDS_LABEL.get(agent_id, agent_id)
    except Exception:  # noqa: BLE001 - label is cosmetic, never break a call
        return agent_id


def model_error_hint(exc: Exception) -> str:
    """错误 → 台账里那一句人话解释。LLMError 自带分类，其余异常没有解释可给。"""
    etype = ""
    try:
        etype = exc.kind  # LLMError.kind: 显式分类，缺省时从文案反推
    except AttributeError:
        etype = getattr(exc, "error_type", "") or ""
    return MODEL_ERROR_HINTS.get(str(etype or ""), "")


@dataclass
class AgentContext:
    """Everything an agent is allowed to read."""

    run_id: str
    brief: dict[str, Any]
    settings: Settings = field(default_factory=lambda: default_settings)
    # Image paths uploaded by the user (absolute local paths).
    image_paths: list[str] = field(default_factory=list)
    # Set by the reviewer when a step is being redone.
    revision_feedback: str = ""
    # Accumulated cost, enforced by the orchestrator before paid steps.
    spent_usd: float = 0.0
    # Arbitrary per-agent scratch space (e.g. generated image URLs).
    scratch: dict[str, Any] = field(default_factory=dict)
    # Per-agent model bindings. Resolved by the orchestrator from the saved
    # provider settings; an agent uses `config_for(self.name)` to find its own.
    model_settings: Any = None
    # Shared workflow state: what earlier agents produced, addressed by key.
    # Workflow agents read and write this; it is the same dictionary the
    # executor's predicates evaluate against, so agents and control flow agree
    # on one source of truth.
    state: dict[str, Any] = field(default_factory=dict)
    #: Tool access (MCP). None when tools are unavailable.
    tools: Any = None
    #: RAG index for this run. None when retrieval is unavailable — agents must
    #: treat retrieval as an enhancement and degrade to reading the chapter text.
    rag: Any = None

    async def retrieve(self, query: str, *, top_k: int = 4) -> str:
        """Retrieve passages for a query and format them with citations.

        Returns an empty string when RAG is unavailable, so a caller can
        concatenate the result into a prompt without branching on availability.

        Every call is traced: "which passages did it read before deciding" is
        exactly the question a reviewer asks about a fact-check, and it cannot be
        answered from the conclusion alone.
        """
        if self.rag is None or not query.strip():
            return ""
        started = time.perf_counter()
        try:
            result = await self.rag.retrieve(query, top_k=top_k)
            elapsed = int((time.perf_counter() - started) * 1000)
            if not result.hits:
                self.trace_step("retrieve", f"检索「{query[:28]}」", "无命中", ok=False, elapsed_ms=elapsed)
                return ""
            hits = result.hits[:top_k]
            self.trace_step(
                "retrieve",
                f"检索「{query[:28]}」命中 {len(hits)} 段",
                "、".join(str(h.get("citation") or "?") for h in hits),
                elapsed_ms=elapsed,
            )
            return "\n".join(
                f"- 〔{h.get('citation', '?')}〕{h.get('text', '')}"
                for h in hits
            )
        except Exception:  # noqa: BLE001 - retrieval must never break a step
            return ""

    #: Which agent this context belongs to, for tracing. Set by the task handler.
    name: str = ""

    def trace_step(
        self,
        kind: str,
        title: str,
        detail: str = "",
        *,
        ok: bool = True,
        elapsed_ms: int = 0,
    ) -> None:
        """Record one step for this agent, best effort.

        Failures are swallowed: the trace is diagnostic output, and losing a step
        must never abort the work it was describing.
        """
        if not self.run_id:
            return
        try:
            from ..doc.trace import current

            current(self.run_id, self.name or "agent").add(
                kind, title, detail, ok=ok, elapsed_ms=elapsed_ms
            )
        except Exception:  # noqa: BLE001
            pass

    @property
    def input(self) -> dict[str, Any]:
        """Alias for the run's original input, so workflow agents can read either."""
        return self.brief

    def brief_str(self, key: str, default: str = "（未填写）") -> str:
        value = (self.brief or {}).get(key, "")
        return str(value).strip() or default

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.state:
            return self.state.get(key)
        return (self.brief or {}).get(key, default)

    async def agent_ask(
        self,
        agent_id: str,
        *,
        system: str,
        user: str,
        images: list[str] | None = None,
        max_tokens: int = 2000,
        independent_from: str = "",
        stream: bool | None = None,
        live_trace: bool = True,
        thinking: bool = True,
    ) -> Any:
        """Call the LLM, announcing who is calling and for how long.

        The announcement is the point. With streaming switched off a model call
        produces nothing observable until it returns — one stage can be silent for
        a minute — so a client that only renders events has nothing to render and
        the run reads as hung. `mark_working` gives it a heartbeat to display
        (agent + elapsed seconds) even though the call itself reports nothing.

        The heartbeat is cleared in `finally`, including on failure: a stale
        "still working" row after a crash is worse than no row at all.
        """
        from ..doc.trace import clear_working, mark_working

        mark_working(self.run_id, agent_id, "正在调用大模型", keep_label=True)
        try:
            return await self._agent_ask_bound(
                agent_id,
                system=system,
                user=user,
                images=images,
                max_tokens=max_tokens,
                independent_from=independent_from,
                stream=stream,
                live_trace=live_trace,
                thinking=thinking,
            )
        finally:
            # Release only this agent's hold: another segment may be inside the
            # same agent right now, and clearing the whole run would blank its row.
            clear_working(self.run_id, agent_id)

    async def _agent_ask_bound(
        self,
        agent_id: str,
        *,
        system: str,
        user: str,
        images: list[str] | None = None,
        max_tokens: int = 2000,
        independent_from: str = "",
        stream: bool | None = None,
        live_trace: bool = True,
        thinking: bool = True,
    ) -> Any:
        """Call the LLM using a *named* agent's model binding.

        Workflow agents are plain functions, not BaseAgent subclasses, so they
        cannot call `self.ask_json`. This method gives them the same behaviour:
        per-agent model binding, fallback chain, JSON extraction with one repair.

        `independent_from` requests a binding different from another agent's —
        used by the fact checker and verifier so cross-validation is real rather
        than the same model agreeing with itself.
        """
        from ..tools.llm import chat_json

        # Trace the call with its input shape and output shape, not the full body:
        # a reviewer needs to see what the agent was told and what it answered,
        # and keeping the whole prompt would make a poll response enormous.
        #
        # Streaming when available: the reasoning deltas reach the trace as they
        # are produced, so a 40-second think shows progress instead of freezing.
        started = time.perf_counter()
        # `None` means "streaming did not produce an answer" and is what the check
        # below tests for. It must NOT be `False`: with streaming switched off the
        # branch below never runs, and a `False` seed made `is not None` true, so
        # every agent_ask returned `False` and the first `.get()` on it raised
        # "'bool' object has no attribute 'get'". Latent while streaming was always
        # on; the 设置 → 调用方式 switch is what made it reachable.
        streamed: Any = None
        # Most agents follow the user's global display preference. A stage may
        # opt out when its response is a large structured object: the model can
        # still think, but the reasoning and answer are committed to the trace
        # only after the complete response arrives. `None` deliberately means
        # "inherit the setting" so existing callers keep their behaviour.
        use_stream = stream_enabled(self.model_settings) if stream is None else stream
        if use_stream:
            streamed = await self._ask_streaming(
                agent_id,
                system=system,
                user=user,
                images=images,
                max_tokens=max_tokens,
                independent_from=independent_from,
                started=started,
                live_trace=live_trace,
                thinking=thinking,
            )
        if streamed is not None:
            return streamed

        try:
            out = await chat_json(
                self.config_for(agent_id, require_independent_from=independent_from),
                system=system,
                user=user,
                images=images,
                max_tokens=max_tokens,
                thinking=thinking,
            )
        except Exception as exc:
            # Record a terminal cause so the pipeline can halt the run. Without
            # this, an empty balance makes every remaining segment fail identically
            # and the surfaced error describes the last attempt rather than the
            # real reason. The document agents go through `agent_ask`, not
            # `BaseAgent.ask_json`, so the flag must be set here too.
            try:
                from ..free_models import is_terminal

                if is_terminal(str(exc)):
                    self.state["__terminal"] = str(exc)[:200]
            except Exception:  # noqa: BLE001 - bookkeeping must not mask the cause
                pass
            self.trace_step(
                "model",
                f"这次调用大模型失败（{agent_label(agent_id)}要做的这一步）",
                f"原始报错：{str(exc)[:200]}"
                + (f"\n响应详情：{getattr(exc, 'detail', '')[:1000]}" if getattr(exc, 'detail', '') else "")
                + (f"\n{model_error_hint(exc)}" if model_error_hint(exc) else ""),
                ok=False,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
            raise

        elapsed = int((time.perf_counter() - started) * 1000)
        cfg = self.config_for(agent_id, require_independent_from=independent_from)

        # Record the model's own plan, not just a size summary. This is the
        # difference between "it called the model" and "here is what it thought".
        # Reasoning-capable models (glm-5, glm-4.6) return planning text in a
        # separate field, and showing it is what the trace is for.
        reasoning = ""
        try:
            from ..tools.llm import take_reasoning

            reasoning = take_reasoning()
        except Exception:  # noqa: BLE001
            pass

        if reasoning:
            self.trace_step(
                "reasoning",
                "模型思考过程",
                # 20k characters is roughly "the whole thought" for glm-5 (it thinks
                # 4–6k on a normal call); the old 1500 cut every plan mid-sentence,
                # which is the opposite of what a trace is for. The UI keeps the
                # block scrollable, so length costs nothing on screen.
                reasoning[:20000],
                elapsed_ms=elapsed,
            )

        self.trace_step(
            "model",
            f"调用大模型 {cfg.model}（{agent_label(agent_id)}这一步的模型）",
            f"把任务说明和材料共 {len(system) + len(user)} 字发给模型"
            f"（要求它按约定格式返回结果），"
            f"{elapsed / 1000:.1f} 秒后收到 {_json_len(out)} 字的回答"
            + (
                f"；模型另外留下了 {len(reasoning)} 字的思考过程，已单独记录在下面"
                if reasoning
                else ""
            ),
            elapsed_ms=elapsed,
        )
        # Every caller of `agent_ask` expects an object and calls `.get` on it.
        # Coercing here fixes all of them at once: a top-level array otherwise
        # raises "'list' object has no attribute 'get'" at whichever agent happened
        # to receive it, which reads like a code bug rather than a reply shape.
        return _coerce_object(out)

    async def _ask_streaming(
        self,
        agent_id: str,
        *,
        system: str,
        user: str,
        images: list[str] | None,
        max_tokens: int,
        independent_from: str,
        started: float,
        live_trace: bool = True,
        thinking: bool = True,
    ) -> Any:
        """Stream the completion and turn deltas into live trace steps.

        Returns the parsed object, or None when streaming is unavailable or the
        streamed text cannot be parsed — the caller then falls back to the
        non-streaming path rather than failing the stage.

        Why the deltas go into the trace rather than straight to the browser:
        the pipeline emits one message per completed stage, and the trace is what
        it carries. Writing deltas there means the existing message stream shows
        progress without a second transport (SSE) running alongside it.
        """
        from ..tools.llm import LLMError, extract_json
        from ..tools.llm_stream import stream_chat

        # Buffer the reasoning so it can be written as one growing step, and the
        # content so it can be parsed at the end.
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        last_flush = 0.0
        step_key = f"stream-{agent_id}"

        async def on_delta(channel: str, piece: str) -> None:
            nonlocal last_flush
            if channel == "reasoning":
                reasoning_parts.append(piece)
            else:
                content_parts.append(piece)
            # Flush at most twice a second: every token would make the trace
            # unreadable and the poll response large.
            now = time.perf_counter()
            if now - last_flush < 0.5:
                return
            last_flush = now
            if live_trace and reasoning_parts:
                # Upsert, not add: this fires every 0.5 s while a stage thinks,
                # and appending would fill the capped trace with duplicates.
                self._trace_reasoning("模型思考（进行中）", "".join(reasoning_parts))

        try:
            result = await stream_chat(
                self.config_for(agent_id, require_independent_from=independent_from),
                system=system,
                user=user,
                images=images,
                max_tokens=max_tokens,
                json_mode=True,
                on_delta=on_delta,
                thinking=thinking,
            )
        except LLMError:
            # Streaming is an enhancement; a provider that cannot stream should
            # degrade to the normal path, not fail the stage.
            return None
        except Exception:  # noqa: BLE001
            return None

        elapsed = int((time.perf_counter() - started) * 1000)
        reasoning = "".join(reasoning_parts).strip() or result.reasoning

        if result.truncated:
            # Cut off at the token ceiling — most often because a reasoning model
            # spent the budget thinking. Parsing it here would hand the agent a
            # half object (the repair step closes the braces), so hand the job to
            # the non-streaming path, which retries with double the budget.
            self.trace_step(
                "warn",
                "输出被截断，改用非流式重试",
                f"模型 {result.model}：思考 {len(reasoning)} 字后达到 max_tokens",
                ok=False,
                elapsed_ms=elapsed,
            )
            return None

        if reasoning:
            # Update the in-progress row instead of appending a second one. The
            # stream wrote "模型思考（进行中）" as text arrived, and writing
            # "模型思考过程" here produced a near-duplicate: one thought displayed
            # twice, while the model was only called once.
            self._trace_reasoning("模型思考过程", reasoning, elapsed_ms=elapsed)

        try:
            out = extract_json(result.text)
        except LLMError:
            # Unparseable streaming output is not fatal: return None so the caller
            # retries the non-streaming path, which has a repair step.
            return None

        cfg = self.config_for(agent_id, require_independent_from=independent_from)
        self.trace_step(
            "model",
            f"调用大模型 {cfg.model}（"
            f"{'流式接收' if live_trace else '后台接收，完成后显示'}，"
            f"{agent_label(agent_id)}这一步）",
            f"把任务说明和材料共 {len(system) + len(user)} 字发给模型，"
            + (
                "回答边生成边传回，"
                if live_trace
                else "传输期间保持界面静默，完成后一次性显示，"
            )
            + f"{elapsed / 1000:.1f} 秒后收到 {_json_len(out)} 字"
            + (f"；思考过程 {len(reasoning)} 字已单独记录在下面" if reasoning else ""),
            elapsed_ms=elapsed,
        )
        return _coerce_object(out)

    def _trace_reasoning(self, title: str, detail: str, *, elapsed_ms: int = 0) -> None:
        """Record or update the reasoning row for this model call."""
        if not self.run_id:
            return
        try:
            from ..doc.trace import upsert

            upsert(
                self.run_id, self.name or "agent", "reasoning", title, detail,
                elapsed_ms=elapsed_ms,
            )
        except Exception:  # noqa: BLE001
            pass

    def config_for(self, agent_id: str, *, require_independent_from: str = ""):
        """The LLM config this specific agent should use.

        Falls back to the global settings when no provider is configured, so the
        system keeps working for a user who has only set an environment variable.

        `require_independent_from` names another agent whose config this call must
        NOT match. Cross-validation is meaningless when the verifier runs on the
        same model as the producer — it agrees by construction. When the two
        resolve to the same provider, the checker is pushed to the next available
        provider instead of silently sharing one.
        """
        if self.model_settings is not None:
            profile = self.model_settings.provider_for(agent_id)
            if profile is not None:
                from ..models_config import provider_to_llm_config

                if require_independent_from:
                    other = self.model_settings.provider_for(require_independent_from)
                    if other is not None and other.id == profile.id:
                        # Try every other configured provider before giving up.
                        for candidate in self.model_settings.providers:
                            if candidate.id != profile.id and candidate.model:
                                profile = candidate
                                break
                return provider_to_llm_config(
                    profile,
                    needs_vision=agent_id in {"intake"},
                )
        return self.settings.llm

    def fallback_configs(self, agent_id: str) -> list[Any]:
        """Ordered alternative configs to try if the primary call fails.

        The chain is the agent's own fallback list, then the shared "our free
        model" when the deployment offers one, then the default provider. Ordered
        by preference, deduplicated by endpoint+model so the same target is never
        attempted twice.

        This is what keeps a run alive when a user's key expires mid-way: the
        agent degrades to another model instead of failing the pipeline.
        """
        from .. import models_config
        from ..free_models import load_shared

        candidates: list[Any] = []

        if self.model_settings is not None:
            chain = getattr(self.model_settings, "fallbacks", {}) or {}
            for pid in chain.get(agent_id, []) or []:
                for profile in self.model_settings.providers:
                    if profile.id == pid:
                        from ..models_config import provider_to_llm_config

                        candidates.append(
                            provider_to_llm_config(profile, needs_vision=agent_id in {"intake"})
                        )

        shared = load_shared()
        if shared.enabled and shared.configured:
            from ..config import LLMConfig

            candidates.append(
                LLMConfig(
                    base_url=shared.base_url,
                    api_key=shared.api_key,
                    model=shared.model,
                    vision_model=shared.vision_model or shared.model,
                    timeout_seconds=shared.timeout,
                    temperature=self.settings.llm.temperature,
                )
            )

        primary = self.config_for(agent_id)
        seen = {(primary.base_url, primary.model)}
        unique: list[Any] = []
        for cfg in candidates:
            key = (cfg.base_url, cfg.model)
            if key in seen:
                continue
            seen.add(key)
            unique.append(cfg)
        return unique


class BaseAgent(ABC):
    """Common timing/error handling so subclasses only implement the work."""

    name: str = "base"
    #: Human-facing label shown in the UI trace.
    label: str = "Agent"
    #: Whether this agent spends money on a paid API (drives the budget gate).
    paid: bool = False

    @abstractmethod
    async def _run(self, ctx: AgentContext) -> AgentResult: ...

    async def run(self, ctx: AgentContext) -> AgentResult:
        started = time.perf_counter()
        try:
            result = await self._run(ctx)
            result.agent = self.name
            result.duration_ms = int((time.perf_counter() - started) * 1000)
            return result
        except Exception as exc:  # noqa: BLE001 - surfaced to the run, never crashes the loop
            elapsed = int((time.perf_counter() - started) * 1000)
            return AgentResult(
                agent=self.name,
                ok=False,
                error=str(exc) or exc.__class__.__name__,
                duration_ms=elapsed,
            )

    def _record_moderation(self, ctx: AgentContext) -> None:
        """记下这次调用为了让审核放行改写过的表述。

        收集在 state 里而不是逐层返回，是因为需要它的只有最后那条「本次生成提示
        已改写」的通知，而中间每一层都不关心。清空动作必须无条件做：写入的判定
        是「这次用了改写」，留着上一次的记录会让没改写的分段也报一句。
        """
        try:
            from ..tools.llm import take_moderation

            notes = take_moderation()
            if notes:
                ctx.state.setdefault("moderation_notes", []).extend(notes)
        except Exception:  # noqa: BLE001 - bookkeeping must not fail a stage
            pass

    async def ask_json(
        self,
        ctx: AgentContext,
        *,
        system: str,
        user: str,
        images: list[str] | None = None,
        max_tokens: int = 2000,
    ) -> Any:
        """Call the LLM using *this agent's* model binding, with fallback.

        A failed primary call falls through the agent's fallback chain, but only
        for failures another model could actually fix (expired key, quota, timeout).
        A content-policy rejection or unparseable output would fail identically
        everywhere, so it surfaces immediately instead of multiplying calls.
        """
        from ..free_models import should_fallback
        from ..tools.llm import LLMError

        primary = ctx.config_for(self.name)
        try:
            result = await llm_tool.chat_json(
                primary,
                system=system,
                user=user,
                images=images,
                max_tokens=max_tokens,
            )
            self._record_moderation(ctx)
            return result
        except LLMError as primary_error:
            # Only failures another model could fix are worth retrying; a content
            # rejection or unparseable output fails identically everywhere.
            if not should_fallback(
                str(primary_error), getattr(primary_error, "error_type", "")
            ):
                # Record a terminal cause (empty balance) so the pipeline can stop
                # the run, instead of letting every remaining segment fail the same
                # way and burying the real reason under a pile of shot errors.
                try:
                    from ..free_models import is_terminal

                    if is_terminal(str(primary_error)):
                        self.state["__terminal"] = str(primary_error)[:200]
                except Exception:  # noqa: BLE001 - bookkeeping must not mask the cause
                    pass
                raise
            attempted = 1
            last_error: Exception = primary_error

        fallbacks = ctx.fallback_configs(self.name)
        for cfg in fallbacks:
            attempted += 1
            try:
                result = await llm_tool.chat_json(
                    cfg,
                    system=system,
                    user=user,
                    images=images,
                    max_tokens=max_tokens,
                )
                self._record_moderation(ctx)
                return result
            except LLMError as inner:
                last_error = inner
                if not should_fallback(str(inner), getattr(inner, "error_type", "")):
                    raise
                continue

        raise LLMError(
            f"所有可用模型都调用失败（已尝试 {attempted} 个）。最后一次错误：{last_error}",
            detail=str(last_error),
        )


SYSTEM_PREAMBLE = """你是一个历史纪录片内容生产系统中的专职 Agent。

通用规则：
1. 只输出一个合法 JSON 对象，不要 markdown 代码块，不要任何解释性前后缀。
2. 所有结论必须基于给定事实。事实中没有的信息（年份、人名、地点、数字）
   一律不得臆造；需要用到时放进 unknowns 或使用保守表述。
3. 不写没有信息量的评价（「具有重要意义」「影响深远」），不写导语式开场。
4. 中文输出，专有名词可保留英文。
"""

#: Whether streaming is enabled.
#:
#: The setting in 设置 → 调用方式 (stored in providers.json) is the decision; the
#: environment variable is an operator override for a deployment, and the only way
#: to force it without touching the file. Read per call rather than at import, so
#: toggling it takes effect on the next stage instead of the next restart.
#:
#: What the switch actually buys: with streaming on, a reply that is cut off at
#: `max_tokens` costs two calls — the streamed attempt, then a non-streaming retry
#: with double the budget. With it off there is only the second call, so a run
#: stops paying twice for the same truncation. The trade is visibility: nothing is
#: shown until the model finishes.
def stream_enabled(model_settings: Any = None) -> bool:
    env = (os.environ.get("STUDIO_LLM_STREAM") or "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    if env in {"1", "true", "yes", "on"}:
        return True
    settings = model_settings
    if settings is None:
        try:
            from .. import models_config

            settings = models_config.load()
        except Exception:  # noqa: BLE001 - an unreadable config keeps the default
            return True
    return bool(getattr(settings, "stream", True))


def _coerce_object(value: Any) -> dict[str, Any]:
    """Turn a model reply into an object, whatever shape it came back in.

    Models occasionally answer with a top-level array (`[{"shots": [...]}]`) even
    when the prompt asks for an object. Every agent then calls `.get` on it and
    dies. Unwrapping here means no agent needs its own guard.

    Rules:
      · dict   → unchanged
      · list   → keep the complete list under `items`; when its first item is an
                 object, mirror that object's keys for backward compatibility
      · other  → `{"value": ...}` so the caller does not crash on a bare string

    Keeping the complete list is essential. A storyboard model sometimes returns
    a bare array of shots; the old implementation returned only the first dict and
    silently discarded every remaining shot, after which the storyboard stage
    reported an empty board even though the model had supplied one.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        out: dict[str, Any] = {}
        first = next((item for item in value if isinstance(item, dict)), None)
        if first is not None:
            out.update(first)
        out["items"] = value
        return out
    if value is None:
        return {}
    return {"value": value}


def _json_len(value: Any) -> int:
    """Approximate serialised size of a model response, for the trace line."""
    import json

    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return len(str(value))
