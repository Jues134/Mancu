"""Execution trace — record what an agent actually did, not just its conclusion.

The problem this solves: the UI shows one line per stage ("锁定画风 + 2 个人物定妆照"),
which answers "what came out" but not the questions a reviewer actually asks:

  · what did it read, and what did it decide from that?
  · which tool did it call, with what arguments, and what came back?
  · why did the critic reject — which rule, on which shot?
  · when rework happened, what changed on the second attempt?

Those are not summarisable into one line, so they are recorded as structured steps
and attached to the stage's message. The summary stays as the headline; the steps
are the detail behind it.

Design constraints that shape this:

  · **Bounded.** A run makes hundreds of tool and model calls; keeping them all
    would bloat a poll response to megabytes. Each agent keeps the most recent
    `MAX_STEPS` and marks truncation rather than silently dropping the middle.
  · **Not a log file.** Steps are attached to the message that produced them, so
    the UI can render them inline against the agent instead of correlating
    timestamps across a separate stream.
  · **Free of secrets.** Prompts are truncated and API keys never appear; a trace
    that leaks the key would be worse than no trace.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


#: Per-agent cap. Enough to explain a stage, small enough to keep polling cheap.
MAX_STEPS = 24

#: How much of a prompt or tool result to keep. Long enough to see the decision,
#: short enough that twenty steps stay readable.
SNIPPET = 220

#: Reasoning steps get far more room: a model's plan is the substance the reader
#: wants, not a preview of it. Truncating it to a line would defeat the purpose.
#:
#: Raised to 20k after a run showed the panel cutting off at 1800 characters while
#: glm-5 routinely thinks 4–6k: the reader saw a plan stop mid-sentence and asked,
#: reasonably, why the thinking "was not finished". The cap now only guards
#: against a runaway stream, not against normal output.
REASONING_LIMIT = 20000


@dataclass
class TraceStep:
    """One observable action taken by an agent."""

    kind: str          # think | tool | retrieve | decide | reject | accept | write | warn
    title: str         # short label, e.g. "调用 search_reference"
    detail: str = ""   # the substance: args, snippet of result, or reasoning
    ok: bool = True
    elapsed_ms: int = 0
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "detail": self.detail,
            "ok": self.ok,
            "elapsed_ms": self.elapsed_ms,
        }


class AgentTrace:
    """Collects steps for one stage of one agent."""

    def __init__(self, agent: str, *, limit: int = MAX_STEPS) -> None:
        self.agent = agent
        self.limit = limit
        self._steps: list[TraceStep] = []
        self._dropped = 0
        self._t0 = time.perf_counter()

    # ---------------------------------------------------------------- record

    def add(
        self,
        kind: str,
        title: str,
        detail: str = "",
        *,
        ok: bool = True,
        elapsed_ms: int = 0,
    ) -> None:
        if len(self._steps) >= self.limit:
            # Keep the newest and count the loss; silently dropping the middle
            # would make a long stage look shorter than it was.
            self._steps.pop(0)
            self._dropped += 1
        self._steps.append(
            TraceStep(
                kind=kind,
                title=title[:60],
                # A reasoning step carries the model's own plan, which is the point
                # of showing it; clipping it to a summary line would defeat that.
                detail=_clip(detail, REASONING_LIMIT if kind == "reasoning" else SNIPPET),
                ok=ok,
                elapsed_ms=elapsed_ms,
            )
        )

    def tool(self, name: str, args: dict[str, Any], result: Any, *, ok: bool, elapsed_ms: int) -> None:
        """Record a tool or retrieval call with its arguments and outcome."""
        self.add(
            "tool",
            f"调用 {name}",
            f"参数：{_clip(_json(args), 120)}\n结果：{_clip(_json(result), 160)}",
            ok=ok,
            elapsed_ms=elapsed_ms,
        )

    def think(self, why: str) -> None:
        self.add("think", "判断依据", why)

    def decide(self, what: str, why: str = "") -> None:
        self.add("decide", f"决定：{what[:50]}", why)

    def reject(self, what: str, why: str) -> None:
        self.add("reject", f"打回：{what[:50]}", why, ok=False)

    # ---------------------------------------------------------------- output

    def to_list(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._steps]

    def summary_line(self) -> str:
        total_ms = int((time.perf_counter() - self._t0) * 1000)
        parts = [f"{len(self._steps)} 步"]
        if self._dropped:
            parts.append(f"省略 {self._dropped} 步")
        parts.append(f"{total_ms / 1000:.1f}s")
        return " · ".join(parts)


def _clip(text: str, limit: int = SNIPPET) -> str:
    """Trim a string without cutting a surrogate pair.

    Newlines are preserved for long-form text (reasoning) because a plan is
    structured, and flattening it to one line loses that structure. Short
    snippets are flattened so a table of steps stays visually tidy.
    """
    text = (text or "").strip()
    if limit > SNIPPET:
        # Long-form: keep line structure, collapse runs of blank lines.
        import re as _re

        text = _re.sub(r"\n{3,}", "\n\n", text)
    else:
        text = text.replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _json(value: Any, limit: int = SNIPPET) -> str:
    import json

    try:
        return _clip(json.dumps(value, ensure_ascii=False, default=str), limit)
    except (TypeError, ValueError):
        return _clip(str(value), limit)


#: Registry of live traces, keyed by (run_id, agent). Module-level because the
#: context that records is created deep inside an agent, while the pipeline is what
#: attaches the result to a message.
_live: dict[str, AgentTrace] = {}


def current(run_id: str, agent: str) -> AgentTrace:
    """Get or create the trace for this agent in this run."""
    key = f"{run_id}:{agent}"
    if key not in _live:
        _live[key] = AgentTrace(agent)
    return _live[key]


def peek(run_id: str, agent: str) -> list[dict[str, Any]]:
    """Read an agent's steps WITHOUT draining them.

    Used by the run status endpoint: the UI polls once a second, and the only way
    to show reasoning that is still being written is to read the live trace.
    `take` is for when the message is finalised.
    """
    trace = _live.get(f"{run_id}:{agent}")
    return trace.to_list() if trace else []


def peek_all(run_id: str) -> dict[str, list[dict[str, Any]]]:
    """Every live trace for a run, keyed by agent. Empty after the run ends."""
    out: dict[str, list[dict[str, Any]]] = {}
    for key, trace in list(_live.items()):
        if not key.startswith(f"{run_id}:"):
            continue
        out[key.split(":", 1)[1]] = trace.to_list()
    return out


def upsert(
    run_id: str,
    agent: str,
    kind: str,
    title: str,
    detail: str = "",
    *,
    ok: bool = True,
    elapsed_ms: int = 0,
) -> None:
    """Write a step that may repeat, replacing the previous one of that kind.

    Streaming emits the same logical step many times as text arrives. Appending
    each emission filled the capped trace with near-identical rows and pushed out
    the steps that mattered. Replacing keeps one growing row, which is what a
    reader expects of "思考中…".
    """
    trace = current(run_id, agent)
    for existing in trace._steps:
        # Match on kind, and on the title only when the kind can legitimately
        # repeat (tool calls, retrieval). Reasoning and think rows are unique per
        # model call, so the title must not be part of the match: two code paths
        # worded it differently ("模型思考（进行中）" vs "模型思考过程") and the
        # merge silently failed, showing one thought twice.
        if existing.kind != kind:
            continue
        if kind in {"reasoning", "think"}:
            existing.title = title[:60]
            existing.detail = _clip(detail, REASONING_LIMIT)
            existing.elapsed_ms = elapsed_ms
            existing.at = time.time()
            return
        if existing.title.startswith(title[:12]):
            existing.detail = _clip(detail, SNIPPET)
            existing.elapsed_ms = elapsed_ms
            existing.at = time.time()
            return
    trace.add(kind, title, detail, ok=ok, elapsed_ms=elapsed_ms)


def take(run_id: str, agent: str) -> list[dict[str, Any]]:
    """Read and reset an agent's accumulated steps, so each message owns its own."""
    key = f"{run_id}:{agent}"
    trace = _live.pop(key, None)
    return trace.to_list() if trace else []


def clear(run_id: str) -> None:
    """Drop every trace for a run, called when the run ends."""
    for key in [k for k in _live if k.startswith(f"{run_id}:")]:
        _live.pop(key, None)

#: What each agent is doing right now, keyed by run and then by agent.
#:
#: Why this is separate from the step list: a step records something that
#: HAPPENED, and with streaming switched off nothing happens for a minute at a
#: time — one model call is a single silent block. The UI then has no event to
#: render and the run looks stuck, which is exactly the complaint this answers.
#:
#: Why keyed by agent and not one entry per run: segments run concurrently
#: (`pipeline_six` gathers one task per segment), so several agents are busy at
#: the same moment. The old single dict was overwritten on every call, leaving
#: the UI showing whichever agent spoke last — "3 个并发执行" in the counter, one
#: row in the list. Each entry carries `holds`, a count of the callers currently
#: keeping it alive, because two segments can be inside the same agent at once
#: and the first one out must not erase the other's row.
_WORKING: dict[str, dict[str, dict[str, Any]]] = {}


def mark_working(
    run_id: str,
    agent: str,
    label: str = "",
    *,
    progress: tuple[int, int] | None = None,
    keep_label: bool = False,
) -> None:
    """Say that `agent` started doing `label` just now.

    `progress` is `(done, total)` of the agent's own stage list — the only honest
    percentage available for a sub-agent. `keep_label` lets a caller that only
    wants to hold the entry open (the LLM wrapper) avoid replacing the richer
    label the pipeline already set.
    """
    if not run_id:
        return
    book = _WORKING.setdefault(str(run_id), {})
    entry = book.get(agent)
    if entry is None:
        entry = {"agent": agent, "label": label, "started_at": time.time(), "holds": 0, "progress": None}
        book[agent] = entry
    elif label and not keep_label:
        entry["label"] = label
    entry["holds"] = int(entry.get("holds") or 0) + 1
    if progress is not None:
        entry["progress"] = (int(progress[0]), int(progress[1]))


def mark_progress(run_id: str, agent: str, done: int, total: int) -> None:
    """Update the stage counter without touching the `holds` count.

    Kept separate from `mark_working` on purpose: this runs between two paired
    mark/clear calls, so bumping `holds` here would leave the row stuck on screen
    after the agent finished.
    """
    if not run_id:
        return
    entry = (_WORKING.get(str(run_id)) or {}).get(agent)
    if entry is not None:
        entry["progress"] = (int(done), int(total))


def clear_working(run_id: str, agent: str | None = None) -> None:
    """Release the heartbeat for one agent, or for every agent when omitted.

    Called when the call returns, however it returns. Release rather than delete:
    a second segment may still be inside the same agent.
    """
    if not run_id:
        return
    book = _WORKING.get(str(run_id))
    if not book:
        return
    if agent is None:
        _WORKING.pop(str(run_id), None)
        return
    entry = book.get(agent)
    if entry is None:
        return
    entry["holds"] = int(entry.get("holds") or 0) - 1
    if entry["holds"] <= 0:
        book.pop(agent, None)
    if not book:
        _WORKING.pop(str(run_id), None)


def _shape_working(entry: dict[str, Any], now: float) -> dict[str, Any]:
    started = float(entry.get("started_at") or 0.0)
    shaped: dict[str, Any] = {
        "agent": entry.get("agent") or "",
        "label": entry.get("label") or "",
        "started_at": started,
        # Computed here rather than in the browser: a client whose clock is
        # minutes off would otherwise show a negative duration for a task that is
        # plainly running.
        "seconds": max(0, int(now - started)),
    }
    progress = entry.get("progress")
    if progress:
        shaped["progress"] = {"done": int(progress[0]), "total": int(progress[1])}
    return shaped


def peek_workers(run_id: str) -> list[dict[str, Any]]:
    """Every agent busy right now, newest first. Empty when nothing runs."""
    book = _WORKING.get(str(run_id)) or {}
    now = time.time()
    entries = sorted(book.values(), key=lambda item: float(item.get("started_at") or 0), reverse=True)
    return [_shape_working(entry, now) for entry in entries]


def peek_working(run_id: str) -> dict[str, Any]:
    """The single busiest agent — kept for callers that show only one row.

    `peek_workers` is the list; this returns its newest entry so the existing
    conversation UI keeps working unchanged.
    """
    workers = peek_workers(run_id)
    return workers[0] if workers else {}
