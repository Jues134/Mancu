"""How strictly a review judges the work, and what it is allowed to block.

Three modes, one decision table. The point of this module is that the mode is not
a word in a prompt: it is the input to a deterministic function that decides
PASS/REJECT and which findings block. A model asked to "review strictly" is
unpredictable; a model that reports severities and a rule that maps severity +
mode to a verdict is not.

Why severities rather than a pass/fail judgement:

The reviewer kept failing the same material. It reported a CRITICAL issue
meaning "the script lacks a complete three-act structure" on a 400-character
historical fragment, and under the old rule any CRITICAL blocked the run. Nothing
in the pipeline could distinguish "this cannot be delivered" from "this could be
better", so every run died on the second kind. Expressing the finding as a
severity and letting the mode decide what that severity means is what makes
NORMAL deliverable and STRICT demanding, instead of both being impossible.
"""

from __future__ import annotations

import os

STRICT = "STRICT"
NORMAL = "NORMAL"
FAST = "FAST"

MODES = (STRICT, NORMAL, FAST)
# 快速审核是出厂默认：只拦 P0，其余只记录。多数运行是「先出草稿看看」，
# 默认严格会让第一次运行就因为 P1/P2 反复返工，等不到可看的结果。
DEFAULT = FAST

#: Severity levels, highest first. P0 blocks in every mode; P3 never blocks.
P0 = "P0"
P1 = "P1"
P2 = "P2"
P3 = "P3"

SEVERITIES = (P0, P1, P2, P3)

#: Old severities the review prompt and stored reports still use. Mapped rather
#: than rejected so an existing run record keeps working.
_LEGACY_SEVERITY = {
    "CRITICAL": P0,
    "MAJOR": P1,
    "MINOR": P2,
    "SUGGESTION": P3,
}

#: Which severities block, per mode.
#:
#:   P0  fatal — blocks always, in every mode. Safety and unusable output do not
#:       become acceptable because someone asked for a fast draft.
#:   P1  important — blocks in STRICT and NORMAL, not in FAST.
#:   P2  cosmetic — blocks only in STRICT, where the user asked for item-by-item
#:       checking and a list of defects is the deliverable.
#:   P3  suggestion — never blocks. A suggestion that can stop delivery teaches
#:       the reviewer that raising its voice gets results, and the run never ends.
_BLOCKS = {
    STRICT: {P0, P1, P2},
    NORMAL: {P0, P1},
    FAST: {P0},
}


def normalise(value: object) -> str:
    """Read a review mode from user input, tolerating case and stray text."""
    raw = str(value or "").strip().upper()
    if raw in MODES:
        return raw
    return DEFAULT


def from_env() -> str:
    """The deployment-wide default, overridable per run."""
    return normalise(os.environ.get("STUDIO_REVIEW_MODE", ""))


def severity_of(value: object) -> str:
    """Map whatever the reviewer wrote onto the four levels.

    Unknown values become P1 rather than P3: an unrecognised severity is more
    likely a badly-worded real problem than a nitpick, and silently downgrading
    it to "never blocks" is how a broken deliverable passes review.
    """
    raw = str(value or "").strip().upper()
    if raw in SEVERITIES:
        return raw
    if raw in _LEGACY_SEVERITY:
        return _LEGACY_SEVERITY[raw]
    # Tolerate "P1 - important" and similar prose.
    for sev in SEVERITIES:
        if raw.startswith(sev):
            return sev
    return P1


def blocks(mode: str, severity: object) -> bool:
    """Whether this finding stops the run under this mode."""
    return severity_of(severity) in _BLOCKS[normalise(mode)]


def blocking_set(mode: str) -> set[str]:
    return set(_BLOCKS[normalise(mode)])


def describe(mode: str) -> str:
    """One line for the prompt: what this mode accepts and what stops it."""
    mode = normalise(mode)
    if mode == STRICT:
        return (
            "严格审核。逐项核对：历史事实、用户原始资料、用户明确提出的硬性要求、"
            "各阶段之间的一致性。P0、P1、P2 都会阻断。"
            "但轻微美术瑕疵、个人审美差异、非必要优化仍不得阻断——"
            "严格不等于要求绝对完美。"
        )
    if mode == FAST:
        return (
            "快速审核。只判断两件事：用户核心要求是否满足，是否存在明显错误。"
            "只有 P0（无法使用、核心需求完全未满足、重大事实错误、关键内容缺失）阻断。"
            "P1 及以下记录但不返工。不因镜头优化、节奏调整、美感建议触发返工。"
        )
    return (
        "普通审核（默认）。判断结果是否达到可交付标准：用户核心要求是否满足、"
        "剧情逻辑是否成立、分镜是否可执行、角色场景道具是否一致、"
        "是否存在明显影响成片质量的问题。"
        "P0、P1 阻断；P2、P3 记录但不阻断。不得因为「还可以更好」打回。"
    )


def rules(mode: str) -> dict[str, object]:
    """Machine-readable summary for the report and the UI."""
    mode = normalise(mode)
    return {
        "mode": mode,
        "blocks": sorted(_BLOCKS[mode]),
        "summary": describe(mode),
    }


#: Names a picker shows, and the one-line explanation under it.
#:
#: These live next to `_BLOCKS` rather than in the page because both are claims
#: about what the mode does. A UI that keeps its own copy of "P0/P1 打回重做"
#: starts lying the moment the table changes, and the user then picks a mode
#: expecting behaviour this module no longer implements. `blocks` is shipped
#: alongside so the wording can be checked against the rule instead of trusted.
LABELS = {STRICT: "严格审核", NORMAL: "普通审核", FAST: "快速审核"}

_SUMMARY = {
    STRICT: "逐项核对史实、原始资料与硬性要求；P0–P2 都会打回重做",
    NORMAL: "交付标准（默认）；P0、P1 打回重做，P2、P3 只记录",
    FAST: "出草稿用；只有 P0 打回重做，P1 及以下只记录",
}


def options() -> list[dict[str, object]]:
    """The three modes as a picker should list them.

    Order is deliberate: strictest first, so the choice reads as a scale rather
    than an alphabetical dump. `default` marks the one to preselect, which is
    also what a blank request resolves to.
    """
    return [
        {
            "id": mode,
            "label": LABELS[mode],
            "summary": _SUMMARY[mode],
            "blocks": sorted(_BLOCKS[mode]),
            "default": mode == DEFAULT,
        }
        for mode in MODES
    ]
