"""The six-agent implementations for the production spec.

This module holds the implementations declared by the six-agent roster.

Conventions every agent here follows, because the pipeline depends on them:

  · **Return a dict of state keys.** The pipeline merges it into `ctx.state`.
    Returning a fresh dict rather than mutating a shared one is what makes
    per-segment concurrency safe — see the merge note in `script_adaptor_agent`.

  · **Never invent an id.** Ids come from `doc.ids`, generated per run. A model
    echoing `SB-3` for `SB-003` is normalised rather than trusted.

  · **Record what it did.** `ctx.agent_ask` already traces the model call and its
    reasoning. Anything decided outside a model call — a rule check, a mapping —
    gets an explicit `ctx.trace_step` so the UI can answer "why did it decide that".

  · **Degrade, don't crash, when media is withheld.** A run in prompt-only mode
    must still produce its deliverable.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..agents.base import AgentContext
from ..platform.registry import AgentDefinition
from . import media_mode
from . import review_mode
from . import schemas as s
from .ids import format_id


# ============================================================================
# Shared helpers
# ============================================================================


def _system(agent_id: str) -> str:
    """Load an agent's prompt (with the shared production rules in front)."""
    from .prompts import load_prompt

    return load_prompt(agent_id)


def _json_only(instruction: str, schema: str) -> str:
    return f"{instruction}\n\n只输出一个合法 JSON 对象：\n{schema}"


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce a model reply fragment to a dict.

    Models occasionally answer a field with a bare string or a list. Calling
    `.get` on those is the most common crash in this codebase.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}


def _as_list(value: Any, limit: int = 12) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value[:limit]


def _clip(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _shot_hint(target_seconds: float) -> str:
    """A rough shot count for the segment, given in the prompt as a sanity check.

    Six seconds per shot is the middle of the range the storyboard is asked for
    (5–8s), and the point of quoting a number is to catch fragmentation: a 33s
    segment that comes back with 11 shots means shots were split, not written.
    """
    return str(max(1, round(float(target_seconds or 0) / 6)))


def _segment_text(seg: dict[str, Any], limit: int = 2400) -> str:
    return _clip(seg.get("text") or seg.get("content") or "", limit)


def _feedback_brief(feedback: dict[str, Any], agent_id: str) -> str:
    """Render the review's findings as the paragraph a rework prompt needs.

    Returns "" when there is nothing to say, so the caller can interpolate it
    unconditionally without leaving a blank section in the prompt.

    Why the instructions are listed separately from the descriptions: a description
    states what is wrong, an instruction states what to do. Sending only the former
    produced attempts that addressed the symptom — the review said "人物背景缺失"
    and the script came back with a new paragraph of background that still had no
    section structure, because nothing said the sections were missing.

    Filtering is the pipeline's job (it already narrowed the findings to the agent
    it re-dispatched to), so everything here is meant for this agent.
    """
    if not feedback:
        return ""

    issues = [i for i in (feedback.get("issues") or []) if isinstance(i, dict)]
    if not issues:
        return ""

    lines = [
        "",
        f"【上次被打回：{feedback.get('rejected_stage') or '审查'}】",
        "审查指出了以下问题，本次必须逐条解决，不要只改其中一部分：",
    ]
    for i, issue in enumerate(issues[:6], 1):
        kind = str(issue.get("type") or "").strip()
        desc = str(issue.get("description") or "").strip()
        fix = str(issue.get("repair_instruction") or "").strip()
        lines.append(f"{i}. {f'[{kind}] ' if kind else ''}{desc}")
        if fix:
            lines.append(f"   应这样改：{fix}")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# 2. Script adaptor — 编剧 + 历史考据 + 资料改编
# ============================================================================

#: 改编模式的代号 → 中文说明。这是给用户看的翻译表：模型在 mode 字段里只回
#: 代号（它自己的工作分类），界面上直接显示 MODE_B 对读者是天书。代号保留在
#: 括号里，因为运维排查时需要拿着代号去查提示词和日志。
MODE_LABELS = {
    "MODE_A": "整理（忠实重排原文，不加创作）",
    "MODE_B": "资料改编（以史料为底本，按拍摄需要重新组织）",
    "MODE_C": "梗概扩写（材料只有要点，由模型补足成片）",
    "MODE_D": "原创（材料仅作背景，内容由模型创作）",
    "MODE_E": "修改（在上一次稿子的基础上改）",
}


def _mode_label(mode: str) -> str:
    """人话 + 代号一起给。没命中表时说明是模型自创的说法，原样保留。"""
    mode = (mode or "").strip()
    if mode in MODE_LABELS:
        return f"{MODE_LABELS[mode]}（{mode}）"
    return mode or "未声明"

#: The model's reply shape. A literal rather than a pydantic model because the
#: section list is open-ended: a model returning an extra field must not make the
#: whole response fail validation.
SCRIPT_ADAPTOR_SCHEMA = """{
  "mode": "MODE_B",
  "title": "",
  "content_type": "",
  "core_theme": "",
  "preservation_rules": ["必须保留的原文或事实"],
  "historical_facts": [{"fact": "", "certainty": "史实|存疑"}],
  "unverified_facts": ["无法核实的内容"],
  "creative_elements": ["属于艺术创作的内容"],
  "sections": [
    {
      "section_id": "SEC-001",
      "section_title": "",
      "narrative_purpose": "",
      "time_period": "",
      "location": "",
      "scene_description": "",
      "narration": ["旁白"],
      "dialogue": [{"character": "", "line": ""}],
      "events": ["事件"],
      "sound_notes": ["声音要求"],
      "transition": ""
    }
  ],
  "era": {"era": "", "year_range": "",
          "required_visuals": [], "forbidden_visuals": [],
          "character_rules": [], "color_palette": ""},
  "issues": [],
  "adaptation_notes": []
}"""


async def script_adaptor_agent(
    ctx: AgentContext, definition: AgentDefinition, options: dict[str, Any]
) -> dict[str, Any]:
    """Adapt one segment's source text into a shootable script.

    Two outputs from one call, on purpose:

      · `adapted_script` — the script the storyboard stage cuts into shots.
      · `constraints` — the era prohibitions the shot prompts must respect.

    Asking one agent for both outputs forces it to state the era its own script
    belongs to. The prohibitions still travel separately because the shot stage
    consumes them as a guard list.
    """
    segments = ctx.state.get("segments") or []
    if not segments:
        raise ValueError("没有分段可改编（segments）")

    index = int(options.get("segment_index") or 0)
    seg = segments[index] if 0 <= index < len(segments) else segments[0]
    target_seconds = int(options.get("target_seconds") or 60)

    # A rework must be told what was rejected.
    #
    # Before this was wired, a rejected script was re-asked the identical question
    # and produced another rejected answer — the review's findings never reached
    # the agent, so "fix it" meant "try again blindly".
    feedback = options.get("rework_feedback") or {}
    feedback_block = _feedback_brief(feedback, "script_adaptor")

    if feedback:
        ctx.trace_step(
            "decide",
            f"按审查意见返工（上次在 {feedback.get('rejected_stage', '')} 被退回）",
            feedback_block or str(feedback.get("summary") or "")[:120],
            ok=False,
        )

    # Retrieval grounds the historical assertions in the source rather than in the
    # model's general impression of "an old era".
    passage = await ctx.retrieve(
        str(seg.get("title") or "") + " " + _segment_text(seg, 60), top_k=3
    )

    # This segment's slice of the whole-film requirement list. Passing it in is
    # what keeps the adaptor from writing an "era" list that belongs to four
    # segments at once; the block below also states what this segment is *not*
    # responsible for, because "not mine" has to be said out loud to be obeyed.
    assigned = [
        r for r in ((ctx.state.get("requirements_by_segment") or {}).get(
            str(seg.get("id") or index))
            or [])
        if isinstance(r, dict)
    ]
    requirement_block = ""
    if assigned:
        requirement_block = (
            "本段必须完成的必现项（只有这些，别段的不由本段负责）：\n"
            + "\n".join(
                f"- [{r.get('requirement_id')}] {r.get('requirement')}"
                + (f"（验收：{r.get('acceptance')}）" if r.get("acceptance") else "")
                for r in assigned[:8]
            )
            + "\n"
        )

    raw = await ctx.agent_ask(
        "script_adaptor",
        system=_system("script_adaptor"),
        user=_json_only(
            f"""请把下面这段材料改编成可拍摄的脚本。

段落标题：{seg.get('title', '')}
目标时长：{target_seconds} 秒
正文：
{_segment_text(seg)}

{requirement_block}
{f'相关原文段落：{chr(10)}{passage}' if passage else ''}
{feedback_block}
要求：
- 先判断属于哪种改编模式（MODE_A 整理 / MODE_B 资料改编 / MODE_C 梗概扩写 /
  MODE_D 原创 / MODE_E 修改），并在 mode 字段说明
- 按节（section）组织，每节有明确的叙事目的
- era 字段只写**本段**的：required_visuals 是本段必须拍到的画面，
  forbidden_visuals 是时代穿帮的禁项。**不要把整部片子的必现清单抄进来**，
  也不要把别段的事件写进本段的 required_visuals
- 无法核实的内容写进 unverified_facts，不要混进 historical_facts
- 属于艺术创作的写进 creative_elements
- sections 不得为空：至少一节，否则后续分镜无从开始""",
            SCRIPT_ADAPTOR_SCHEMA,
        ),
        max_tokens=4200,
    )

    sections: list[dict[str, Any]] = []
    for i, item in enumerate(_as_list(raw.get("sections"), 8)):
        d = _as_dict(item)
        sections.append(
            {
                "section_id": _clip(d.get("section_id"), 20) or f"SEC-{i + 1:03d}",
                "section_title": _clip(d.get("section_title"), 40),
                "narrative_purpose": _clip(d.get("narrative_purpose"), 80),
                "time_period": _clip(d.get("time_period"), 30),
                "location": _clip(d.get("location"), 40),
                "scene_description": _clip(d.get("scene_description"), 200),
                "narration": [_clip(x, 160) for x in _as_list(d.get("narration"), 6)],
                "dialogue": [
                    {
                        "character": _clip(_as_dict(x).get("character"), 20),
                        "line": _clip(_as_dict(x).get("line"), 120),
                    }
                    for x in _as_list(d.get("dialogue"), 6)
                    if isinstance(x, (dict, str))
                ],
                "events": [_clip(x, 80) for x in _as_list(d.get("events"), 6)],
                "sound_notes": [_clip(x, 60) for x in _as_list(d.get("sound_notes"), 4)],
                "transition": _clip(d.get("transition"), 60),
            }
        )

    if not sections:
        # A script with no sections cannot be cut into shots — but throwing the
        # whole reply away loses work that IS usable.
        #
        # The rejected run demonstrated the cost: the model returned a real script
        # (mode, era constraints, historical facts) with an empty `sections` list,
        # and the pipeline discarded all of it and failed the run. Nothing was
        # shown to the user and the prompt only said "改编结果没有 sections".
        #
        # A synthesised section from the material is a defensible outcome: the
        # storyboard can cut it, the reader sees the script that was written, and
        # the defect is recorded as a note instead of a crash.
        # The salvage text must be the whole material, not a 400-character slice of
        # it. That slice is what produced the review finding
        #     "剧本在关键处中断，句子在'斧头'处不完整"
        # — the reviewer was reading a sentence cut in half by the salvage code
        # itself, not by the model. A truncated fallback manufactures the very
        # defect the fallback exists to avoid.
        #
        # `scene_description` and `core_theme` are used only when the model wrote
        # them; otherwise the source segment carries the real text and is passed
        # whole, bounded by the same generous ceiling used for a normal segment.
        detail = str(raw.get("scene_description") or "").strip()
        theme = str(raw.get("core_theme") or "").strip()
        source = _segment_text(seg, 6000)
        salvage = detail or theme or source
        if len(salvage) < 200 and len(source) > len(salvage):
            # A one-line summary is not a script: prefer the material, which can
            # actually be cut into shots.
            salvage = source
        sections = [
            {
                "section_id": "SEC-001",
                "section_title": _clip(raw.get("title"), 40) or "全篇",
                "narrative_purpose": "（改编未给出分节，按材料整体作为一节）",
                "time_period": "",
                "location": "",
                "scene_description": salvage,
                "narration": [],
                "dialogue": [],
                "events": [],
                "sound_notes": [],
                "transition": "",
            }
        ]
        ctx.trace_step(
            "warn",
            "改编结果没有 sections，已按整体降级为一节",
            "剧本的其他部分（模式、时代约束、史实）已保留；"
            "分镜将基于这一节展开。若需要更细的段落，请重新改编。",
            ok=False,
        )

    adapted = {
        "script_id": f"{seg.get('id', 'seg')}-script",
        "mode": _clip(raw.get("mode"), 20),
        "title": _clip(raw.get("title"), 60) or _clip(seg.get("title"), 60),
        "content_type": _clip(raw.get("content_type"), 30),
        "target_duration": target_seconds,
        # The source is echoed but never overwritten: the spec is explicit that
        # `source_content` belongs to the user and no agent may replace it.
        "source_content": _segment_text(seg),
        "preservation_rules": [_clip(x, 90) for x in _as_list(raw.get("preservation_rules"), 8)],
        "historical_facts": [
            {
                "fact": _clip(_as_dict(x).get("fact"), 110),
                "certainty": _clip(_as_dict(x).get("certainty"), 10) or "存疑",
            }
            for x in _as_list(raw.get("historical_facts"), 10)
            if isinstance(x, (dict, str))
        ],
        "unverified_facts": [_clip(x, 110) for x in _as_list(raw.get("unverified_facts"), 8)],
        "creative_elements": [_clip(x, 90) for x in _as_list(raw.get("creative_elements"), 8)],
        "sections": sections,
        "issues": [_clip(x, 100) for x in _as_list(raw.get("issues"), 6)],
        "adaptation_notes": [_clip(x, 90) for x in _as_list(raw.get("adaptation_notes"), 6)],
        # Recorded so a reviewer can tell a clean adaptation from a salvaged one.
        "degraded": not _as_list(raw.get("sections"), 8),
    }

    # Merge into the existing map in place. Each segment runs concurrently and
    # holds its own copy; returning a fresh dict would let the last finisher
    # overwrite the others' entries.
    scripts = ctx.state.get("adapted_script")
    if not isinstance(scripts, dict):
        scripts = {}
        ctx.state["adapted_script"] = scripts
    scripts[str(seg.get("id") or index)] = adapted

    # ---- era constraints, in the shape the shot stage already reads ----
    era_raw = _as_dict(raw.get("era"))

    # 必现画面以**导演的分配**为准（上面已取到本段那一份）。以前这里直接用模型
    # 返回的 era.required_visuals，而提示词问的是「这个**时代**必须出现什么」——
    # 于是每一段都写出一份几乎相同的时代清单（望志路、南陈北李…），分镜和审查照着
    # 它验收，撤离段被要求拍开会，两次返工一字不差，撞上限停止。
    # 模型只在导演没分配时才保留这份自查结果。
    if assigned:
        required_visuals = [
            _clip(r.get("requirement"), 60) for r in assigned if r.get("requirement")
        ]
        requirement_source = "director"
    else:
        required_visuals = [
            _clip(x, 40) for x in _as_list(era_raw.get("required_visuals"), 8)
        ]
        requirement_source = "self"

    era = s.EraConstraint(
        era=_clip(era_raw.get("era"), 40),
        year_range=_clip(era_raw.get("year_range"), 30),
        required_visuals=required_visuals,
        forbidden_visuals=[_clip(x, 40) for x in _as_list(era_raw.get("forbidden_visuals"), 10)],
        character_rules=[_clip(x, 60) for x in _as_list(era_raw.get("character_rules"), 5)],
        color_palette=_clip(era_raw.get("color_palette"), 40),
        note=_clip(era_raw.get("note"), 120),
    )
    constraint = s.HistoryConstraint(
        segment_id=str(seg.get("id")),
        era=era,
        key_figures=[],
        hard_facts=[
            _clip(f.get("fact") if isinstance(f, dict) else f, 80)
            for f in _as_list(raw.get("historical_facts"), 6)
        ],
    )

    constrains = ctx.state.get("constraints")
    if not isinstance(constrains, dict):
        constrains = {}
        ctx.state["constraints"] = constrains
    # 旧字段一律保留（era.required_visuals 仍是最常读的那份），新增的两个字段
    # 让审查能按 requirement_id 追溯，也让它知道这一段的边界在哪。
    dumped = constraint.model_dump()
    dumped["requirement_ids"] = [str(r.get("requirement_id") or "") for r in assigned]
    dumped["requirements"] = assigned
    dumped["requirement_source"] = requirement_source
    constrains[constraint.segment_id] = dumped

    # The two facts a reader will ask about: which mode it chose, and whether the
    # history is actually verified. Neither is visible from the script length.
    # 每个词都写全：这是给非开发者看的台账，不是内部缩写。
    ctx.trace_step(
        "decide",
        f"改编完成：{_mode_label(adapted['mode'])}",
        f"脚本分 {len(sections)} 节"
        f"（每节是一段有独立叙事目的的戏）"
        f"；有史料依据、直接采用的 {len(adapted['historical_facts'])} 条，"
        f"无法核实、单独标出以免冒充史实的 {len(adapted['unverified_facts'])} 条，"
        f"属于艺术创作（场景氛围、镜头设计等）的 {len(adapted['creative_elements'])} 处",
    )

    return {
        "adapted_script": scripts,
        "constraints": constrains,
        "script_summary": (
            f"{_mode_label(adapted['mode'])}：{len(sections)} 节；"
            f"导演要求本段必须拍到的画面 {len(era.required_visuals)} 项，"
            f"这个年代不许出现（防穿帮）的画面 {len(era.forbidden_visuals)} 项"
        ),
    }



# ============================================================================
# 3. Consistency — 人物 / 场景 / 道具 / 声音资产
# ============================================================================

#: 一致性阶段只有一个交付物：完整的图片提示词清单。人物、场景和道具必须在
#: 同一次模型调用里返回，之后程序才会展示清单并决定是否进入出图队列。
CONSISTENCY_ASSET_SCHEMA = """{
  "style_guide": {"style_prompt": "photorealistic cinematic film still, 35mm film grain",
                  "negative_prompt": "anime, illustration, plastic skin",
                  "palette": "暗褐、靛青"},
  "character_assets": [
    {"name": "孙中山", "age": "40岁上下",
     "appearance": "长脸，颧骨略高，短发，西装",
     "four_view_prompt": "横向角色设定板，左侧面部近景，右侧正面/侧面/背面全身",
     "negative_prompt": "现代服饰，雷同脸",
     "is_main": true}
  ],
  "scene_assets": [
    {"name": "海外旧式旅馆房间", "time_of_day": "夜晚", "location": "海外",
     "layout": "木桌靠窗，油灯在桌上", "lighting": "暖色油灯侧光，背景昏暗",
     "prompt": "夜晚的旧式旅馆房间，木桌、油灯、窗外街景"}
  ],
  "prop_assets": [
    {"name": "旧报纸", "material": "泛黄纸张", "shape": "竖排版中文报",
     "era_note": "民国初年报纸版式", "prompt": "桌上一份泛黄的竖排中文旧报纸"}
  ],
  "continuity_rules": ["孙中山全程同一身西装", "旅馆房间布局不变"],
  "missing_assets": []
}"""


async def consistency_agent(
    ctx: AgentContext, definition: AgentDefinition, options: dict[str, Any]
) -> dict[str, Any]:
    """Build the unified visual assets for one segment, plus its voice plan.

    Voice is decided here rather than in the video stage because a voice is an
    *asset*: it belongs to a character, it must stay consistent across shots, and
    the user supplies some of them by upload. Deciding it at generation time would
    mean re-deriving it per shot and letting two shots of one character disagree.

    The uploads come from `ctx.brief["voice_uploads"]`, recorded at submission.
    A character with no sample is marked `model` — that is the documented
    fallback, not a failure.
    """
    boards = ctx.state.get("storyboards") or {}
    segments = ctx.state.get("segments") or []
    if not boards:
        raise ValueError("没有分镜表（storyboards）")

    index = int(options.get("segment_index") or 0)
    board = _select_board(boards, segments, index)
    shots = board.get("shots") or []
    script = _select_script(ctx.state.get("adapted_script"), segments, index)

    # Everything the imagery must respect. Passing it to the art director is what
    # stops a portrait being drawn in modern clothing: the constraint list is the
    # only place that knowledge lives.
    constraint = (ctx.state.get("constraints") or {}).get(str(board.get("segment_id") or "")) or {}
    era = _as_dict(constraint.get("era")) if isinstance(constraint.get("era"), dict) else {}

    # ---- partial rebuild: only the assets the review named ----
    #
    # A segment's assets are four portraits, a few scenes and some props. Rebuilding
    # all of them because one portrait was wrong is the "整段 4 份四视图重做" cost.
    # When the review names asset ids, this pass regenerates those and keeps the
    # rest — including their already-rendered reference images, which are the
    # expensive part.
    #
    # Only meaningful when a manifest already exists; on a first pass there is
    # nothing to preserve and `partial_ids` stays empty.
    feedback = options.get("rework_feedback") or {}
    previous_manifest = (ctx.state.get("asset_index") or {}).get(
        str(board.get("segment_id") or "")
    ) or {}
    partial_ids = [str(x) for x in (feedback.get("asset_ids") or []) if str(x).strip()]
    if partial_ids and previous_manifest:
        known_ids = {
            str(asset.get(id_key) or "")
            for id_key, bucket in (
                ("character_id", previous_manifest.get("character_assets") or []),
                ("scene_id", previous_manifest.get("scene_assets") or []),
                ("prop_id", previous_manifest.get("prop_assets") or []),
            )
            for asset in bucket
        }
        partial_ids = [x for x in partial_ids if x in known_ids]
    else:
        partial_ids = []

    characters_in_shots: list[str] = []
    for shot in shots:
        for name in shot.get("characters") or []:
            if isinstance(name, str) and name.strip():
                characters_in_shots.append(name.strip())

    segment_id = str(board.get("segment_id") or "")
    cast_names = "、".join(dict.fromkeys(characters_in_shots)) or "（未标注）"
    shot_lines = chr(10).join(
        f"- {_clip(x.get('visual_description'), 80)}" for x in shots[:6]
    )
    era_lines = (
        f"- 必现画面元素：{'、'.join(era.get('required_visuals') or []) or '（无）'}"
        f"{chr(10)}- 绝对不能出现：{'、'.join(era.get('forbidden_visuals') or []) or '（无）'}"
        f"{chr(10)}- 人物规则：{'、'.join(era.get('character_rules') or []) or '（无）'}"
    )
    script_excerpt = ""
    if script and script.get("sections"):
        script_excerpt = (
            f"剧本节选：{chr(10)}"
            f"{_clip(script.get('sections')[0].get('scene_description'), 200)}"
        )

    # A targeted rebuild still uses this one request. The model is told which ids
    # may change; `_merge_manifest` below keeps every unnamed asset and its image.
    previous_char_id_set = {
        str(a.get("character_id") or "") for a in (previous_manifest.get("character_assets") or [])
    }
    previous_set_ids = {
        str(a.get(k) or "")
        for k, key in (("scene_id", "scene_assets"), ("prop_id", "prop_assets"))
        for a in (previous_manifest.get(key) or [])
    }
    cast_partial = [x for x in partial_ids if x in previous_char_id_set]
    set_partial = [x for x in partial_ids if x in previous_set_ids]

    ctx.trace_step(
        "decide",
        "编写完整图片提示词",
        "一次返回人物、场景和道具的全部图片提示词；这里只调用文本模型，"
        "提示词展示完成前不会判断出图开关，也不会调用生图接口",
    )

    partial_instruction = ""
    if partial_ids:
        partial_instruction = (
            f"\n- 本次是局部返工，只重写这些资产：{'、'.join(partial_ids)}。"
            "未点名的资产不要重新描述；对应类别没有返工项时返回空数组。"
            f"{_partial_asset_block(cast_partial + set_partial, previous_manifest)}"
        )

    raw = _as_dict(
        await ctx.agent_ask(
            "consistency",
            system=_system("consistency"),
            user=_json_only(
                f"""请一次性写完这一段的全部图片提示词，包括统一风格、人物、场景和道具。

段落：{segment_id}
出现的人物：{cast_names}

镜头内容：
{shot_lines}

时代约束：
{era_lines}

{script_excerpt}

要求：
- style_prompt 必须是英文摄影术语；中文会画出二次元插画
- palette 用中文色名，2-4 个
- 每个主要人物给出四视图提示词（正面/90度侧面/背面全身 + 面部近景），
  四张必须是同一个人，服装发型一致
- 明确写出容易穿帮、必须避免的元素（穿在人物身上就是穿帮）
- 只列出这段真正需要的场景与道具，不要凑数
- 每个场景写清 time_of_day / location / layout / lighting，prompt 必须与统一风格一致
- 道具要写清材质、形制、年代特征；一旦在画面里会被看清，就必须描述得能画出来
- 画面里将要出现的文字（横幅、报刊标题、门牌）写进该资产或场景的 prompt，
  并注明「文字不可辨认时按模糊色块处理」，不要让模型去拼写中文字
- 必须在这一次回复里同时返回人物、场景和道具，不要拆步骤{partial_instruction}""",
                CONSISTENCY_ASSET_SCHEMA,
            ),
            max_tokens=12000,
        )
    )

    # A partial rebuild must keep the established style even when the model only
    # returns the named asset. The actual asset splice happens after ids are
    # normalised below.
    previous_style = _as_dict(previous_manifest.get("style_guide"))
    if previous_style.get("style_prompt") and not _as_dict(
        raw.get("style_guide")
    ).get("style_prompt"):
        raw["style_guide"] = previous_style

    # ---- style ----
    style_raw = _as_dict(raw.get("style_guide"))
    style = s.StyleBible(
        style_prompt=_ensure_english(style_raw.get("style_prompt")),
        negative_prompt=_clip(
            style_raw.get("negative_prompt") or s.StyleBible().negative_prompt, 300
        ),
        palette=_clip(style_raw.get("palette"), 60),
    )

    # ---- characters ----
    #
    # Repeated names are collapsed. The model emits variants of one person
    # ("孙中山" and "孙中山的手"), and each variant would otherwise become its own
    # portrait and its own cast entry, so the film would show several people who
    # are all one man.
    # On a partial rebuild the model is asked to re-describe only the named
    # assets, but it may drop or rename them. Ids are therefore matched back by
    # name against the previous manifest before a fresh one is issued: reusing
    # `CHAR-002` is what lets the merge recognise "this is the same character,
    # re-drawn" instead of appending a fifth portrait.
    previous_char_ids = _safe_previous_ids(
        ctx, str(board.get("segment_id") or ""),
        "character_assets", "character_id", canonical=True,
    )
    seq_char = _sequence_for(ctx, "character", "character_assets")
    seen: set[str] = set()
    characters: list[dict[str, Any]] = []
    for item in _as_list(raw.get("character_assets"), 8):
        d = _as_dict(item)
        name = _canonical_name(_clip(d.get("name"), 40))
        if not name or name in seen:
            continue
        seen.add(name)
        characters.append(
            {
                "character_id": previous_char_ids.get(name) or seq_char.next(),
                "name": name,
                "age": _clip(d.get("age"), 20),
                "appearance": _clip(d.get("appearance"), 160),
                "four_view_prompt": _clip(d.get("four_view_prompt"), 900),
                "negative_prompt": _clip(d.get("negative_prompt"), 200),
                "is_main": bool(d.get("is_main")),
            }
        )

    # ---- prompts are a deliverable, with or without pictures ----
    #
    # Whether the user ticked 出图 decides whether these prompts get rendered,
    # not whether they get written. A card with an empty prompt field is the
    # one thing a prompt-mode run must never ship, and the model does omit the
    # four-view often enough to matter. The fallback is written back onto the
    # asset — not only onto the task — so the asset card and the queue agree.
    for c in characters:
        if not c["four_view_prompt"]:
            c["four_view_prompt"] = _fallback_four_view(c)

    # ---- scenes ----
    previous_scene_ids = _safe_previous_ids(
        ctx, str(board.get("segment_id") or ""),
        "scene_assets", "scene_id",
    )
    seq_scene = _sequence_for(ctx, "scene", "scene_assets")
    scenes = [
        {
            "scene_id": previous_scene_ids.get(_clip(_as_dict(x).get("name"), 40))
            or seq_scene.next(),
            "name": _clip(_as_dict(x).get("name"), 40),
            "time_of_day": _clip(_as_dict(x).get("time_of_day"), 20),
            "location": _clip(_as_dict(x).get("location"), 40),
            "layout": _clip(_as_dict(x).get("layout"), 120),
            "lighting": _clip(_as_dict(x).get("lighting"), 80),
            "prompt": _clip(_as_dict(x).get("prompt"), 600),
        }
        for x in _as_list(raw.get("scene_assets"), 6)
    ]

    # ---- props ----
    previous_prop_ids = _safe_previous_ids(
        ctx, str(board.get("segment_id") or ""),
        "prop_assets", "prop_id",
    )
    seq_prop = _sequence_for(ctx, "prop", "prop_assets")
    props = [
        {
            "prop_id": previous_prop_ids.get(_clip(_as_dict(x).get("name"), 30))
            or seq_prop.next(),
            "name": _clip(_as_dict(x).get("name"), 30),
            "material": _clip(_as_dict(x).get("material"), 40),
            "shape": _clip(_as_dict(x).get("shape"), 50),
            "era_note": _clip(_as_dict(x).get("era_note"), 60),
            "prompt": _clip(_as_dict(x).get("prompt"), 400),
        }
        for x in _as_list(raw.get("prop_assets"), 6)
    ]

    # Same rule as the portraits: a scene or prop without a prompt cannot be
    # rendered and cannot be copied out, so it is not allowed to stay empty.
    for sc in scenes:
        if not sc["prompt"]:
            sc["prompt"] = _fallback_scene(sc)
    for pr in props:
        if not pr["prompt"]:
            pr["prompt"] = _fallback_prop(pr)

    # ---- image tasks: one per asset that needs a reference picture ----
    seq_img = _sequence_for(ctx, "image", "image_generation_tasks")
    image_tasks: list[dict[str, Any]] = []
    for c in characters:
        image_tasks.append(
            {
                "image_task_id": seq_img.next(),
                "asset_id": c["character_id"],
                "asset_type": "character_four_view",
                "image_model": _image_model(ctx),
                "image_prompt": c["four_view_prompt"] or _fallback_four_view(c),
                "negative_prompt": c["negative_prompt"] or style.negative_prompt,
                "aspect_ratio": "1:1",
                "status": "pending",
            }
        )
    for sc in scenes:
        image_tasks.append(
            {
                "image_task_id": seq_img.next(),
                "asset_id": sc["scene_id"],
                "asset_type": "scene",
                "image_model": _image_model(ctx),
                "image_prompt": sc["prompt"],
                "negative_prompt": style.negative_prompt,
                "aspect_ratio": "16:9",
                "status": "pending",
            }
        )
    for pr in props:
        image_tasks.append(
            {
                "image_task_id": seq_img.next(),
                "asset_id": pr["prop_id"],
                "asset_type": "prop",
                "image_model": _image_model(ctx),
                "image_prompt": pr["prompt"],
                "negative_prompt": style.negative_prompt,
                "aspect_ratio": "1:1",
                "status": "pending",
            }
        )

    # ---- pictures are not drawn here any more ----
    #
    # This stage used to render its assets on the spot. With two segments in
    # flight that drew the same person twice, in two different faces, and billed
    # it twice — and three submissions per segment at once is what the image
    # endpoints rate-limit. Rendering now belongs to the run-level serial queue
    # (`asset_queue.render_all`), which walks one table for the whole film, one
    # submission at a time. The prompts produced above are the deliverable
    # either way; the switch only decides whether they get drawn.

    # ---- voices ----
    #
    # Bound to characters by id first, then by name: the user uploads before any
    # run exists, so on a first project there is no id to match and the name is
    # the only link available.
    from .voice import plan_voices, unbound_uploads

    uploads = list((ctx.brief or {}).get("voice_uploads") or [])
    voice_plan = plan_voices(characters, uploads, start_at=1)
    unmatched = unbound_uploads(characters, uploads)
    if unmatched:
        # Surfaced rather than ignored: a user who misspelled a character name
        # believes they supplied a voice, and the film would come back in the
        # model's voice with no explanation.
        ctx.trace_step(
            "warn",
            f"{len(unmatched)} 个声音样本未匹配到角色",
            "、".join(unmatched[:5]),
            ok=False,
        )

    voices = voice_plan.to_dicts()

    # ---- merge per segment ----
    manifest = {
        "segment_id": str(board.get("segment_id") or ""),
        "character_assets": characters,
        "scene_assets": scenes,
        "prop_assets": props,
        "voice_profiles": voices,
        "image_generation_tasks": image_tasks,
        "style_guide": style.model_dump(),
        "continuity_rules": [_clip(x, 90) for x in _as_list(raw.get("continuity_rules"), 8)],
        "missing_assets": [_clip(x, 80) for x in _as_list(raw.get("missing_assets"), 6)],
    }

    # A saved run may already contain an id collision from an older version.
    # `_safe_previous_ids` deliberately gives the conflicting asset a fresh id;
    # update the storyboard at the same boundary so downstream video generation
    # does not keep pointing at the old, ambiguous id.
    _rebind_board_asset_ids(board, previous_manifest, manifest)

    # ---- splice a partial rebuild into the previous manifest ----
    #
    # Named assets are replaced; every other asset keeps its entry AND its already
    # rendered reference image. Appending instead would duplicate a character that
    # already existed — which is how one person ended up with four portraits — and
    # dropping the unnamed ones would re-bill work that had already passed review.
    if partial_ids and previous_manifest:
        manifest = _merge_manifest(previous_manifest, manifest, partial_ids)
        ctx.trace_step(
            "decide",
            f"局部重建 {len(partial_ids)} 项资产",
            f"只重建 {('、'.join(partial_ids))}；其余资产沿用上一版（含已生成的参考图）",
        )

    # ---- one table for the whole film ----
    #
    # Register after the partial merge, so what is registered is the manifest
    # that will actually be used. An asset another segment already owns is
    # adopted here: same prompt, same picture. Without this, two segments each
    # describe 孙中山 in their own words and the film ends up with two men who
    # share a name — the defect that reads as a continuity bug but is really a
    # duplicated asset.
    from . import asset_queue

    linked = asset_queue.link(
        ctx, manifest, segment_index=int(options.get("segment_index") or 0)
    )
    if linked["reused_image"] or linked["reused_prompt"]:
        ctx.trace_step(
            "decide",
            f"复用前序段资产：图片 {linked['reused_image']} 张、"
            f"提示词 {linked['reused_prompt']} 条",
            "同一个人/场景/道具在全片只保留一套提示词和一张参考图；"
            "重复出图会得到两张不一样的脸，也付两次钱",
        )

    ctx.state.setdefault("asset_index", {})[manifest["segment_id"]] = manifest

    # The first segment establishes the shared style; later segments reuse it so
    # two segments cannot drift into different looks.
    if not ctx.state.get("art_bible"):
        ctx.state["art_bible"] = {
            "style": style.model_dump(),
            "characters": [
                {
                    "character_id": c["character_id"],
                    "name": c["name"],
                    "appearance": c["appearance"],
                    "portrait_prompt": c["four_view_prompt"],
                }
                for c in characters
            ],
        }

    ctx.trace_step(
        "decide",
        f"资产：人物 {len(characters)} / 场景 {len(scenes)} / 道具 {len(props)}",
        f"四视图 {len(characters)} 份，图片任务 {len(image_tasks)} 条；"
        f"上传声音 {len(voice_plan.bound())} 个，模型生成 {len(voice_plan.generated())} 个",
    )

    return {
        "asset_manifest": manifest,
        "asset_index": ctx.state["asset_index"],
        "art_bible": ctx.state["art_bible"],
        "consistency_summary": (
            f"人物 {len(characters)} 位（四视图 {len(characters)} 份）、"
            f"场景 {len(scenes)} 个、道具 {len(props)} 件；"
            f"声音：上传 {len(voice_plan.bound())} / 模型 {len(voice_plan.generated())}；"
            f"图片提示词 {len(image_tasks)} 条已完成，等待后续判断是否出图"
        ),
    }


# ---- helpers used by the agents above ----


def _partial_rewrite_block(
    shot_ids: list[str],
    previous_shots: list[dict[str, Any]],
    feedback: dict[str, Any],
) -> str:
    """The instruction that turns a rework into a surgical edit.

    Returns "" for a normal (full) authoring pass, so the caller can interpolate
    it unconditionally.

    Two things travel here that a plain "fix the storyboard" never conveyed:

      · **which shots** — so the model rewrites two instead of thirteen;
      · **what the neighbours look like** — because the replacement has to chain
        (start_state continues the previous end_state, link_type agrees with the
        scene change). Without them the model re-invents the chaining and the
        fixed board no longer connects to the shots that already passed review.

    The neighbour block is explicitly marked read-only: a model shown the full
    board tends to "improve" it while it is there, which is how a rework meant to
    touch two shots comes back having re-planned the whole segment.
    """
    if not shot_ids:
        return ""

    by_id = {str(s.get("shot_id") or ""): s for s in previous_shots}
    ordered = [str(s.get("shot_id") or "") for s in previous_shots]
    lines: list[str] = [
        "",
        "【这是返工：只重写被点名的镜头】",
        f"只需要重写这些镜头：{'、'.join(shot_ids)}。"
        "其余镜头保持原样，不要重新规划。",
        "",
        "被点名镜头的当前版本（要替换掉的内容）：",
    ]
    for shot_id in shot_ids:
        shot = by_id.get(shot_id) or {}
        lines.append(
            f"- {shot_id}｜{float(shot.get('duration') or 0):.0f} 秒｜"
            f"{_clip(shot.get('visual_description'), 90)}"
        )
        if shot.get("end_state"):
            lines.append(f"    当前结尾状态：{_clip(shot.get('end_state'), 80)}")

    neighbours: list[str] = []
    for shot_id in shot_ids:
        if shot_id not in ordered:
            continue
        at = ordered.index(shot_id)
        for pos in (at - 1, at + 1):
            if not 0 <= pos < len(ordered):
                continue
            neighbour_id = ordered[pos]
            if neighbour_id in shot_ids or neighbour_id in neighbours:
                continue
            neighbours.append(neighbour_id)
    if neighbours:
        lines.append("")
        lines.append("相邻镜头（只读，不得改写，新版本必须与它们接得上）：")
        for neighbour_id in neighbours:
            shot = by_id.get(neighbour_id) or {}
            lines.append(
                f"- {neighbour_id}｜{float(shot.get('duration') or 0):.0f} 秒｜"
                f"{_clip(shot.get('visual_description'), 70)}"
            )
            if shot.get("end_state"):
                lines.append(f"    结尾状态：{_clip(shot.get('end_state'), 70)}")
            if shot.get("start_state"):
                lines.append(f"    起始状态：{_clip(shot.get('start_state'), 70)}")

    issues = [i for i in (feedback.get("issues") or []) if isinstance(i, dict)]
    if issues:
        lines.append("")
        lines.append("审查指出的问题（只改这些，不要顺带重写没被点名的镜头）：")
        for issue in issues[:6]:
            shot_of = str(issue.get("shot_id") or "")
            desc = str(issue.get("description") or "").strip()
            fix = str(issue.get("repair_instruction") or "").strip()
            prefix = f"[{shot_of}] " if shot_of else ""
            lines.append(f"- {prefix}{desc}" + (f" → {fix}" if fix else ""))

    lines.extend((
        "",
        "输出要求：",
        f"1. storyboard 数组**只包含** {'、'.join(shot_ids)} 这几个镜头，"
        "不要输出其他镜头，也不要重新编号。",
        "2. 每镜仍然要满足下面全部写作要求（时间轴节拍、景别、人物命名、时长上限）。",
        "3. 新版本的 start_state 必须承接上一镜（只读）的 end_state；"
        "如果上一镜是场景切换，first_frame_strategy 用 scene_reference。",
        "4. duration_analysis 可以省略或按被改写镜头重算，其余字段照常给。",
        "",
    ))
    return "\n".join(lines)


def _merge_partial_shots(
    previous: list[dict[str, Any]], patched: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Splice rewritten shots into the previous board, keeping the rest intact.

    Only shots that already existed are replaced; a rewritten shot carrying an
    unknown id is dropped rather than appended. Appending would let a model that
    ignored "do not renumber" silently grow the segment — which is exactly how a
    60-second brief turned into a 98-second board.

    Order follows the previous board: the shots that were not rewritten define the
    sequence, so a retry cannot reorder the segment as a side effect.
    """
    replacements = {
        str(s.get("shot_id") or ""): s
        for s in patched
        if str(s.get("shot_id") or "")
    }
    merged: list[dict[str, Any]] = []
    for old in previous:
        shot_id = str(old.get("shot_id") or "")
        merged.append(replacements.get(shot_id) or old)
    # A replacement's `index` must match its new position, or downstream payloads
    # (which read `index` for ordering) would disagree with the list order.
    for position, shot in enumerate(merged):
        shot["index"] = position
    return merged


def _partial_asset_block(
    asset_ids: list[str], previous_manifest: dict[str, Any]
) -> str:
    """The instruction that turns an asset rework into a targeted rebuild.

    Returns "" for a first pass. A model asked to "fix the assets" re-lists the
    whole cast, and the manifest is then rebuilt wholesale — four portraits to
    correct one, each a paid image call.

    The existing assets are shown as locked context because the rebuilt one has to
    stay consistent with them: a new portrait of 邓小平 must match the 邓小平 that
    already appears in the shots that were not touched.
    """
    if not asset_ids or not previous_manifest:
        return ""

    wanted = set(asset_ids)
    lines: list[str] = [
        "",
        "",
        "【这是返工：只重建被点名的资产】",
        f"只需要重建：{'、'.join(asset_ids)}。其余资产已经生成并通过审查，不要重做，也不要重复列出。",
    ]

    existing: list[str] = []
    for key, id_key, label in (
        ("character_assets", "character_id", "人物"),
        ("scene_assets", "scene_id", "场景"),
        ("prop_assets", "prop_id", "道具"),
    ):
        for asset in (previous_manifest.get(key) or []):
            asset_id = str(asset.get(id_key) or "")
            if not asset_id or asset_id in wanted:
                continue
            name = str(asset.get("name") or "")
            existing.append(f"- [{asset_id}] {label} {name}")
    if existing:
        lines.append("")
        lines.append("已存在、必须保持一致的资产（只读参考，不要重新描述）：")
        lines.extend(existing[:10])

    lines.append("")
    lines.append(
        "输出要求：character_assets / scene_assets / prop_assets 里**只列出**要重建的那几项，"
        "其余留空数组即可；已经存在的资产会在合并时自动保留。"
        "新提示词必须与上面已存在的资产风格、服装、光线一致。"
    )
    return "\n".join(lines)


def _merge_manifest(
    previous: dict[str, Any],
    rebuilt: dict[str, Any],
    asset_ids: list[str],
) -> dict[str, Any]:
    """Keep every asset that was not named, replacing only the ones that were.

    The three buckets are keyed differently (`character_id` / `scene_id` /
    `prop_id`), so the merge walks them explicitly rather than pretending one key
    fits all.

    A rebuilt asset takes the *new* prompt but inherits the *old* rendered image
    only if it is not one of the named ones — a named asset's picture is exactly
    what the reviewer asked to be redone, so its stale `image_path` must not
    survive into the next render. For unnamed assets the old image is preserved,
    which is the whole saving: those pictures were already paid for.

    Order follows the previous manifest so the asset ids stay stable across a
    rework — a shot references `CHAR-001` by id, and renumbering on every retry
    would silently repoint every reference.
    """
    named = set(asset_ids)
    merged = dict(previous)

    for key, id_key in (
        ("character_assets", "character_id"),
        ("scene_assets", "scene_id"),
        ("prop_assets", "prop_id"),
    ):
        fresh = {
            str(item.get(id_key) or ""): item
            for item in (rebuilt.get(key) or [])
            if str(item.get(id_key) or "")
        }
        fresh_by_name = {
            str(item.get("name") or "").strip(): item
            for item in (rebuilt.get(key) or [])
            if str(item.get("name") or "").strip()
        }
        kept: list[dict[str, Any]] = []
        seen: set[str] = set()
        for old in (previous.get(key) or []):
            asset_id = str(old.get(id_key) or "")
            old_name = str(old.get("name") or "").strip()
            seen.add(asset_id)
            same_id = fresh.get(asset_id)
            same_id_name = str((same_id or {}).get("name") or "").strip()
            if asset_id in named and old_name in fresh_by_name:
                # Repairing a historical collision intentionally changes the bad
                # id. Match the replacement by its stable name, otherwise the old
                # ambiguous entry survives and the new one is merely appended —
                # leaving the review failure in place forever.
                # Name matching comes before id matching on purpose. In the bad
                # record CHAR-006 names both 毛泽东 and 与会代表; a dict lookup by
                # id alone can only return one of them and would replace both rows
                # with the same person.
                replacement = fresh_by_name[old_name]
                kept.append(replacement)
                seen.add(str(replacement.get(id_key) or ""))
            elif asset_id in named and same_id is not None and (
                not old_name or not same_id_name or old_name == same_id_name
            ):
                kept.append(same_id)
                seen.add(str(same_id.get(id_key) or ""))
            else:
                kept.append(old)
        # An asset the rebuild introduced that the old manifest never had (a new
        # character discovered in a rewritten shot) is appended rather than
        # dropped, as long as it was not one of the named replacements.
        for asset_id, item in fresh.items():
            if asset_id in seen:
                continue
            kept.append(item)
        merged[key] = kept

    # Image tasks are rebuilt from the merged asset buckets: a kept asset that
    # already has its picture needs no task, while a named one needs a new job.
    #
    # The union of both sides matters. Walking only `rebuilt` silently dropped the
    # tasks of assets that were not re-described, and their pictures — already paid
    # for — disappeared from the manifest with them.
    previous_tasks = {
        str(t.get("asset_id") or ""): t
        for t in (previous.get("image_generation_tasks") or [])
        if str(t.get("asset_id") or "")
    }
    rebuilt_tasks = {
        str(t.get("asset_id") or ""): t
        for t in (rebuilt.get("image_generation_tasks") or [])
        if str(t.get("asset_id") or "")
    }
    valid_asset_ids = {
        str(item.get(id_key) or "")
        for key, id_key in (
            ("character_assets", "character_id"),
            ("scene_assets", "scene_id"),
            ("prop_assets", "prop_id"),
        )
        for item in (merged.get(key) or [])
        if str(item.get(id_key) or "")
    }
    tasks: list[dict[str, Any]] = []
    for asset_id in dict.fromkeys([*previous_tasks, *rebuilt_tasks]):
        if asset_id not in valid_asset_ids:
            # The asset was reissued under a new id while repairing a collision;
            # its stale generation task must not keep the ambiguous id alive.
            continue
        previous_task = previous_tasks.get(asset_id) or {}
        if asset_id not in named and (
            previous_task.get("image_uri") or previous_task.get("local_path")
        ):
            # Untouched and already rendered: keep the finished job as-is.
            tasks.append(previous_task)
            continue
        tasks.append(rebuilt_tasks.get(asset_id) or previous_task)
    merged["image_generation_tasks"] = tasks

    # The style guide is global to the segment; a rebuild must not repaint it.
    merged["style_guide"] = previous.get("style_guide") or rebuilt.get("style_guide") or {}
    # Voices belong to characters, not to a rebuild, and a re-derived plan could
    # reassign a user's uploaded sample to the wrong person.
    merged["voice_profiles"] = previous.get("voice_profiles") or rebuilt.get("voice_profiles") or []
    merged["segment_id"] = rebuilt.get("segment_id") or previous.get("segment_id") or ""
    return merged


def _select_board(boards: dict[str, Any], segments: list[dict[str, Any]], index: int) -> dict[str, Any]:
    """Pick this segment's storyboard by id, not by position.

    Segments run concurrently, so "the first value" is whichever finished first —
    using it would render segment 3 with segment 1's shots.
    """
    if not boards:
        return {}
    if 0 <= index < len(segments):
        sid = str(segments[index].get("id") or "")
        if sid in boards:
            return _as_dict(boards[sid])
    return _as_dict(next(iter(boards.values()), {}))


def _select_script(scripts: Any, segments: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if not isinstance(scripts, dict) or not scripts:
        return {}
    if 0 <= index < len(segments):
        sid = str(segments[index].get("id") or "")
        if sid in scripts:
            return _as_dict(scripts[sid])
    return {}


def _index_of_segment(segments: list[dict[str, Any]], segment_id: str) -> int:
    """Where this segment sits in the plan, looked up by id.

    A reviewer must see **its own** segment's script. The digest used to ask for
    position 0 regardless, so every review was handed segment 1's script — and
    reviewing seg3 (the withdrawal) it was told the segment "completely lacks the
    Zhezhi Road meeting", which belongs to seg1. No rewrite could ever satisfy
    that; the segment burned both retries and stopped. Position is only a
    fallback for runs whose storyboard carries no segment id.
    """
    for i, seg in enumerate(segments):
        if str(seg.get("id") or "") == str(segment_id):
            return i
    return 0


def _sequence_for(ctx: AgentContext, family: str, state_key: str):
    """An id sequence seeded from what this run has already issued.

    Seeding matters on a resumed run: restarting at 001 would hand out ids that
    already belong to existing assets, and a downstream reference would point at
    different content.
    """
    from .ids import is_valid, sequence

    # All concurrently running segments must draw from the SAME live sequence.
    # Previously every call built a private counter and only seeded it from a
    # top-level list that does not exist for character/scene/prop assets (they are
    # nested under `asset_index`). Segment 2 and segment 3 consequently both
    # started at CHAR-006 and assigned that id to different people. A retry then
    # preserved the bad id in the name of stability, making the review impossible
    # to pass.
    registry = ctx.state.setdefault("__id_sequences", {})
    current = registry.get(family)
    if current is not None:
        return current

    existing: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str):
            if is_valid(family, value):
                existing.append(value)
            return
        if isinstance(value, dict):
            for key, child in value.items():
                if key != "__id_sequences":
                    collect(child)
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                collect(child)

    # Scan the whole durable state, including every segment manifest and every
    # downstream reference. References count too: issuing an id that is still
    # mentioned by a saved shot would silently repoint that shot.
    collect(ctx.state)
    current = sequence(family, existing)
    registry[family] = current
    return current


def _asset_name(value: Any, *, canonical: bool = False) -> str:
    name = str(value or "").strip()
    return _canonical_name(name) if canonical else name


def _safe_previous_ids(
    ctx: AgentContext,
    segment_id: str,
    bucket: str,
    id_key: str,
    *,
    canonical: bool = False,
) -> dict[str, str]:
    """Ids this segment may safely retain from its previous manifest.

    Existing projects can contain the old cross-segment collision. Assets in
    already-approved segments are immutable during a targeted retry, so the
    segment being rebuilt yields the contested id and receives a new one. The
    same named asset may legitimately share an id across segments; only an id
    attached to a *different* name is considered a collision.

    Duplicate ids inside the current manifest are handled deterministically: the
    first name keeps the id and later, different names are reissued.
    """
    manifests = ctx.state.get("asset_index") or {}
    previous = _as_dict(manifests.get(segment_id))

    outside: dict[str, set[str]] = {}
    for other_segment, manifest in manifests.items():
        if str(other_segment) == str(segment_id):
            continue
        for item in (_as_dict(manifest).get(bucket) or []):
            if not isinstance(item, dict):
                continue
            asset_id = str(item.get(id_key) or "").strip()
            name = _asset_name(item.get("name"), canonical=canonical)
            if asset_id and name:
                outside.setdefault(asset_id, set()).add(name)

    retained: dict[str, str] = {}
    local_owner: dict[str, str] = {}
    for item in (previous.get(bucket) or []):
        if not isinstance(item, dict):
            continue
        asset_id = str(item.get(id_key) or "").strip()
        name = _asset_name(item.get("name"), canonical=canonical)
        if not asset_id or not name:
            continue
        other_names = outside.get(asset_id, set())
        local_name = local_owner.get(asset_id)
        if (other_names and other_names != {name}) or (
            local_name is not None and local_name != name
        ):
            # Do not preserve it. The caller's project-level sequence will issue
            # a fresh id, and `_rebind_board_asset_ids` updates shot references.
            continue
        local_owner.setdefault(asset_id, name)
        retained[name] = asset_id
    return retained


def _rebind_board_asset_ids(
    board: dict[str, Any],
    previous: dict[str, Any],
    current: dict[str, Any],
) -> None:
    """Point saved shots at ids reissued while repairing an old collision."""
    if not previous or not board.get("shots"):
        return

    def by_name(manifest: dict[str, Any], bucket: str, id_key: str, *, canonical: bool = False):
        return {
            _asset_name(item.get("name"), canonical=canonical): str(item.get(id_key) or "")
            for item in (manifest.get(bucket) or [])
            if isinstance(item, dict) and item.get("name") and item.get(id_key)
        }

    old_chars = by_name(previous, "character_assets", "character_id", canonical=True)
    new_chars = by_name(current, "character_assets", "character_id", canonical=True)
    old_scenes = by_name(previous, "scene_assets", "scene_id")
    new_scenes = by_name(current, "scene_assets", "scene_id")
    old_props = by_name(previous, "prop_assets", "prop_id")
    new_props = by_name(current, "prop_assets", "prop_id")

    def remap(old: dict[str, str], new: dict[str, str]) -> dict[str, str]:
        return {
            old[name]: new[name]
            for name in old.keys() & new.keys()
            if old[name] and new[name] and old[name] != new[name]
        }

    scene_ids = remap(old_scenes, new_scenes)
    prop_ids = remap(old_props, new_props)
    char_ids = remap(old_chars, new_chars)

    for shot in board.get("shots") or []:
        if not isinstance(shot, dict):
            continue
        # Names disambiguate the historical bad state where two people had the
        # same old id. Prefer them over a one-to-one old-id map.
        named = [
            new_chars.get(_canonical_name(str(name or "")))
            for name in (shot.get("characters") or [])
        ]
        named = [asset_id for asset_id in named if asset_id]
        if named:
            shot["character_ids"] = list(dict.fromkeys(named))
        else:
            shot["character_ids"] = [
                char_ids.get(str(asset_id), str(asset_id))
                for asset_id in (shot.get("character_ids") or [])
            ]
        scene_id = str(shot.get("scene_id") or "")
        if scene_id in scene_ids:
            shot["scene_id"] = scene_ids[scene_id]
        shot["prop_ids"] = [
            prop_ids.get(str(asset_id), str(asset_id))
            for asset_id in (shot.get("prop_ids") or [])
        ]


def _ensure_english(value: Any) -> str:
    """Keep the style terms English.

    A Chinese style phrase makes the image model produce illustration instead of
    photography. If the model returned Chinese anyway, the English default is kept
    rather than passed through.
    """
    text = _clip(value, 300)
    if not text:
        return s.StyleBible().style_prompt
    # A prompt with no ASCII letters is certainly not the photography vocabulary.
    if not any("a" <= ch.lower() <= "z" for ch in text):
        return s.StyleBible().style_prompt
    return text


def _fallback_four_view(character: dict[str, Any]) -> str:
    """A four-view prompt assembled without the model's help.

    Used when the model returned a character but no prompt for it. Building one
    from the appearance text is better than shipping an empty task: the four-view
    is what makes portraits consistent across shots.
    """
    appearance = character.get("appearance") or character.get("name") or ""
    return (
        "four-view character reference sheet of one single person. "
        "Layout: left a head-and-shoulders close-up of the face; right three "
        "full-body views of the same person — front, side, back. Same face, same "
        "hair and same clothing in all four panels. Plain light grey seamless "
        "background, full body visible head to shoes. "
        "photorealistic cinematic film still, 35mm film grain. "
        f"Subject: {appearance}"
    )


def _fallback_scene(scene: dict[str, Any]) -> str:
    """A set prompt assembled without the model's help.

    Same reason as the portrait fallback: an empty prompt is an empty
    deliverable, and an empty string must never reach the image endpoint —
    it is charged and produces an unrelated picture.
    """
    detail = "；".join(
        str(x) for x in (
            scene.get("location"), scene.get("time_of_day"),
            scene.get("lighting"), scene.get("layout"),
        ) if x
    ) or str(scene.get("name") or "")
    return (
        "empty period establishing shot, no people in frame. "
        "photorealistic cinematic film still, 35mm film grain. "
        f"Scene: {scene.get('name') or ''}（{detail}）"
    )


def _fallback_prop(prop: dict[str, Any]) -> str:
    """A prop prompt assembled without the model's help. See `_fallback_scene`."""
    detail = "；".join(
        str(x) for x in (
            prop.get("material"), prop.get("shape"), prop.get("era_note"),
        ) if x
    )
    return (
        "single object still life on a neutral surface, no people. "
        "photorealistic, period-accurate props. "
        f"Object: {prop.get('name') or ''}（{detail}）"
    )


def _image_model(ctx: AgentContext) -> str:
    """The image model this run will use, for the task record."""
    try:
        from ..doc.generator import Generator

        return str(getattr(Generator.from_settings(), "image_model", "") or "")
    except Exception:  # noqa: BLE001 - a missing model is reported elsewhere
        return ""


def _infer_workflow(input_type: str, operation: str) -> str:
    """Best guess for the workflow letter when the model omitted it.

    The six paths differ in where the material starts, so the input type is the
    only thing worth reading here:

      · a complete script        → A (organise, then shoot)
      · source material/history  → B (adapt, then shoot)
      · an outline or idea       → C (write, then shoot)
      · an existing project      → F (modify)

    An unrecognised type falls back to B. That is the path this pipeline actually
    implements (adapt → shoot), so the record describes reality rather than a
    confidently wrong letter. Nothing downstream branches on this value.
    """
    kind = (input_type or "").upper()
    if "MODIFY" in (operation or "").upper() or "EXISTING" in kind:
        return "F"
    if any(w in kind for w in ("SCRIPT", "FULL_SCRIPT", "SCREENPLAY")):
        return "A"
    if any(w in kind for w in ("IDEA", "OUTLINE", "CONCEPT", "SYNOPSIS")):
        return "C"
    return "B"


#: Names a review model invents for the agents it is reviewing.
#:
#: Observed live: a SCRIPT_REVIEW blamed "content_adaptor", which is not an agent
#: in this roster — the id is `script_adaptor`. The rework then dispatched to a
#: name nothing matched, so the pipeline re-ran every stage instead of the one
#: that needed fixing, and the retry cost a full pass.
#:
#: Keys are lower-cased and compared after stripping non-alphanumerics, so
#: "Content Adaptor", "content-adaptor" and "content_adaptor" all collapse to one.
_AGENT_ALIASES: dict[str, str] = {
    "director": "director",
    "scriptadaptor": "script_adaptor",
    "contentadaptor": "script_adaptor",
    "contentadapter": "script_adaptor",
    "scriptadapter": "script_adaptor",
    "adaptor": "script_adaptor",
    "adapter": "script_adaptor",
    "writer": "script_adaptor",
    "script": "script_adaptor",
    "storyboard": "storyboard",
    "storyboarder": "storyboard",
    "shotplanner": "storyboard",
    "visual": "storyboard",
    "consistency": "consistency",
    "consistencychecker": "consistency",
    "character": "consistency",
    "assetmanager": "consistency",
    "videogen": "video_gen",
    "video": "video_gen",
    "videoagent": "video_gen",
    "generator": "video_gen",
    # Reviewer names are deliberately absent. A review names
    # the agent that must FIX the problem, never itself: mapping one of those back
    # to `quality_review` would send the rework to the reviewer, which has nothing
    # to repair. Unmapped names fall through to the stage-based default instead.
}


def _resolve_agent_name(raw: str, stage: str) -> str:
    """Map whatever a review model called an agent onto a real id.

    Returns "" when the name cannot be mapped, so the caller falls back to its
    stage-based default instead of dispatching to a name nothing matches.

    An unmapped name is worse than no name: the pipeline would re-dispatch to a
    stage that does not exist, match nothing, and re-run the whole queue — a
    rework that silently does everything again, paid for in full.
    """
    name = (raw or "").strip()
    if not name:
        return ""

    # Already a real id?
    for candidate in ("director", "script_adaptor", "storyboard", "consistency",
                      "video_gen", "quality_review"):
        if candidate == name.lower():
            return candidate

    key = "".join(ch for ch in name.lower() if ch.isalnum())
    mapped = _AGENT_ALIASES.get(key)
    if mapped:
        return mapped

    # Unrecognised: the review is talking about an earlier stage, so blame that
    # rather than nothing. A SCRIPT_REVIEW can only be fixed before the
    # storyboard; a VIDEO_REVIEW can only be fixed at the video stage.
    fallback = {
        "SCRIPT_REVIEW": "script_adaptor",
        "STORYBOARD_REVIEW": "storyboard",
        "ASSET_REVIEW": "consistency",
        "VIDEO_REVIEW": "video_gen",
    }.get(stage, "")
    return fallback


def _revision_scope(blocking: list[dict[str, Any]], stage: str) -> list[dict[str, str]]:
    """The narrowest thing that has to change, per blocking issue.

    A rework must be局部: one wrong shot re-renders one shot, one wrong character
    re-renders that character. Saying "redo the stage" is what makes a retry cost
    the whole segment again, so the scope names the unit explicitly.
    """
    out: list[dict[str, str]] = []
    for issue in blocking:
        shot = str(issue.get("shot_id") or "")
        kind = str(issue.get("issue_type") or "")
        if shot:
            scope = f"shot:{shot}"
        elif "character" in kind or "asset" in kind:
            scope = f"assets:{stage}"
        elif stage == "SCRIPT_REVIEW":
            scope = "script:segment"
        elif stage == "STORYBOARD_REVIEW":
            scope = "storyboard:segment"
        else:
            scope = f"{stage.lower().replace('_review', '')}:segment"
        out.append({
            "scope": scope,
            "agent": str(issue.get("responsible_agent") or ""),
            "instruction": str(issue.get("repair_instruction") or ""),
        })
    return out


def _canonical_name(raw: str) -> str:
    """Collapse a model-written character name into one cast member.

    The model emits variants of one person: "孙中山", "孙中山的手", "孙中山（特写）".
    Each variant would become its own portrait and its own cast entry, so the film
    would show several people who are all one man.

    Conservative on purpose — dropping a real second character is worse than
    keeping a duplicate, so only clear modifiers are stripped.
    """
    import re

    text = (raw or "").strip()
    text = re.split(r"['’]s\s|\bof\b|（|\(|\s-\s", text)[0].strip()
    text = re.sub(r"(的手|的背|的侧脸|面部特写|特写|近景)$", "", text).strip()
    return text[:40]



# ============================================================================
# 4. Storyboard — 拆镜 + 时长分析 + 模型适配
# ============================================================================

STORYBOARD_SIX_SCHEMA = """{
  "planning_summary": "用 1-3 句话说明镜头数量、时长分配和合并/拆分依据",
  "duration_analysis": {
    "narration_duration": 0, "dialogue_duration": 0, "action_duration": 0,
    "visual_hold_duration": 0, "transition_duration": 0,
    "overlap_notes": ["旁白与对白同时进行的说明"]
  },
  "estimated_duration": 0,
  "storyboard": [
    {"shot_id": "SB-001", "section_id": "SEC-001", "duration": 5,
     "scene_id": "SCENE-001", "character_ids": ["CHAR-001"], "prop_ids": ["PROP-001"],
     "characters": ["李明远"],
     "requirement_ids": ["REQ-001"],
     "shot_size": "中近景", "camera_angle": "平视", "composition": "",
     "camera_movement": "push_in", "action": "阅读报纸", "expression": "专注转凝重",
     "start_state": "首帧上看得见的状态：人物姿势、朝向、手在哪、道具在哪、光从哪来",
     "end_state": "尾帧上看得见的状态，下一镜要靠它接上",
     "link_type": "CONTINUOUS_ACTION",
     "first_frame_strategy": "prev_last_frame",
     "last_frame_strategy": "chain_to_next",
     "visual_description": "", "narration": "", "dialogue": [],
     "sound_notes": "", "transition_in": "", "transition_out": "",
     "reference_assets": [], "video_model": "",
     "video_prompt": "中近景，平视，镜头缓慢推近。0–2 秒：…… 2–4 秒：…… 4–5 秒：…… 环境变化：…… 声音：环境音——……；动作音——……",
     "generation_notes": []}
  ],
  "continuity_notes": [],
  "generation_risks": []
}"""


def _storyboard_rows(raw: Any) -> list[Any]:
    """Read common model spellings without discarding a bare shot array.

    Providers do not all honour the requested root key. In observed replies the
    same valid deliverable appeared as `storyboard`, `shots`, `shot_list`, a bare
    array (preserved by AgentContext as `items`), or nested under `data/result`.
    Shape variation is not an empty storyboard.
    """
    if isinstance(raw, list):
        return raw
    root = _as_dict(raw)
    candidates: list[Any] = [
        root.get("storyboard"),
        root.get("shots"),
        root.get("shot_list"),
        root.get("items"),
    ]
    for wrapper in (root.get("data"), root.get("result"), root.get("output")):
        nested = _as_dict(wrapper)
        if nested:
            candidates.extend(
                (nested.get("storyboard"), nested.get("shots"), nested.get("items"))
            )

    for value in candidates:
        if isinstance(value, list) and value:
            return value
        if isinstance(value, dict):
            for key in ("storyboard", "shots", "items"):
                rows = value.get(key)
                if isinstance(rows, list) and rows:
                    return rows
    return []


def _canonical_storyboard_rows(rows: list[Any]) -> list[dict[str, Any]]:
    """Normalise harmless field-name drift before completeness checks."""
    aliases = {
        "duration": ("seconds", "duration_seconds", "shot_duration"),
        "video_prompt": ("prompt", "final_prompt", "generation_prompt"),
        "visual_description": ("visual", "description", "scene_description"),
        "camera_movement": ("camera_move", "movement"),
        "shot_size": ("shot_type", "framing"),
        "scene_id": ("scene",),
    }
    out: list[dict[str, Any]] = []
    for item in rows:
        row = dict(_as_dict(item))
        for canonical, alternatives in aliases.items():
            if row.get(canonical) not in (None, "", []):
                continue
            for alternative in alternatives:
                if row.get(alternative) not in (None, "", []):
                    row[canonical] = row[alternative]
                    break
        out.append(row)
    return out


def _incomplete_storyboard_rows(rows: list[dict[str, Any]]) -> list[str]:
    """Return locators for skeleton shots that cannot be shown or rendered.

    An array length alone is not a storyboard. The failure captured in production
    contained seven objects with no prompt and no duration; normalisation turned
    each missing duration into the 10-second ceiling and displayed 70 seconds of
    blank cards. Duration and video_prompt are therefore the minimum acceptance
    contract. Other omissions remain review findings, but these two mean there is
    no generation instruction at all.
    """
    problems: list[str] = []
    for index, row in enumerate(rows):
        label = str(row.get("shot_id") or f"第 {index + 1} 镜")
        missing: list[str] = []
        if _num(row.get("duration")) <= 0:
            missing.append("时长")
        if not str(row.get("video_prompt") or "").strip():
            missing.append("视频提示词")
        if missing:
            problems.append(f"{label} 缺{'、'.join(missing)}")
    return problems


def _segment_requirements_block(constraint: dict[str, Any], era: dict[str, Any]) -> str:
    """这一段的必现项，以及一句「别段的不归你管」。

    后半句是修复的核心：以前每段拿到的是一份「这个时代的必现画面」，四段几乎一样，
    分镜照着拍、审查照着验，撤离段永远不达标。现在只发本段那几条；确实没有分配时
    退回 `era.required_visuals`（导演没给分配的老路径），保证旧数据还能跑。
    """
    reqs = [r for r in (constraint.get("requirements") or []) if isinstance(r, dict)]
    if reqs:
        lines: list[str] = []
        for r in reqs[:8]:
            rid = r.get("requirement_id") or ""
            text = r.get("requirement") or ""
            how = r.get("acceptance") or ""
            lines.append(f"- [{rid}] {text}" + (f"（验收：{how}）" if how else ""))
        lines.append(
            "- 其他分段负责的事件不在本段范围内：不要为了显得完整去拍别段的情节，"
            "也不要因为没拍到别段的内容而扩写。"
        )
        return "\n".join(lines)
    vis = era.get("required_visuals") or []
    return "- " + "、".join(str(v) for v in vis) if vis else "- （无特别要求，按剧本拍）"


async def storyboard_six_agent(
    ctx: AgentContext, definition: AgentDefinition, options: dict[str, Any]
) -> dict[str, Any]:
    """Cut the adapted script into shots, with a duration budget first.

    The order matters: the delay analysis runs before the shot list, and the model
    is asked for the analysis explicitly. Without it the model distributes seconds
    by instinct, and the plan comes back at 90 seconds for a 60-second brief with
    nobody noticing until assembly.

    Narration and dialogue overlap in reality — a narrator talks over an action —
    so they are not summed. The schema asks for an explicit overlap note for
    exactly that reason.
    """
    segments = ctx.state.get("segments") or []
    if not segments:
        raise ValueError("没有分段可分镜（segments）")

    index = int(options.get("segment_index") or 0)
    seg = segments[index] if 0 <= index < len(segments) else segments[0]
    target_seconds = int(options.get("target_seconds") or ctx.state.get("target_seconds") or 60)

    script = _select_script(ctx.state.get("adapted_script"), segments, index)
    sections = script.get("sections") or []
    constraint = (ctx.state.get("constraints") or {}).get(str(seg.get("id"))) or {}
    era = _as_dict(constraint.get("era")) if isinstance(constraint.get("era"), dict) else {}

    # ---- partial rewrite: only the shots the review actually named ----
    #
    # Re-authoring the whole board to fix two bad shots is the single largest
    # avoidable cost in a rework: a 13-shot segment came back at 98 seconds
    # because the retry was asked to "fix the storyboard" and rewrote all of it.
    # A review that names shots (`shot_id` on the finding) lets this stage rewrite
    # exactly those and splice them into the existing board.
    #
    # Neighbours are supplied as read-only context on purpose: a fix often involves
    # the adjacent shot (the one that must now hand over a different end_state).
    # Anything the model returns for an unnamed shot is discarded when merging, so
    # the rewrite can be informed by its surroundings without silently re-planning
    # them.
    feedback = options.get("rework_feedback") or {}
    previous_board = _select_board(ctx.state.get("storyboards") or {}, segments, index)
    previous_shots = list(previous_board.get("shots") or [])
    partial_ids = [str(x) for x in (feedback.get("shot_ids") or []) if str(x).strip()]
    if partial_ids and previous_shots:
        known = {str(s.get("shot_id") or "") for s in previous_shots}
        partial_ids = [x for x in partial_ids if x in known]
    else:
        partial_ids = []

    # What the video model can actually do decides the prompt structure. Read it
    # here rather than at generation time: a storyboard written for a model that
    # supports first frames is unusable by one that does not.
    caps = _video_capabilities(ctx)
    model_name = _video_model(ctx)

    # A shot cannot exceed the model's single-call limit; dividing the target by
    # that ceiling gives the minimum shot count the plan must respect. 10 s is the
    # storyboard's own ceiling (prompts/storyboard.txt §2) — a backend may claim
    # less, never more, or a 15-second claim would silently re-legalise padding.
    per_shot = min(10.0, float(caps.get("max_seconds_per_shot") or 10.0))

    raw = await ctx.agent_ask(
        "storyboard",
        system=_system("storyboard"),
        user=_json_only(
            f"""请把下面的剧本拆成分镜表，先做时长分析再拆镜。
{_partial_rewrite_block(partial_ids, previous_shots, feedback)}

段落：{seg.get('title', '')}
目标总时长：{target_seconds} 秒

剧本：
{_script_digest(script)}

时代约束：
- 时代：{era.get('era') or ''}（{era.get('year_range') or ''}）
- 禁止元素：{'、'.join(era.get('forbidden_visuals') or []) or '无'}

本段必现（只做这些）：
{_segment_requirements_block(constraint, era)}

视频模型：{model_name or '未配置'}
单镜最长：{per_shot:.0f} 秒（必须遵守，否则无法生成）
{f'- 支持首帧：{"是" if caps.get("first_frame") else "否"}' if caps else ''}
{f'- 支持参考音频：{"是" if caps.get("reference_audio") else "否"}' if caps else ''}

要求：
1. 先输出 duration_analysis：旁白/对白/动作/画面停留/转场各自的秒数，
   并在 overlap_notes 说明哪些是同时发生的（旁白与对白重叠时不得重复累加）；
   planning_summary 用 1-3 句话说明镜头数量、时长分配和合并/拆分依据，作为
   完成后展示给用户的规划摘要，不要输出冗长的内部推演
2. 中文旁白按 3.5-5.5 字/秒估算；庄重沉缓的旁白取下限
3. 时长由内容决定，不要凑数。旁白念完所需的时间是下限，动作需要的时间
   才是这一镜的真实长度。内容少就出短片，内容多就出长片，不要为了接近
   某个数字而拖长镜头或砍掉内容。{target_seconds} 秒只是按材料体量估出的
   参考值，与它相差较大是可以接受的：以内容为准，并在 generation_notes
   里说明原因。
4. 每镜 1-{per_shot:.0f} 秒（{per_shot:.0f} 秒是硬上限，含 10 秒；没有下限），
   由内容动态决定：一句完整台词、一个完整动作阶段需要多久，这一镜就多长。
   不得所有镜头同一个时长，不得为凑时长加无意义动作，也不得为缩时长把一句话
   拆到两个镜头里。**镜头结束时那件事必须已经做完**：动作不能停在半路，台词
   不能说到一半就被切断。
5. **先合并，再拆分**：镜头数量不是目标，是代价。相邻两镜只要满足「同 scene_id
   + 机位景别没有实质变化 + 动作没有中断 + 两镜相加 ≤ {per_shot:.0f} 秒」，就必须
   合并成一个镜头（用时间轴分 2-3 拍写），不许劈成两半。允许切分的理由只有：
   换场景、时间跳跃、景别/机位带来新信息、动作阶段结束、情绪需要停顿。
   「想多给一个画面」「想凑镜头数」「觉得 5 秒太短」都不是理由。
   - 纯转场空镜（翻书页、曙光、檐水）**不单独成镜**：同空间的环境细节并进相邻
     镜头的「环境随时间变化」；跨空间的道具特写交给转场或 MONTAGE，不要塞进别的空间
   - 镜头总数参考 {target_seconds}÷6 ≈ {_shot_hint(target_seconds)} 个；
     30 秒内容写出 10 个以上镜头基本就是被拆碎了，回去合并
   - 输出前扫一遍镜头列表：有没有连着的 2-3 秒碎片镜？有就合并
   - **合并的前提是同一个空间**：一个镜头里只允许一个空间。禁止在一镜内写「叠化
     到……」「淡出至黑」「黑屏」「划像」——那是剪辑，不是镜头内容；要换场就切一镜。
     唯一例外是 link_type=MONTAGE，并且要在 transition_in/out 里写清叠化意图
6. 镜头编号用 SB-001 起的稳定编号
7. 每镜的 video_prompt 必须完整独立可用，不能写"参考上一镜"
8. video_prompt 的开头必须是这一镜独有的景别或动作，禁止每镜都用时间地点开场
   （例如"1977年深秋傍晚。"）。时间地点只在场景真正变化时交代一次；同段内不得有
   3 条以上提示词共用同一个开场短语，相邻镜头前 10 个字不得相同
9. video_prompt 必须分四块写，缺一块就算不合格：
   ① 总述：景别 + 机位 + 运镜 + 速度（一句话）
   ② **时间轴**：以秒数开头分节拍，写成「0–1.5 秒：……」「1.5–3 秒：……」
      「3–5 秒：……」；节拍数按总时长定，≤3 秒最多 2 拍、4–6 秒 2 拍、
      7–10 秒 3 拍，绝不超过 3 拍
   ③ 环境随时间变化：光、水、尘埃、影子、烟雾在这一镜里怎么变
   ④ 声音：环境音与动作音分开，并与节拍对齐
   每个节拍都要写到身体部位与力度（"指节泛白"而不是"很用力"）、面部细节
   （眉/眼/嘴/喉结/呼吸至少两样），禁止写"悲伤""激动""表情平静"这类情绪词；
   多人同框时写明站位：谁在画面左/右/中、彼此距离、朝向
10. 有台词时写成 角色名（语气）说："……"，字数上限按镜头时长（3 字/秒，留换气）：
   1-2 秒≤6 字、3 秒≤9 字、4 秒≤12 字、5 秒≤15 字、6 秒≤18 字、8 秒≤24 字、
   10 秒≤30 字；超了就拆镜，不要压缩台词，也不要把一句话拆到两个镜头
11. 需要参考素材的，写清 asset_id 和参考它的什么
12. 至少一半镜头要有人物在画面里
13. 每镜必须写 start_state / end_state / link_type / first_frame_strategy /
   last_frame_strategy：CONTINUOUS_ACTION 的 start_state 要逐字承接上一镜的
   end_state（上一镜结束什么姿势，本镜就从那个姿势开始）；SCENE_TRANSITION、
   TIME_JUMP 和每段第一镜的 first_frame_strategy 必须是 scene_reference，
   不得写 prev_last_frame；段尾镜的 last_frame_strategy 是 terminal
14. 人物要分级填 character_ids，并且**同时填 characters（名字）**，两个字段缺一不可：
   - 主要角色（全片出场 ≥2 镜且至少一镜能看到正脸/近景）→ character_ids 写资产 id
     （取自人物资产表，不得编造），characters 写同名
   - 一次性角色与群演（只出现 1 镜，或全程背影、远景、看不清脸）→ **不写进
     character_ids**，不建三视图、不占参考图名额，但 characters 要写名字，并在
     video_prompt 正文里给一次性文字描述：年龄段 + 衣着 + 体态 + 一个辨识特征
   - 一次性角色也要写清长相，否则模型会随手生成一张脸，观众会把它和主要角色混淆
   - 同一角色在各镜必须用同一个名字；禁止一镜写"中年男人"、另一镜写"监考老师"
     却指同一个人
   - 只写了 id 不写名字，界面和审查都无法判断这一镜里有谁，视为不合格
15. 每一镜的 characters 里的人名，必须**原样**出现在该镜的 video_prompt 正文里：
   - 正文第一次提到他时用全名（与 characters 字面一致），之后才可以简称或代词
   - 不许写成"老周师傅"这种多一个字的变体，也不许全篇只用"他"
   - 原因：参考图注按全名生成（"图片2为老周的人物设定"），正文对不上名字时模型
     无法把图和人对起来，设定稿等于白给
16. 每镜填 requirement_ids：这一镜完成的是「本段必现」里的哪几条（填上面的 REQ 编号）。
   一条必现项可以被多镜完成，但本段每条必现项至少要有 1 镜认领；没有对应关系的镜头
   留空数组。不要填别段的编号，也不要为了凑数把不相关的编号写上去——最终覆盖检查
   就按这个字段核对全片有没有漏
17. 写给「生理上说得通」的人：
   - 女性角色不许写喉结、胡须、胡茬、低沉男声；男性角色不许写哺乳、孕肚这类体征
   - 不确定某个角色性别时，只写不依赖性别的动作和表情（转头、垂眼、握拳、呼吸变快），
     不要为了"细节感"加一个可能错的体征
   - 同理：老人不写奔跑如飞、伤者不写健步、被绑住的人不写挥手——一处反常识会让整个
     画面显得假，而模型不会替你改
18. **需要「明确知道」的信息必须落到逐字文案上**（名称、日期、地点、结论、人事、
   章程、职能这类 fact）。画面本身不会自证，观众看不出"这是哪一年、哪个会"：
   - 成片会把每镜的 `narration` / `dialogue` 逐句烧成硬字幕，所以验收条件里写着
     「字幕标明」「明确标明」「说明」的，**一律把要显示的那句话原样写进本镜
     `narration`**（一行一整句，不要写"见画面""同上"）
   - 能由人物说出来的，写进 `dialogue`
   - 画内文字（横幅、报刊标题、门牌）只作为画面元素写进 `video_prompt`，并注明
     「文字不可辨认时按模糊色块处理」：视频模型画不出规整中文，**不得**把必现信息
     只放在画内文字上
   - 一条必现项如果只有"看起来像"的画面、而旁白/对白/画内文字都没写出那句话，
     视为未完成——审查会按这个口径判 P0""",
            STORYBOARD_SIX_SCHEMA,
        ),
        # A beat-by-beat timeline costs tokens: 400–700 characters per shot where a
        # one-paragraph description cost 150. The old budget (1500 + 90/s) was sized
        # for prose and would have truncated every detailed storyboard mid-JSON,
        # which then triggers the `chat_json` double-and-retry loop — the run gets
        # slower and more expensive instead of better. 140/s leaves the per-shot
        # room, and the 16000 ceiling is the same one `llm.chat_json` retries up to,
        # so a raise here cannot outrun the client's own limit.
        max_tokens=max(6000, min(16000, 2000 + target_seconds * 140)),
        # Keep the SSE transport alive because this model commonly needs more
        # than the gateway's 60-second buffered-response timeout. Deltas stay
        # hidden from the UI; the answer and completed reasoning are committed
        # together only after the full storyboard has arrived.
        stream=True,
        live_trace=False,
        thinking=False,
    )

    rows = _canonical_storyboard_rows(_storyboard_rows(raw))
    incomplete = _incomplete_storyboard_rows(rows)
    if not rows or incomplete:
        # Empty output and skeleton shots are model-format failures, not evidence
        # that the source is too short. Retry once in place with a smaller,
        # explicit request before failing the segment. The already-approved script
        # remains untouched.
        ctx.trace_step(
            "warn",
            "分镜回复为空或只有镜头骨架，原地重试一次",
            (
                "没有识别到镜头"
                if not rows
                else "；".join(incomplete[:4])
            )
            + "；不会退回内容改编",
            ok=False,
        )
        raw = await ctx.agent_ask(
            "storyboard",
            system=_system("storyboard"),
            user=_json_only(
                f"""上一次回复没有提供可识别的镜头数组。请只重新生成本段分镜。

段落：{seg.get('title', '')}
目标时长：{target_seconds} 秒
单镜最长：{per_shot:.0f} 秒

剧本：
{_script_digest(script)}

本段必现：
{_segment_requirements_block(constraint, era)}

必须返回非空 storyboard 数组。每镜至少包含 shot_id、duration、scene_id、
requirement_ids、visual_description、narration、video_prompt、start_state、
end_state、link_type。duration 必须是 1-{per_shot:.0f} 的数字；video_prompt
必须是完整可提交的提示词。planning_summary 用 1-3 句话说明规划依据。
不要只返回镜头编号，不要解释，不要返回空数组。""",
                STORYBOARD_SIX_SCHEMA,
            ),
            max_tokens=max(6000, min(16000, 2000 + target_seconds * 140)),
            stream=True,
            live_trace=False,
            thinking=False,
        )
        rows = _canonical_storyboard_rows(_storyboard_rows(raw))
        incomplete = _incomplete_storyboard_rows(rows)
        if rows and not incomplete:
            ctx.trace_step(
                "decide", "分镜空结果重试成功", f"识别到 {len(rows)} 个镜头"
            )

    if not rows or incomplete:
        detail = "；".join(incomplete[:6]) if incomplete else "没有镜头数组"
        raise ValueError(
            "模型连续两次没有返回可用分镜；已保留剧本。"
            f"问题：{detail}。请重试分镜阶段或检查分镜模型的 JSON 输出"
        )

    raw = _as_dict(raw)

    # ---- duration analysis ----
    planning_summary = _clip(raw.get("planning_summary"), 500)
    da_raw = _as_dict(raw.get("duration_analysis"))
    duration_analysis = {
        "narration_duration": _num(da_raw.get("narration_duration")),
        "dialogue_duration": _num(da_raw.get("dialogue_duration")),
        "action_duration": _num(da_raw.get("action_duration")),
        "visual_hold_duration": _num(da_raw.get("visual_hold_duration")),
        "transition_duration": _num(da_raw.get("transition_duration")),
        "overlap_notes": [_clip(x, 80) for x in _as_list(da_raw.get("overlap_notes"), 6)],
    }

    # ---- shots ----
    seq_sb = _sequence_for(ctx, "storyboard", "storyboards")
    shots: list[dict[str, Any]] = []
    for i, item in enumerate(rows[:24]):
        d = _as_dict(item)
        raw_id = _clip(d.get("shot_id"), 20)
        # The project sequence is authoritative on a normal pass. Models restart
        # at SB-001 for every segment, so accepting any syntactically valid model
        # id created the same cross-segment collision as the old asset allocator.
        # A targeted rewrite is the one exception: its identity must stay stable
        # so `_merge_partial_shots` can replace the original in place.
        if partial_ids and raw_id in partial_ids:
            shot_id = raw_id
            seq_sb.observe(raw_id)
        else:
            shot_id = seq_sb.next()

        shots.append(
            {
                "shot_id": shot_id,
                "segment_id": str(seg.get("id")),
                # The era prohibitions travel with every shot: the prompt builder
                # turns them into a negative list, and a shot without them is the
                # one that shows up wearing a wristwatch.
                "avoid": _as_list(d.get("avoid"), 6) + _as_list(era.get("forbidden_visuals"), 6),
                "section_id": _clip(d.get("section_id"), 20),
                # 1 second is a legitimate shot (§2: the ceiling is 10, there is no
                # floor). The old 2.0 floor silently lengthened every one-beat shot
                # — a cut on a hand touching a letter became a two-second hold — and
                # the storyboard's own estimate was overruled without a trace.
                "duration": max(1.0, min(per_shot, _num(d.get("duration")) or per_shot)),
                "scene_id": _clip(d.get("scene_id"), 20),
                "character_ids": [_clip(x, 20) for x in _as_list(d.get("character_ids"), 6)],
                # Which whole-film requirement this shot delivers. Carried into
                # the artifact so the final coverage check can answer "was REQ-002
                # actually shot?" without re-reading prose.
                "requirement_ids": [
                    _clip(x, 24) for x in _as_list(d.get("requirement_ids"), 6)
                ],
                # Names, alongside ids. Two reasons: the UI can only show a name
                # (`CHAR-001` means nothing to a reader), and `_reference_images`
                # falls back to a name match when the model wrote names but no ids
                # — which was every shot before this field was normalised here.
                "characters": [_clip(x, 24) for x in _as_list(d.get("characters"), 8)],
                "prop_ids": [_clip(x, 20) for x in _as_list(d.get("prop_ids"), 6)],
                "shot_size": _clip(d.get("shot_size"), 20),
                "camera_angle": _clip(d.get("camera_angle"), 20),
                "composition": _clip(d.get("composition"), 80),
                "camera_movement": _clip(d.get("camera_movement"), 30),
                "action": _clip(d.get("action"), 100),
                "expression": _clip(d.get("expression"), 40),
                # The state machine the video stage chains frames by (§9 of the
                # prompt). start/end_state are what the previous shot ended as and
                # this one begins as; link_type decides whether the tail frame may
                # be inherited; the strategies record the plan for both edges.
                "start_state": _clip(d.get("start_state"), 200),
                "end_state": _clip(d.get("end_state"), 200),
                "link_type": _normalise_link(d.get("link_type")),
                "first_frame_strategy": _clip(d.get("first_frame_strategy"), 30),
                "last_frame_strategy": _clip(d.get("last_frame_strategy"), 30),
                "visual_description": _clip(d.get("visual_description"), 300),
                "narration": _clip(d.get("narration"), 200),
                "dialogue": [
                    {
                        "character": _clip(_as_dict(x).get("character"), 20),
                        "line": _clip(_as_dict(x).get("line"), 120),
                    }
                    for x in _as_list(d.get("dialogue"), 6)
                    if isinstance(x, (dict, str))
                ],
                "sound_notes": _clip(d.get("sound_notes"), 60),
                "transition_in": _clip(d.get("transition_in"), 40),
                "transition_out": _clip(d.get("transition_out"), 40),
                "reference_assets": [_clip(x, 40) for x in _as_list(d.get("reference_assets"), 6)],
                "video_model": _clip(d.get("video_model"), 40) or model_name,
                # Kept verbatim: the video stage must not overwrite it, and the
                # review stage compares the adapted prompt against this original.
                #
                # 2400 rather than 1400 because the prompt now carries a beat-by-beat
                # timeline (§8.1 of prompts/storyboard.txt): a 10-second shot with
                # three beats, an environment note and a sound note runs to roughly
                # 500–700 Chinese characters. `_clip` cuts silently, so a tight limit
                # would amputate the sound note — the loss would look like the model
                # forgot to write it.
                "video_prompt": _clip(d.get("video_prompt"), 2400),
                "generation_notes": [_clip(x, 80) for x in _as_list(d.get("generation_notes"), 4)],
                # Legacy keys the downstream stages still read (`_select`,
                # prompt_check, the shot renderer). Keeping them here avoids a
                # second translation layer.
                "index": i,
                "narration_legacy": _clip(d.get("narration"), 200),
            }
        )

    if not shots:  # Defensive: completeness validation above should make this unreachable.
        raise ValueError("分镜规范化后没有留下镜头；已保留剧本")

    # ---- splice a partial rewrite back into the existing board ----
    #
    # Only reached when this pass was a targeted rework. The merge keeps every
    # shot the review did not name, so the segment cannot change length, order or
    # content behind the reviewer's back — the failure the user reported was a
    # 13-shot/98-second board produced while fixing 2 shots.
    partial_note = ""
    if partial_ids and shots:
        before = list(previous_shots)
        shots = _merge_partial_shots(previous_shots, shots)
        replaced = sum(
            1 for old, new in zip(before, shots) if old is not new
        )
        partial_note = (
            f"（返工：只重写 {('、'.join(partial_ids))}，实际替换 {replaced} 镜；"
            f"其余 {len(shots) - replaced} 镜保留原样）"
        )
        ctx.trace_step(
            "decide",
            f"局部重写 {replaced}/{len(shots)} 镜",
            f"只改 {('、'.join(partial_ids))}；未点名的镜头沿用上一版",
        )
        # `duration_analysis` this pass only describes the shots that were
        # rewritten, so presenting it as the board's analysis would misreport a
        # 60-second segment. The narration figure is recomputed from the merged
        # board; the other buckets have no per-shot source to recompute from, so
        # they keep the values the reviewer will still judge the board by.
        narration_chars = sum(
            len(str(x.get("narration") or "")) for x in shots
        )
        if narration_chars:
            duration_analysis = dict(duration_analysis)
            duration_analysis["narration_duration"] = round(narration_chars / 4.5, 1)

    total = sum(x["duration"] for x in shots)
    board = {
        "segment_id": str(seg.get("id")),
        "shots": shots,
        "total_seconds": total,
        "estimated_duration": _num(raw.get("estimated_duration")) or total,
        "planning_summary": planning_summary,
        "duration_analysis": duration_analysis,
        "style_note": "",
        "continuity_notes": [_clip(x, 80) for x in _as_list(raw.get("continuity_notes"), 8)],
        "generation_risks": [_clip(x, 90) for x in _as_list(raw.get("generation_risks"), 6)],
    }

    boards = dict(ctx.state.get("storyboards") or {})
    boards[board["segment_id"]] = board
    # Written back as well as returned. The later stages (consistency, video_gen,
    # the review digest) read `ctx.state["storyboards"]`, so a board that only
    # travelled in the return value would leave those stages looking at the
    # pre-rework shots — the partial rewrite would appear to have done nothing.
    ctx.state["storyboards"] = boards

    # The signed gap against the estimate. Reported for information, NOT as a
    # shortfall: the plan follows the content, so a difference means the estimate
    # from raw character count was crude — not that the storyboard under-delivered.
    # Labelling it as a miss would push a future prompt back toward padding shots
    # to hit a number, which is the behaviour this change removes.
    drift = total - target_seconds
    ctx.trace_step(
        "reasoning",
        "分镜规划依据（结构化摘要）",
        (planning_summary + "\n" if planning_summary else "")
        + f"旁白约 {duration_analysis['narration_duration']:.0f}s；"
        f"对白约 {duration_analysis['dialogue_duration']:.0f}s；"
        f"动作约 {duration_analysis['action_duration']:.0f}s；"
        f"画面停留约 {duration_analysis['visual_hold_duration']:.0f}s；"
        f"转场约 {duration_analysis['transition_duration']:.0f}s"
        + (
            "\n重叠处理：" + "；".join(duration_analysis["overlap_notes"][:3])
            if duration_analysis["overlap_notes"]
            else ""
        ),
    )
    ctx.trace_step(
        "decide",
        f"{len(shots)} 个镜头，共 {total:.0f} 秒{partial_note}",
        f"内容估算 {target_seconds} 秒（本片 {total:.0f} 秒，{drift:+.0f}）；"
        f"旁白 {duration_analysis['narration_duration']:.0f}s / "
        f"对白 {duration_analysis['dialogue_duration']:.0f}s / "
        f"动作 {duration_analysis['action_duration']:.0f}s"
        + (f"；重叠：{'、'.join(duration_analysis['overlap_notes'][:2])}"
           if duration_analysis["overlap_notes"] else ""),
    )

    return {
        "storyboards": boards,
        "shot_count": len(shots),
        "duration_analysis": duration_analysis,
        "storyboard_summary": (
            f"{len(shots)} 个镜头，共 {total:.0f} 秒（内容估算 {target_seconds} 秒）"
        ),
    }


def _script_digest(script: dict[str, Any], limit: int = 1400) -> str:
    """Flatten an adapted script into the text a storyboard model needs.

    Sections in order, each with its purpose and content. Not the raw JSON: the
    model would spend its budget re-reading field names.
    """
    if not script:
        return "（未提供改编脚本，请按原始材料拆分）"

    parts: list[str] = []
    if script.get("title"):
        parts.append(f"标题：{script['title']}")
    if script.get("core_theme"):
        parts.append(f"主题：{script['core_theme']}")

    for section in (script.get("sections") or [])[:6]:
        d = _as_dict(section)
        block = [f"【{d.get('section_title') or d.get('section_id') or '一节'}】"]
        if d.get("narrative_purpose"):
            block.append(f"叙事目的：{d['narrative_purpose']}")
        if d.get("scene_description"):
            block.append(f"场景：{d['scene_description']}")
        for line in (d.get("narration") or [])[:3]:
            block.append(f"旁白：{line}")
        for dl in (d.get("dialogue") or [])[:3]:
            dd = _as_dict(dl)
            if dd.get("line"):
                block.append(f"对白（{dd.get('character') or '人物'}）：{dd['line']}")
        if d.get("transition"):
            block.append(f"转场：{d['transition']}")
        parts.append("\n".join(block))

    return _clip("\n\n".join(parts), limit)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return default


def _video_model(ctx: AgentContext) -> str:
    """The video model configured for this run."""
    try:
        from ..doc.generator import Generator

        return str(getattr(Generator.from_settings(), "video_model", "") or "")
    except Exception:  # noqa: BLE001
        return ""


def _video_capabilities(ctx: AgentContext) -> dict[str, Any]:
    """What the configured video model can do, as far as is known.

    Returns a conservative default rather than raising: a missing capability list
    must not stop a storyboard from being written. The defaults are the *safe*
    answers — no first frame, no reference audio — because assuming a capability
    the model lacks produces a prompt it cannot execute.
    """
    caps = dict(ctx.state.get("video_model_capabilities") or {})
    caps.setdefault("max_seconds_per_shot", 5.0)
    caps.setdefault("first_frame", False)
    caps.setdefault("last_frame", False)
    caps.setdefault("reference_audio", False)
    caps.setdefault("reference_images", False)
    return caps



# ============================================================================
# 5. Video generation — 双提示词 + 模型调用
# ============================================================================

async def video_gen_agent(
    ctx: AgentContext, definition: AgentDefinition, options: dict[str, Any]
) -> dict[str, Any]:
    """Adapt each shot's prompt for the configured model, then render it.

    Two prompts per shot, and the distinction is load-bearing:

      · `storyboard_prompt` — what the storyboard agent wrote. Immutable here.
      · `final_video_prompt` — the same content adapted to the target model.

    The adapted prompt is what gets sent; the original is what gets reviewed and
    shown. Without the pair, a review can only judge the text that was sent, so
    "the shot was poorly described" and "the adapter mangled a good description"
    become indistinguishable.

    Rendering is optional: with media withheld the stage still returns a complete
    task record with prompts, so a prompt-only run has something to deliver.
    """
    boards = ctx.state.get("storyboards") or {}
    segments = ctx.state.get("segments") or []
    if not boards:
        raise ValueError("没有分镜表（storyboards）")

    index = int(options.get("segment_index") or 0)
    board = _select_board(boards, segments, index)
    shots = board.get("shots") or []
    if not shots:

        raise ValueError("分镜表没有镜头")

    segment_id = str(board.get("segment_id") or "")
    if options.get("render_prepared"):
        prepared = (ctx.state.get("__prepared_video_segments") or {}).pop(
            segment_id, None
        )
        if prepared:
            return await _render_video_plan(ctx, prepared)
    constraint = (ctx.state.get("constraints") or {}).get(segment_id) or {}
    era_raw = constraint.get("era") if isinstance(constraint.get("era"), dict) else {}
    era = _clip((era_raw or {}).get("era"), 40)
    banned = [_clip(x, 30) for x in _as_list((era_raw or {}).get("forbidden_visuals"), 8)]

    bible = ctx.state.get("art_bible") or {}
    style_prompt = _clip(_as_dict(bible.get("style")).get("style_prompt"), 200)
    caps = _video_capabilities(ctx)
    model_name = _video_model(ctx) or "未配置"

    # ---- validate prompts before spending anything ----
    #
    # A shot whose prompt asks for something the model cannot do is sent back to
    # the storyboard rather than rendered and judged. Fixing text is free; fixing
    # a rendered clip is a re-generation.
    await _check_prompts(ctx, board, forbidden=banned, era=era)

    # ---- adapt prompts for the model ----
    #
    # The H3 adapter is kept as the concrete implementation of "adapt to this
    # model": its bracket camera vocabulary was measured to work, and rewriting it
    # from scratch would lose that. For any other model the adapter is skipped and
    # the storyboard prompt is used as-is, which is the honest default rather than
    # guessing another model's syntax.
    adapted = await _adapt_prompts(
        ctx, shots, era=era, style=style_prompt, banned=banned, model=model_name
    )

    # ---- voice plan: bound samples, or the model's own voice ----
    manifest = (ctx.state.get("asset_index") or {}).get(segment_id) or {}
    from .voice import plan_voices, unbound_uploads

    characters = manifest.get("character_assets") or []
    uploads = list((ctx.brief or {}).get("voice_uploads") or [])
    voice_plan = plan_voices(characters, uploads)
    voice_payload = voice_plan.for_video_model()

    # Map each shot to the voice sample its characters carry, once, before the
    # render loop. Doing it inside the loop would re-walk the character list per
    # shot, and worse, would look up by shot id in a plan keyed by character.
    ctx.state["__voice_audio_by_shot"] = _map_voice_to_shots(shots, characters, voice_plan)

    # ---- audio: can this model make sound at all? ----
    #
    # This is the point the user asked about: the roster has no voiceover agent, so
    # if the video model cannot produce audio, nothing would voice the narration and
    # the film would come back silent. `plan_audio` decides whether the local TTS
    # chain must take over, and produces the notice that goes on the run record.
    from .audio_plan import capability_from_descriptor, plan_audio

    audio = plan_audio(
        model=model_name,
        capability=capability_from_descriptor(caps),
        user_sampled=len(voice_plan.bound()),
    )
    # When the model cannot make sound, `generate_audio` must be False even though
    # characters exist: asking a video-only model for audio produces either a
    # rejected request or a silent clip, and both look like failures.
    generate_audio = audio.model_capability == "native" and bool(voice_payload.get("generate_audio"))

    ctx.state["audio_plan"] = audio.to_dict()

    if audio.notice:
        # Warn rather than fail: the film is still deliverable, and the user needs
        # to know *how* the sound happened, not that something went wrong.
        ctx.trace_step("warn", "音频能力提示", audio.notice, ok=False)

    # ---- prepare every prompt before deciding whether to render ----
    # `_generator()` reads the user's video switch, so it must not run until the
    # complete prompt list has been written to the conversation stream.
    # Local copies of the rendered reference pictures, keyed the way shots refer
    # to them. `image_path` is the file on disk — the previous field read here was
    # `portrait_url`, which the six-agent consistency stage never sets, so every
    # reference arrived empty and 17/17 shots rendered with no anchor at all.
    scenes = manifest.get("scene_assets") or []
    props = manifest.get("prop_assets") or []

    # The spec separates the two: a generation *task* is TASK-xxxx, the video
    # *asset* it produces is VID-xxxx. They are different things — one task may
    # be retried, and the retry produces a new asset under the same task.
    seq_task = _sequence_for(ctx, "task", "video_generation_tasks")
    seq_video = _sequence_for(ctx, "video", "shot_assets")
    tasks: list[dict[str, Any]] = []
    plans: list[tuple[dict[str, Any], dict[str, Any], str, list[dict[str, Any]]]] = []

    # ---- first-frame chaining, within this segment ----
    #
    # Shots render sequentially, so the previous clip's file is always on disk by
    # the time the next shot starts. The chain is broken explicitly (prev_clip="")
    # on failure and across scene changes — anchoring a new scene to the old one's
    # last picture is how a cut ends up in the wrong room.
    scene_img_by_id = {
        str(x.get("scene_id")): str(x.get("image_path") or "")
        for x in scenes
    }
    char_img_by_id = {
        str(x.get("character_id")): str(x.get("image_path") or "")
        for x in characters
    }
    supports_first_frame = bool(caps.get("first_frame"))

    for shot in shots:
        shot_id = str(shot.get("shot_id") or "")
        original = str(shot.get("video_prompt") or "").strip()
        final = str(adjusted_map_get(adapted, shot_id) or original)

        # Which reference images this shot needs, and what for. A bare URL tells
        # the model nothing; the role is what makes it inherit the face and not
        # the background.
        refs = _reference_images(shot, characters, scenes, props)

        task: dict[str, Any] = {
            "task_id": seq_task.next(),
            "video_asset_id": seq_video.next(),
            "project_id": ctx.brief.get("project_id") or ctx.run_id,
            "shot_id": shot_id,
            "storyboard_agent_id": "storyboard",
            "storyboard_prompt": original,
            "storyboard_prompt_version": "v1",
            "final_video_prompt": final,
            "video_model": model_name,
            "reference_images": refs,
            "reference_audio": voice_payload.get("reference_audio") or [],
            "reference_videos": [],
            # What actually anchored the shot, filled in after the render. A
            # dict rather than a bare path so the kind survives alongside it —
            # "上一镜尾帧" and "人物设定稿" are different promises.
            "first_frame": None,
            "last_frame": None,
            "link_type": str(shot.get("link_type") or ""),
            # Whether the *video model* should produce sound. False when the model
            # cannot, in which case the local TTS chain supplies the narration
            # instead — see the audio note on this stage's summary.
            "generate_audio": generate_audio,
            # How audio is actually being produced, recorded per task so the film
            # can be audited later without re-deriving it from the model name.
            "audio_source": "model" if generate_audio else "tts",
            "generation_parameters": {
                "duration": shot.get("duration"),
                "aspect_ratio": _as_dict(bible.get("style")).get("aspect_ratio") or "16:9",
            },
            "status": "prompt_ready",
            "video_uri": None,
            "audio_uri": None,
            "duration": shot.get("duration") or 0,
            "error": None,
            "retry_count": 0,
        }
        tasks.append(task)
        plans.append((task, shot, final, refs))

    # Await persistence of the user-visible prompt card before reading the video
    # switch or resolving a rendering backend.
    prompt_result = {
        "video_generation_tasks": tasks,
        "shot_assets": list(ctx.state.get("shot_assets") or []),
        "audio_plan": audio.to_dict(),
        "video_summary": f"{len(tasks)} 个镜头提示词已产出，等待生成模式判断",
    }
    on_prompts_ready = options.get("on_prompts_ready")
    if callable(on_prompts_ready):
        await on_prompts_ready(prompt_result)

    prepared = {
        "segment_id": segment_id,
        "plans": plans,
        "scene_img_by_id": scene_img_by_id,
        "char_img_by_id": char_img_by_id,
        "supports_first_frame": supports_first_frame,
        "caps": caps,
        "model_name": model_name,
        "voice_payload": voice_payload,
        "audio": audio.to_dict(),
        "audio_notice": audio.notice,
        "style_prompt": style_prompt,
    }
    if options.get("prepare_only"):
        ctx.state.setdefault("__prepared_video_segments", {})[segment_id] = prepared
        return prompt_result

    return await _render_video_plan(ctx, prepared)


async def _render_video_plan(ctx: AgentContext, prepared: dict[str, Any]) -> dict[str, Any]:
    """Render a reviewed plan serially, using local persistence as the barrier."""
    from .cancellation import requested
    from .frame_chain import SCENE_TRANSITION, effective_link
    from .media_steps import _generator
    from .trace import clear_working, mark_working

    segment_id = str(prepared.get("segment_id") or "")
    plans = list(prepared.get("plans") or [])
    tasks = [plan[0] for plan in plans]
    scene_img_by_id = prepared.get("scene_img_by_id") or {}
    char_img_by_id = prepared.get("char_img_by_id") or {}
    supports_first_frame = bool(prepared.get("supports_first_frame"))
    caps = prepared.get("caps") or {}
    model_name = str(prepared.get("model_name") or "未配置")
    voice_payload = prepared.get("voice_payload") or {}
    audio_payload = prepared.get("audio") or {}
    audio_notice = str(prepared.get("audio_notice") or "")
    style_prompt = str(prepared.get("style_prompt") or "")

    # Provider resolution happens only after the orchestrator has collected and
    # reviewed every segment's prompt bundle.
    gen = _generator(ctx, "video")
    prev_clip = ""
    prev_scene = ""

    for task_index, (task, shot, final, refs) in enumerate(plans, 1):
        shot_id = str(task.get("shot_id") or "")
        scene_id = str(shot.get("scene_id") or "")
        scene_image = scene_img_by_id.get(scene_id) or ""
        char_ids = [str(x) for x in (shot.get("character_ids") or [])]
        char_image = next((char_img_by_id[c] for c in char_ids if char_img_by_id.get(c)), "")
        link = effective_link(str(shot.get("link_type") or ""), scene_id, prev_scene)
        chain_from = "" if link == SCENE_TRANSITION and scene_id != prev_scene else prev_clip
        task["link_type"] = link

        if gen is None:
            task["status"] = "completed"
            continue

        task["status"] = "rendering"
        ctx.trace_step(
            "tool", f"等待视频生成：{shot_id}",
            f"第 {task_index}/{len(tasks)} 个镜头；本地保存成功后才提交下一条",
        )
        ctx.state["render_progress"] = {
            "segment": segment_id, "shot_id": shot_id,
            "index": task_index, "total": len(tasks),
            "kind": "video", "model": model_name,
        }
        if requested(ctx.run_id):
            task["status"] = "cancelled"
            ctx.trace_step("warn", "已取消，跳过生成", shot_id, ok=False)
            continue

        mark_working(ctx.run_id, "video_gen", f"渲染 {shot_id}")
        try:
            url, local_path, frame_kind, frame_path = await _render_shot(
                ctx, gen, {**shot, "link_type": link}, final, refs, style_prompt,
                prev_clip=chain_from, scene_image=scene_image,
                character_image=char_image,
                supports_first_frame=supports_first_frame,
                dialect=str(caps.get("image_dialect") or "by_parameter"),
                max_images=int(caps.get("max_reference_images") or 1),
            )
            task["status"] = "completed"
            task["video_uri"] = url
            task["local_path"] = local_path
            task["first_frame"] = {"kind": frame_kind, "path": frame_path}
            prev_clip = local_path
            prev_scene = scene_id
        except Exception as exc:  # noqa: BLE001 - recorded below, then queue stops
            task["status"] = "failed"
            prev_clip = ""
            task["error"] = _clip(exc, 160)
            kind = _failure_kind(exc)
            ctx.trace_step(
                "reject", f"生成失败（{kind}）", f"{shot_id}：{task['error']}", ok=False
            )
            # Do not submit the next prompt until this shot has a verified local
            # file.  The remaining prompt-ready tasks stay available for retry.
            break
        finally:
            clear_working(ctx.run_id, "video_gen")

    completed = [t for t in tasks if t["status"] == "completed" and t["video_uri"]]
    failed = [t for t in tasks if t["status"] == "failed"]
    shot_ids = {str(t.get("shot_id") or "") for t in tasks}
    old_tasks = [
        t for t in (ctx.state.get("video_generation_tasks") or [])
        if str(t.get("shot_id") or "") not in shot_ids
    ]
    ctx.state["video_generation_tasks"] = old_tasks + tasks

    shot_assets = [
        a for a in (ctx.state.get("shot_assets") or [])
        if str(a.get("shot_id") or "") not in shot_ids
    ]
    for task in tasks:
        shot_assets.append({
            "shot_id": task["shot_id"], "url": task["video_uri"] or "",
            "ok": bool(task["video_uri"] and task.get("local_path")),
            "local_path": task.get("local_path") or "",
            "seconds": task["duration"],
            "reference_used": bool(task["reference_images"]),
            "error": task["error"] or "", "link_type": task.get("link_type") or "",
            "first_frame_kind": (task.get("first_frame") or {}).get("kind") or "",
        })
    ctx.state["shot_assets"] = shot_assets
    ctx.state.pop("render_progress", None)

    ctx.trace_step(
        "decide",
        f"视频任务 {len(tasks)} 条（成功 {len(completed)}，失败 {len(failed)}）",
        f"模型 {model_name}；严格串行并已核验本地文件；"
        f"上传声音 {len(voice_payload.get('reference_audio') or [])} 个",
        ok=not failed,
    )
    return {
        "video_generation_tasks": tasks,
        "shot_assets": shot_assets,
        "audio_plan": audio_payload,
        "audio_notice": audio_notice,
        "video_summary": (
            (f"{len(completed)}/{len(tasks)} 镜头已生成并保存到本地"
             if gen is not None else f"{len(tasks)} 个镜头提示词已产出（未生成视频）")
            + (f"｜{audio_notice}" if audio_notice else "")
        ),
    }


def adjusted_map_get(adapted: dict[str, str], shot_id: str) -> str:
    """Look up an adapted prompt, tolerating id spelling drift."""
    if not adapted:
        return ""
    if shot_id in adapted:
        return adapted[shot_id]
    # `SB-9` and `SB-009` are the same shot; a model may have echoed either.
    from .ids import parse

    try:
        _, number = parse(shot_id)
    except ValueError:
        return ""
    for key, value in adapted.items():
        try:
            if parse(key)[1] == number:
                return value
        except ValueError:
            continue
    return ""


async def _check_prompts(
    ctx: AgentContext, board: dict[str, Any], *, forbidden: list[str], era: str
) -> None:
    """Flag shot prompts that cannot work, before any money is spent.

    Findings are recorded on `prompt_issues`, which the review stage reads. One
    source of truth: if this stage says a prompt is unusable, review must not
    approve it on a model's opinion.
    """
    from . import prompt_check

    try:
        check = prompt_check.check_storyboard(board, forbidden=forbidden, era=era)
    except Exception as exc:  # noqa: BLE001 - a check must never kill generation
        ctx.trace_step("warn", "提示词检查失败", str(exc)[:120], ok=False)
        return

    issues = [i.to_critic_issue() for i in check.issues]
    if issues:
        # Deduplicate against what earlier attempts already recorded. `prompt_issues`
        # is per-segment state that survives a rework, so a stage that runs twice
        # used to append the same finding twice — the review then reported two
        # identical rows for one defect, and the count looked like a worse problem
        # than it was.
        recorded = ctx.state.setdefault("prompt_issues", [])
        seen = {(str(i.get("shot_id") or ""), str(i.get("kind") or "")) for i in recorded}
        fresh = [
            i for i in issues
            if (str(i.get("shot_id") or ""), str(i.get("kind") or "")) not in seen
        ]
        if not fresh:
            return
        recorded.extend(fresh)
        ctx.trace_step(
            "decide",
            f"提示词检查：{len(fresh)} 条需修正",
            "；".join(f"{i.get('shot_id')}:{i.get('kind')}" for i in fresh[:4]),
            ok=False,
        )


async def _adapt_prompts(
    ctx: AgentContext,
    shots: list[dict[str, Any]],
    *,
    era: str,
    style: str,
    banned: list[str],
    model: str,
) -> dict[str, str]:
    """Rewrite the shot prompts for the target model. Returns {shot_id: prompt}.

    Only the H3 family gets a rewrite today. The adapter's version of "adapt to
    this model" is concrete — a bracket camera vocabulary and a duration-first
    structure that were measured to produce motion rather than a still frame.
    Applying it to a model that does not speak that syntax would be worse than
    sending the original, so any other model keeps its storyboard prompt.
    """
    if not shots:
        return {}

    is_h3 = "h3" in model.lower() or "minimax" in model.lower()
    if not is_h3:
        ctx.trace_step(
            "decide",
            f"未适配：{model or '未知模型'} 无专用改写规则",
            "沿用分镜原始提示词",
        )
        return {}

    from . import video_prompt

    try:
        system, user = video_prompt.build_rewrite_prompt(
            shots=shots, era=era, style=style, banned=banned
        )
        # The system prompt carries the model's syntax; the user message carries
        # the shots. Asking for the whole batch in one call keeps the camera
        # vocabulary consistent across shots in a segment.
        raw = await ctx.agent_ask(
            "video_gen",
            system=system,
            user=user,
            max_tokens=max(2000, len(shots) * 380),
        )
    except Exception as exc:  # noqa: BLE001 - adaptation is an optimisation
        ctx.trace_step("warn", "提示词适配失败，沿用原提示词", str(exc)[:120], ok=False)
        return {}

    out: dict[str, str] = {}
    for item in _as_list(raw.get("prompts"), 32):
        d = _as_dict(item)
        shot_id = _clip(d.get("id") or d.get("shot_id"), 20)
        text = _clip(d.get("video_prompt") or d.get("prompt"), 1400)
        if shot_id and text:
            out[shot_id] = text

    if out:
        ctx.trace_step(
            "decide",
            f"已适配 {len(out)} 条提示词（{model}）",
            "保留原始分镜提示词，另存最终提示词",
        )
    return out


def _map_voice_to_shots(
    shots: list[dict[str, Any]],
    characters: list[dict[str, Any]],
    voice_plan: Any,
) -> dict[str, dict[str, Any]]:
    """For each shot, the uploaded voice sample of the character in it.

    A shot with several characters takes the first one that has a sample: the
    backends that accept audio take one track per shot, and splitting a shot to
    give each character their own voice is a scene-design decision, not something
    to invent here.
    """
    by_character: dict[str, dict[str, Any]] = {}
    for profile in getattr(voice_plan, "bound", lambda: [])() or []:
        by_character[str(getattr(profile, "character_id", "") or "")] = {
            "asset_uri": str(getattr(profile, "asset_uri", "") or ""),
            "voice_profile_id": str(getattr(profile, "voice_profile_id", "") or ""),
        }

    out: dict[str, dict[str, Any]] = {}
    for shot in shots:
        for cid in shot.get("character_ids") or []:
            found = by_character.get(str(cid))
            if found:
                out[str(shot.get("shot_id") or "")] = found
                break
    return out


def _accepts_arg(fn: Any, name: str) -> bool:
    """Whether this backend's `render_video` *declares* `name`.

    A named parameter, not a swallowed `**kwargs`. The difference is the whole
    point: a backend that absorbs unknown keywords accepts the picture and drops
    it, so the prompt cites 「图片二为老周」 and no second file ever arrives. The
    model then invents a face while the log shows a normal render — the exact
    failure this contract exists to prevent. Only an explicit parameter is a
    promise that the file is used.
    """
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    param = params.get(name)
    return param is not None and param.kind is not inspect.Parameter.VAR_KEYWORD


def _ref_kind(ref: dict[str, Any]) -> str:
    """Classify a reference as character / scene / prop for the dialect layer."""
    declared = str(ref.get("kind") or "").strip()
    if declared in ("first_frame", "character", "scene", "prop"):
        return declared
    role = str(ref.get("reference_role") or "")
    if "人物" in role:
        return "character"
    if "道具" in role:
        return "prop"
    if "场景" in role:
        return "scene"
    return "character"


def _reference_images(
    shot: dict[str, Any],
    characters: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    props: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Which reference pictures this shot needs, each with its purpose stated.

    An image with no stated role is useless: the model decides for itself what to
    inherit, so it may copy the reference's background into a shot that takes place
    elsewhere. The role is what turns a picture into an instruction.

    Only files that exist locally are listed. Both backends upload from disk, so a
    URL here is not a working reference — it is a promise the renderer silently
    breaks, which is what the hollow `portrait_url` read produced for every shot.
    """
    from pathlib import Path

    refs: list[dict[str, Any]] = []

    def add(
        ref_id: str, role: str, uri: Any, asset_id: str,
        label: str = "", detail: str = "", kind: str = "",
    ) -> None:
        path = str(uri or "")
        if not path or not Path(path).is_file():
            return
        refs.append(
            {
                "reference_id": ref_id,
                "reference_type": "image",
                "reference_role": role,
                "reference_uri": path,
                "associated_asset_id": asset_id,
                "associated_shot_id": shot.get("shot_id"),
                # What the dialect layer needs to say 「图片二引用的是【老周】的资产图」:
                # a recognisable name and a short look description. Without them every
                # reference is an anonymous picture, and a prompt that cites images by
                # position cannot be written at all.
                "label": str(label or ""),
                "detail": str(detail or ""),
                # Which sentence the dialect layer uses for this picture. Guessed
                # from the role text when a caller forgets it, because a wrongly
                # classified picture is rendered as a scene reference — wrong, but
                # still a picture of the right thing.
                "kind": kind or _ref_kind({"reference_role": role}),
            }
        )

    for cid in shot.get("character_ids") or []:
        asset = next((c for c in characters if str(c.get("character_id")) == str(cid)), None)
        add(
            str(cid),
            "人物身份与服装参考：继承脸型、发型、服装，不继承背景",
            asset and asset.get("image_path"),
            str(cid),
            label=str((asset or {}).get("name") or cid),
            detail=_clip((asset or {}).get("appearance"), 60),
            kind="character",
        )

    # Fall back to a name match when the shot used names instead of ids.
    for name in shot.get("characters") or []:
        asset = next((c for c in characters if str(c.get("name")) == str(name)), None)
        if asset is not None:
            add(
                str(name),
                "人物身份与服装参考（按姓名匹配）",
                asset.get("image_path"),
                str(asset.get("character_id") or name),
                label=str(asset.get("name") or name),
                detail=_clip(asset.get("appearance"), 60),
                kind="character",
            )

    scene_id = str(shot.get("scene_id") or "")
    if scene_id:
        scene = next((x for x in scenes if str(x.get("scene_id")) == scene_id), None)
        if scene is not None:
            add(
                scene_id,
                "场景布局与光照参考：继承环境、机位方向与光线，不继承人物",
                scene.get("image_path"),
                scene_id,
                label=str(scene.get("name") or scene_id),
                detail=_clip(scene.get("meta"), 50),
                kind="scene",
            )

    for pid in shot.get("prop_ids") or []:
        prop = next((x for x in props if str(x.get("prop_id")) == str(pid)), None)
        if prop is not None:
            add(
                str(pid),
                "道具外观参考：继承形状与材质，不继承人物与背景",
                prop.get("image_path"),
                str(pid),
                label=str(prop.get("name") or pid),
                detail=_clip(prop.get("meta"), 50),
                kind="prop",
            )

    # De-duplicate: a shot may match the same character by both id and name.
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for ref in refs:
        key = str(ref.get("associated_asset_id") or ref.get("reference_id") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(ref)
    return unique


def _normalise_link(raw: Any) -> str:
    """Validate the storyboard's link_type, defaulting to a chainable value.

    The five values are a closed set the video stage branches on; a typo would
    otherwise silently fall into "no chain" for every shot. An absent value is
    CONTINUOUS_ACTION, the most common relation between two shots of one scene.
    """
    from .frame_chain import CHAINABLE, CONTINUOUS_ACTION

    value = str(raw or "").strip().upper().replace(" ", "_")
    known = CHAINABLE | {"SCENE_TRANSITION", "TIME_JUMP", "MONTAGE"}
    return value if value in known else CONTINUOUS_ACTION


async def _render_shot(
    ctx: AgentContext,
    gen: Any,
    shot: dict[str, Any],
    prompt: str,
    refs: list[dict[str, Any]],
    style: str,
    *,
    prev_clip: str = "",
    scene_image: str = "",
    character_image: str = "",
    supports_first_frame: bool = True,
    dialect: str = "by_parameter",
    max_images: int = 1,
) -> tuple[str, str, str, str]:
    """Render one shot. Returns (url, local_path, anchor_kind, anchor_path).

    The first frame is chosen by the storyboard's declared link (frame_chain):
    the previous clip's tail for a chainable link, the scene's reference
    otherwise. `reference_image` is the one parameter every backend honours —
    the gateway uploads it as `first_frame`, a hosted i2v provider as
    `image_url` — so the chain works unchanged on either.

    `style` was accepted here for the whole six-agent era and never used; it is
    now the front of the prompt, because a style only holds if every shot
    carries it, not if it lives in a field nothing reads.
    """
    from .frame_chain import first_frame_for, kind_label
    from .media_steps import _assets_dir
    from .prompt_dialect import (
        BY_PARAMETER, ImageFact, describe, plan_images, render as render_prompt,
    )

    out_dir = _assets_dir(ctx, "shots")
    anchor, anchor_kind = first_frame_for(
        shot,
        prev_clip=prev_clip,
        scene_image=scene_image,
        character_image=character_image,
        out_dir=out_dir,
        supports_first_frame=supports_first_frame,
    )

    # ---- reference images, ordered once ----
    #
    # The order decided here is both the order the model is *told about* and the
    # order the files are handed over in. A prompt saying 「图片二是老周」 while the
    # second file is the courtyard is not a rendering error — it produces a
    # plausible shot of the wrong person, which is much harder to notice.
    #
    # So the citation list is built from `transmitted`, never from the wish list:
    # an image that is not sent must not take a number, and an image that is sent
    # must not go uncited. `max_images` is 1 on every backend that only accepts a
    # first frame; when a backend takes several, extend `transmitted` here and the
    # prompt follows automatically.
    # The first frame is not always "the previous shot's tail": on a scene change
    # it is the scenery picture, and on the fallback path it is a character sheet.
    # Naming it 「上一镜的最后一帧」 in those cases is a different kind of wrong from
    # a numbering error — the picture is right, the sentence describing it is not,
    # so the model is told to continue from a frame that never existed.
    first_frame_label = {
        "prev_last_frame": "上一镜的最后一帧",
        "scene_reference": "本镜的场景图",
        "character_sheet": "人物设定稿",
    }.get(anchor_kind, "本镜的起始画面")

    # Capacity, not capability: `max_reference_images` says what the backend could
    # take, the signature says whether this build can hand it over. A prompt that
    # cites a picture which is never sent is the same defect as a mis-numbered
    # one — the model invents the face instead of using it — so the slot budget is
    # computed from what will really leave the process.
    takes_many = _accepts_arg(gen.render_video, "reference_images")
    capacity = int(max_images) if takes_many else (1 if anchor else 0)

    plan = plan_images(
        ImageFact(
            kind="first_frame",
            label=first_frame_label,
            detail="本镜的起始画面",
            path=str(anchor),
            continuation=(anchor_kind == "prev_last_frame"),
        )
        if anchor
        else None,
        [
            ImageFact(
                kind=_ref_kind(ref),
                label=str(ref.get("label") or ""),
                detail=str(ref.get("detail") or ""),
                path=str(ref.get("reference_uri") or ""),
            )
            for ref in refs
        ],
        max_images=max(0, capacity),
    )

    if style and style not in prompt:
        prompt = f"{style}. {prompt}"
    prompt = render_prompt(prompt, plan.facts, dialect=dialect)

    ctx.trace_step(
        "tool",
        f"调用视频模型（{shot.get('shot_id')}）",
        f"时长 {shot.get('duration')}s；首帧 {kind_label(anchor_kind)}"
        f"（衔接 {shot.get('link_type') or '未声明'}）；"
        f"图序 {describe(plan.facts)}"
        + (f"；另有 {plan.dropped} 张资产图未送（后端上限 {max_images}）" if plan.dropped else "")
        + f"；提示词 {len(prompt)} 字",
    )
    # Reference audio comes from the voice plan: a user-uploaded sample is the
    # only case where the caller has a real audio file to hand over. Without one,
    # the backend produces its own soundtrack (or the TTS fallback covers it).
    reference_audio = str((_ref_audio_for(shot, ctx) or {}).get("asset_uri") or "")

    # Ask for the frame size the film will actually be cut to. The provider may
    # only offer presets (720p/1080p) and will clamp, but sending the intent beats
    # leaving it to a default that then has to be cropped back — every shot used to
    # come back at whatever the model preferred.
    bible_style = _as_dict((ctx.state.get("art_bible") or {}).get("style"))
    aspect = str(bible_style.get("aspect_ratio") or "16:9")
    size = "1080x1920" if aspect.startswith("9:16") else "1920x1080"

    # Every backend implements this one signature; `media_backend` is the reason it
    # exists. Calling a backend-specific method here is what produced an
    # AttributeError under the hosted provider, after the run had already paid for
    # its prompts and assets.
    # The file handed over is taken out of the plan, never from `anchor` directly:
    # one list, one order, one number per picture. A backend that takes several
    # images gets the tail of the same list, so 「图片二」 is the second file here
    # as well as in the prompt. Backends that only take a first frame (every one
    # today) are given exactly that and nothing else.
    first_file = plan.files[0] if plan.facts and plan.facts[0].kind == "first_frame" else ""
    extra_files = [f.path for f in plan.facts if f.kind != "first_frame"]
    extra_kwargs: dict[str, Any] = {}
    if extra_files and takes_many:
        extra_kwargs["reference_images"] = extra_files

    result = await gen.render_video(
        prompt=prompt,
        out_dir=out_dir,
        seconds=float(shot.get("duration") or 5.0),
        reference_image=first_file,
        reference_audio=reference_audio,
        size=size,
        **extra_kwargs,
    )
    local_path = str(getattr(result, "local_path", "") or "")
    local_file = Path(local_path) if local_path else None
    if not local_file or not local_file.is_file() or local_file.stat().st_size <= 0:
        raise RuntimeError("上游视频未成功保存到本地，停止把本镜标记为完成")
    url = getattr(result, "url", "") or getattr(result, "local_path", "")
    if not url:
        raise RuntimeError("视频模型没有返回文件")
    return str(url), local_path, anchor_kind, first_file


def _ref_audio_for(shot: dict[str, Any], ctx: AgentContext | None = None) -> dict[str, Any]:
    """The uploaded voice sample for this shot's characters, if any.

    Only a user-supplied sample yields a path. A model-generated voice has none, and
    passing a placeholder would make a backend try to open a file that does not
    exist — the backend would then report a render failure for what is actually a
    missing upload.
    """
    if ctx is None:
        return {}
    plan = (ctx.state.get("__voice_audio_by_shot") or {})
    return plan.get(str(shot.get("shot_id") or "")) or {}


def _failure_kind(exc: Exception) -> str:
    """Classify a generation failure so the right agent gets the rework.

    The distinction matters: a parameter error is fixed by the video stage, a
    content error by the storyboard. Retrying the wrong one burns credit without
    changing the outcome.
    """
    text = str(exc).lower()
    if any(k in text for k in ("参数", "parameter", "invalid", "400", "尺寸", "duration")):
        return "模型参数"
    if any(k in text for k in ("余额", "balance", "quota", "429", "rate")):
        return "额度或限流"
    if any(k in text for k in ("prompt", "提示词", "违规", "sensitive")):
        return "提示词"
    return "未知"



# ============================================================================
# 6. Quality review — 五阶段审查
# ============================================================================

#: The stages, in pipeline order. Each has its own check list in the prompt.
REVIEW_STAGES = (
    "SCRIPT_REVIEW",
    "STORYBOARD_REVIEW",
    "ASSET_REVIEW",
    "VIDEO_REVIEW",
    "FINAL_REVIEW",
)

QUALITY_REVIEW_SCHEMA = """{
  "review_stage": "",
  "status": "pass|fail",
  "score": 0,
  "issues": [
    {"issue_id": "REV-001", "shot_id": "SB-003", "asset_id": "CHAR-002",
     "issue_type": "",
     "severity": "P0|P1|P2|P3",
     "description": "", "evidence": "",
     "responsible_agent": "script_adaptor|storyboard|consistency|video_gen",
     "repair_instruction": "", "recheck_required": true}
  ],
  "repair_targets": ["需要返工的 agent"],
  "summary": ""
}"""


async def quality_review_agent(
    ctx: AgentContext, definition: AgentDefinition, options: dict[str, Any]
) -> dict[str, Any]:
    """Review the segment at the stage the caller asks for.

    Rules run before the model, and some of them can fail the review on their own.
    The reason is asymmetry: a shot that was never generated, or was rendered
    without its reference portrait, is a definite failure that costs nothing to
    detect. Letting a model judge it invites a "looks fine" verdict on work that
    does not exist.
    """
    stage = str(options.get("review_stage") or "VIDEO_REVIEW").upper()
    if stage not in REVIEW_STAGES:
        stage = "VIDEO_REVIEW"

    segments = ctx.state.get("segments") or []
    index = int(options.get("segment_index") or 0)
    board = _select_board(ctx.state.get("storyboards") or {}, segments, index)
    segment_id = str(board.get("segment_id") or "")

    issues: list[dict[str, Any]] = []
    deterministic_fail = False

    # ---- deterministic checks, by stage ----
    if stage in ("VIDEO_REVIEW", "FINAL_REVIEW"):
        assets = [
            a for a in (ctx.state.get("shot_assets") or [])
            if str(a.get("shot_id") or "").startswith(segment_id)
        ] or (ctx.state.get("shot_assets") or [])

        # Prompt-only mode deliberately produces no media: the prompt IS the
        # deliverable. Judging it as "no shots, cannot review" would fail every
        # such run and report a working configuration as broken.
        prompt_only = media_mode.prompt_only() or bool(options.get("prompt_review_only"))

        if not assets:
            if not prompt_only:
                issues.append(
                    _issue(
                        1, "", "missing_assets", review_mode.P0,
                        "没有生成任何镜头，无法审查画面",
                        "视频生成阶段没有产出",
                        "video_gen",
                        "检查模型配置与额度后重新生成",
                    )
                )
                deterministic_fail = True
        else:
            # A shot with no URL is a failure only when media was actually
            # requested. In prompt-only mode every shot legitimately has no URL,
            # and counting those as generation failures reported a working
            # configuration as broken.
            failed = [
                a for a in assets
                if not a.get("ok") and str(a.get("error") or "").strip()
            ]
            if failed:
                deterministic_fail = True
                issues.append(
                    _issue(
                        len(issues) + 1, str(failed[0].get("shot_id")), "generation_failed",
                        review_mode.P0,
                        f"{len(failed)} 个镜头生成失败",
                        (failed[0].get("error") or "")[:80],
                        "video_gen",
                        "按失败原因修正后重新生成",
                    )
                )

            # A shot rendered from the wrong reference is the defect this whole
            # design exists to prevent, so it is a rule, not an opinion. In
            # prompt-only mode there is nothing to reference — the rule used to
            # pass there only because the refs list carried hollow entries, and
            # once the hollow field was fixed it started failing every prompt run.
            no_ref = [a for a in assets if not a.get("reference_used")]
            if no_ref and len(no_ref) == len(assets) and not prompt_only:
                issues.append(
                    _issue(
                        len(issues) + 1, str(no_ref[0].get("shot_id")), "no_reference",
                        review_mode.P1,
                        "所有镜头都没有使用人物参考图，人物一致性无法保证",
                        "reference_used=false",
                        "consistency",
                        "确认定妆照已生成并在生成视频时作为首帧传入",
                    )
                )

    # Prompt defects found earlier travel with the review: one source of truth.
    for found in (ctx.state.get("prompt_issues") or []):
        if not isinstance(found, dict):
            continue
        issues.append(
            _issue(
                len(issues) + 1,
                str(found.get("shot_id") or ""),
                str(found.get("kind") or "prompt"),
                review_mode.P1 if str(found.get("severity")) == "fail" else review_mode.P2,
                str(found.get("detail") or ""),
                "提示词检查",
                # A bad prompt originates in the storyboard; re-rendering the same
                # text would reproduce the defect.
                "storyboard",
                str(found.get("fix_hint") or "修正提示词后重新分镜"),
            )
        )

    if stage in ("SCRIPT_REVIEW", "STORYBOARD_REVIEW"):
        script = _select_script(ctx.state.get("adapted_script"), segments, index)
        if stage == "SCRIPT_REVIEW":
            if not script or not script.get("sections"):
                issues.append(
                    _issue(1, "", "script_missing", review_mode.P0,
                           "没有改编脚本，无法审查", "", "script_adaptor",
                           "重新执行内容改编")
                )
                deterministic_fail = True
            else:
                # An unverified fact presented as certain history is the failure
                # that embarrasses a documentary, so it is checked by rule.
                uncertain = script.get("historical_facts") or []
                shaky = [f for f in uncertain if str(f.get("certainty") or "") != "史实"]
                if shaky:
                    issues.append(
                        _issue(
                            len(issues) + 1, "", "unverified_history", review_mode.P1,
                            f"{len(shaky)} 条史实标记为存疑",
                            str(shaky[0].get("fact") or "")[:70],
                            "script_adaptor",
                            "核实或明确标注为艺术创作",
                        )
                    )
        else:
            if not board.get("shots"):
                issues.append(
                    _issue(1, "", "storyboard_missing", review_mode.P0,
                           "没有分镜表，无法审查", "", "storyboard", "重新拆分镜")
                )
                deterministic_fail = True
            else:
                da = board.get("duration_analysis") or {}
                total = float(board.get("total_seconds") or 0)
                target = float(board.get("total_seconds") or 0)
                if da.get("overlap_notes") == [] and (
                    float(da.get("narration_duration") or 0)
                    + float(da.get("dialogue_duration") or 0)
                ) > total > 0:
                    issues.append(
                        _issue(
                            len(issues) + 1, "", "duration_double_count", review_mode.P1,
                            "旁白与对白时长相加超过总时长，可能是重复累计",
                            f"旁白+对白 {float(da.get('narration_duration') or 0) + float(da.get('dialogue_duration') or 0):.0f}s"
                            f" vs 总时长 {total:.0f}s",
                            "storyboard",
                            "确认旁白与对白是否同时发生，不得简单累加",
                        )
                    )

    if stage == "ASSET_REVIEW":
        manifest = (ctx.state.get("asset_index") or {}).get(segment_id) or {}
        characters = manifest.get("character_assets") or []
        if not characters:
            issues.append(
                _issue(1, "", "assets_missing", review_mode.P1,
                       "没有人物资产，四视图与人物一致性无从建立", "",
                       "consistency", "重新执行一致性 Agent")
            )
        elif board.get("shots"):
            # Every character who appears must have an asset, or the first shot
            # they appear in has no reference to render from.
            known = {str(c.get("name")) for c in characters}
            unknown = {
                name
                for shot in (board.get("shots") or [])
                for name in (shot.get("characters") or [])
                if isinstance(name, str) and name not in known
            }
            if unknown:
                # Named after the first shot the missing character appears in, not
                # left blank: a finding with no locator can only be fixed by
                # rebuilding the whole segment's assets, while this one needs one
                # portrait added.
                first_shot_with_unknown = next(
                    (
                        str(shot.get("shot_id") or "")
                        for shot in (board.get("shots") or [])
                        if any(
                            isinstance(name, str) and name in unknown
                            for name in (shot.get("characters") or [])
                        )
                    ),
                    "",
                )
                issues.append(
                    _issue(
                        len(issues) + 1, first_shot_with_unknown,
                        "character_without_asset", review_mode.P1,
                        f"{len(unknown)} 个出场人物没有对应资产",
                        "、".join(sorted(unknown)[:4]),
                        "consistency",
                        "为这些人物补充四视图与参考图",
                    )
                )

    # ---- model pass ----
    #
    # The model sees what the rules cannot: composition, tone, whether the action
    # reads. It is asked only about what the rules did not already settle.
    summary_text = ""
    # The mode is read once, before anything else, because it decides both what the
    # reviewer is asked and how its findings are judged. The director sets it on the
    # context; `options` wins when the pipeline passes it explicitly.
    mode = review_mode.normalise(options.get("review_mode") or ctx.state.get("review_mode"))

    # Stated in the prompt, not only in the digest: a P0 is terminal, and a model
    # that reads "0/16 成功" will keep writing one unless told plainly that this run
    # never asked for video. The extra sentence costs nothing; the false P0 cost a
    # whole run.
    prompt_only_note = (
        "\n【提示词模式】本轮不渲染图片或视频，缺少媒体文件是预期结果，"
        "不得因此判 P0，也不得据此打回。\n"
        if media_mode.prompt_only()
        else ""
    )

    status = "fail" if deterministic_fail else "pass"
    score = 0.0

    if not deterministic_fail:
        payload = _review_digest(
            ctx, stage, board, issues,
            prompt_review_only=bool(options.get("prompt_review_only")),
        )
        try:
            raw = await ctx.agent_ask(
                "quality_review",
                system=_system("quality_review"),
                user=_json_only(
                    f"""请审查这一段（阶段：{stage}）。

【审核模式：{mode}】
{review_mode.describe(mode)}
{prompt_only_note}

请按四级严重度标记每个问题：
- P0 致命：结果无法使用、核心需求完全未满足、重大事实错误、关键内容缺失。
- P1 重要：剧情逻辑明显漏洞、关键镜头无法实现、角色/场景/道具严重不一致、
  用户明确要求的内容缺失、明显影响成片质量或观众理解。
- P2 普通：构图可优化、台词略不自然但不影响理解、动作略生硬、衔接轻微瑕疵。
- P3 建议：可增强镜头语言、可增加细节、可强化情绪、可调整节奏。

当前模式下只有 {'、'.join(sorted(review_mode.blocking_set(mode)))} 会阻断审核；
其余记入 non_blocking_issues，不得因此打回。

{payload}

已知的规则检查结果（不要重复报告这些）：
{chr(10).join(f'- {i["severity"]} {i["issue_type"]}: {i["description"]}' for i in issues) or '（无）'}

只报告规则检查看不到的问题。每个问题必须指明 responsible_agent 与可执行的修复方式。
**每个问题必须尽可能给出定位**，否则只能整段重做，代价极高：
- 与某个镜头有关 → 填 shot_id（如 SB-003）；
- 与某个人物/场景/道具有关 → 填 asset_id（如 CHAR-002）；
- 确实只涉及整段结构（如"缺一个过渡段"）时两个都留空。
不得因为「还可以更好」「建议更有电影感」「整体不够完美」这类没有具体证据的主观评价判定不通过。""",
                    QUALITY_REVIEW_SCHEMA,
                ),
                max_tokens=2200,
            )
            status = "pass" if str(raw.get("status") or "").lower() != "fail" else "fail"
            score = _num(raw.get("score")) or 0.0
            summary_text = _clip(raw.get("summary"), 200)

            for item in _as_list(raw.get("issues"), 10):
                d = _as_dict(item)
                issues.append(
                    _issue(
                        len(issues) + 1,
                        _clip(d.get("shot_id"), 20),
                        _clip(d.get("issue_type"), 30) or "other",
                        _clip(d.get("severity"), 12) or review_mode.P2,
                        _clip(d.get("description"), 160),
                        _clip(d.get("evidence"), 100),
                        _clip(d.get("responsible_agent"), 20) or "video_gen",
                        _clip(d.get("repair_instruction"), 140),
                        asset_id=_clip(d.get("asset_id"), 20),
                    )
                )
        except Exception as exc:  # noqa: BLE001 - a review must still return
            ctx.trace_step("warn", "模型审查失败", str(exc)[:120], ok=False)
            summary_text = summary_text or "规则检查通过，模型审查未能完成"
    else:
        summary_text = "规则检查未通过"

    # ---- consolidate ----
    #
    # ── the mode decides the verdict ──────────────────────────────────────────
    #
    # This is the one place PASS/REJECT is decided. It is a rule, not a request to
    # the model: the model reports severities, the mode says which of them matter.
    # Previously any CRITICAL failed the review, and a reviewer asked "is this a
    # complete screenplay" answers CRITICAL to a 400-character fragment forever —
    # so every run died on a finding nobody could fix.
    mode = review_mode.normalise(options.get("review_mode") or ctx.state.get("review_mode"))

    blocking: list[dict[str, Any]] = []
    non_blocking: list[dict[str, Any]] = []
    for issue in issues:
        sev = review_mode.severity_of(issue.get("severity"))
        # Recorded on the issue itself so a report can be audited later: why this
        # one blocked and that one did not is otherwise unrecoverable.
        issue["severity"] = sev
        issue["blocks"] = review_mode.blocks(mode, sev)
        (blocking if issue["blocks"] else non_blocking).append(issue)

    # Deterministic failures are NOT exempt from the mode.
    #
    # The earlier reasoning was wrong: it treated every rule-check finding as
    # fatal, so a FAST run still failed on "画面描述只有 13 字（下限 15）" — a
    # prompt-length nit reported as P1. The user asked for speed and got a
    # rejection over a threshold the rules themselves invented.
    #
    # What is exempt is the *absence of the deliverable*: an empty shot list means
    # nothing was produced at all, and no mode makes that pass. So the fatal bit
    # survives only when the stage genuinely could not produce its output, which
    # the caller signals separately (`deterministic_fail` and an empty asset list).
    #
    # The mode now governs the rest, exactly as it governs the model's findings.
    if deterministic_fail and not blocking and _nothing_produced(
        stage, ctx.state,
        prompt_review_only=bool(options.get("prompt_review_only")),
    ):
        status = "fail"
    else:
        status = "fail" if blocking else "pass"

    # `repair_targets` is what the director dispatches on. Only blocking issues
    # route a rework: sending an agent to fix a P3 suggestion would undo work that
    # already passed.
    targets: list[str] = []
    for issue in blocking:
        agent = _resolve_agent_name(str(issue.get("responsible_agent") or ""), stage)
        if agent and agent not in targets:
            targets.append(agent)

    report = {
        "review_id": format_id("review", int(options.get("review_number") or 1)),
        "review_stage": stage,
        "review_mode": mode,
        "status": status,
        "score": round(score, 2),
        "issues": issues,
        "repair_targets": targets,
        "summary": summary_text,
        # ---- the review protocol ----
        "blocking_issues": blocking,
        "non_blocking_issues": non_blocking,
        "revision_scope": _revision_scope(blocking, stage),
        "review_evidence": [str(i.get("evidence") or "") for i in issues if i.get("evidence")],
        "responsible_agent": targets,
        "repair_instructions": [
            {"agent": _resolve_agent_name(str(i.get("responsible_agent") or ""), stage),
             "instruction": str(i.get("repair_instruction") or "")}
            for i in blocking
        ],
    }

    ctx.state.setdefault("quality_reports", []).append(report)

    for target in targets:
        ctx.trace_step(
            "reject", f"返工指向：{target}", f"{len(issues)} 条问题",
            ok=False,
        )

    return {
        "critic_verdict": {
            "approved": status == "pass",
            "issues": issues,
            "scores": {"overall": score},
            # The pipeline re-dispatches from this field. Mapping the first
            # responsible agent keeps the existing rework loop working, while the
            # full list stays available in `repair_targets`.
            "rework_target": (targets[0] if targets else ""),
            # ---- the review protocol, mirrored here on purpose ----
            #
            # The pipeline reads the verdict, not the stored report, so anything it
            # needs to act on has to travel in this dict. Keeping the split only in
            # `report` (which is appended to `quality_reports`) made the pipeline
            # read `blocking_issues` as None and fall back to "every issue blocks" —
            # the exact behaviour the modes exist to avoid. The two views must not
            # be able to disagree.
            "blocking_issues": blocking,
            "non_blocking_issues": non_blocking,
            "revision_scope": report["revision_scope"],
        },
        "critic_summary": summary_text or f"{stage}：{status}",
        "quality_reports": ctx.state["quality_reports"],
    }


def _issue(
    number: int,
    shot_id: str,
    issue_type: str,
    severity: str,
    description: str,
    evidence: str,
    responsible_agent: str,
    repair_instruction: str,
    *,
    asset_id: str = "",
) -> dict[str, Any]:
    return {
        "issue_id": format_id("review", max(1, number)),
        "project_id": "",
        "shot_id": shot_id,
        # Which asset (character/scene/prop) is at fault, when the finding is about
        # one. An ASSET_REVIEW finding without this can only be fixed by rebuilding
        # the whole segment's assets, which is how one bad portrait cost four.
        "asset_id": asset_id,
        "issue_type": issue_type,
        "severity": severity,
        "description": description,
        "evidence": evidence,
        "responsible_agent": responsible_agent,
        "repair_instruction": repair_instruction,
        # Normalised before the comparison: the rule-based call sites below still
        # write the legacy names (CRITICAL/MAJOR) while the model writes P-levels,
        # and a raw string test marked a P0 "no recheck needed" purely because of
        # spelling.
        "recheck_required": review_mode.severity_of(severity)
        in (review_mode.P0, review_mode.P1),
    }


def _nothing_produced(
    stage: str, state: dict[str, Any], *, prompt_review_only: bool = False
) -> bool:
    """Whether the stage produced no deliverable at all.

    Distinct from "produced something imperfect". A rule check that fires on a
    short prompt or a missing optional field is a quality note; a stage that
    emitted no shots and no prompts has nothing to judge, and that alone is fatal
    in every mode.

    Prompt-only runs are excluded: there the prompt IS the deliverable, so no
    media is the expected outcome rather than a failure.
    """
    if media_mode.prompt_only() or prompt_review_only:
        return False

    if stage in ("VIDEO_REVIEW", "FINAL_REVIEW"):
        return not (state.get("shot_assets") or [])
    if stage == "STORYBOARD_REVIEW":
        return not (state.get("storyboards") or {})
    if stage == "ASSET_REVIEW":
        return not (state.get("asset_index") or {})
    if stage == "SCRIPT_REVIEW":
        return not (state.get("adapted_script") or {})
    return False


def _review_digest(
    ctx: AgentContext, stage: str, board: dict[str, Any], known: list[dict[str, Any]],
    *, prompt_review_only: bool = False,
) -> str:
    """The evidence the reviewer is given, trimmed to what the stage needs.

    Whole state would be megabytes and would bury the signals: for a video review
    the reviewer needs the shots, the prompts sent and what came back, not the
    script's creative notes.
    """
    lines: list[str] = []

    # This segment, not the first one. Everything below is scoped to `sid`.
    segment_id = str(board.get("segment_id") or "")

    if stage in ("SCRIPT_REVIEW", "STORYBOARD_REVIEW"):
        segments = ctx.state.get("segments") or []
        script = _select_script(
            ctx.state.get("adapted_script"),
            segments,
            _index_of_segment(segments, segment_id),
        )
        if script:
            lines.append(f"剧本模式：{script.get('mode') or ''}")
            # The reviewer needs the text, not a caption of it. With only a 90-char
            # scene caption per section it reported "本次审核请求中未附带任何剧本
            # 内容" as a P0 and sent the stage back — a review of nothing, produced
            # by the digest, not by the script.
            # Every section must reach the reviewer. The live failure behind this
            # rule had six sections: the old ``[:4]`` hid the final two, which
            # carried REQ-005/006/007, so review repeatedly claimed that content
            # visible in the UI did not exist.
            for s in (script.get("sections") or []):
                d = _as_dict(s)
                title = d.get("section_title") or ""
                purpose = d.get("narrative_purpose") or ""
                lines.append(f"- 【{title}】{_clip(purpose, 60)}")
                lines.append(f"  画面：{_clip(d.get('scene_description'), 260)}")
                narration = [str(x) for x in (d.get("narration") or []) if str(x).strip()]
                if narration:
                    lines.append(f"  旁白：{_clip(' '.join(narration), 300)}")
                dialogue = [
                    f"{_clip(_as_dict(x).get('character'), 20)}「{_clip(_as_dict(x).get('line'), 60)}」"
                    for x in (d.get("dialogue") or [])
                    if _as_dict(x).get("line")
                ]
                if dialogue:
                    lines.append("  对白：" + "；".join(dialogue[:4]))
                events = [str(x) for x in (d.get("events") or []) if str(x).strip()]
                if events:
                    lines.append(f"  事件：{_clip('；'.join(events), 160)}")

        # What this segment owes the finished film — and, said plainly, what it
        # does not. Without this line the reviewer applies the whole-film standard
        # to one segment and rejects work that is correct for its own place.
        constraint = (ctx.state.get("constraints") or {}).get(segment_id) or {}
        required = constraint.get("requirements") or []
        if required:
            lines.append("")
            lines.append(f"本段（{segment_id}）必须完成的必现项——只验收这些：")
            for r in required:
                rid = r.get("requirement_id") or ""
                text = r.get("requirement") or ""
                how = r.get("acceptance") or ""
                lines.append(f"- [{rid}] {text}" + (f"（验收：{how}）" if how else ""))
            lines.append(
                "其他分段负责的内容不在本段的验收范围内；"
                "缺少别段的事件不算本段的问题，不得据此打回。"
            )

    if stage in ("STORYBOARD_REVIEW", "VIDEO_REVIEW", "FINAL_REVIEW"):
        da = board.get("duration_analysis") or {}
        lines.append(
            f"分镜：{len(board.get('shots') or [])} 镜，共 {board.get('total_seconds') or 0:.0f} 秒；"
            f"旁白 {da.get('narration_duration') or 0:.0f}s，对白 {da.get('dialogue_duration') or 0:.0f}s"
        )
        # Do not review a prefix of the storyboard. In the observed run the board
        # had nine valid shots and SB-013 alone carried REQ-007; ``[:8]`` removed
        # precisely that evidence and created a false P0 rejection.
        for shot in (board.get("shots") or []):
            req = "、".join(shot.get("requirement_ids") or [])
            lines.append(
                f"- {shot.get('shot_id')}（{shot.get('duration')}s，"
                f"{shot.get('shot_size') or ''}"
                + (f"，必现 {req}" if req else "")
                + f"）：{_clip(shot.get('visual_description'), 100)}"
            )
            # Narration travels with the shot because it *is* the on-screen text:
            # the assembly burns every narration line into hard subtitles. Leaving
            # it out is what produced "SB-010 只有场景全景和红旗横幅，未规定字幕卡"
            # on a shot whose narration already read "1954年12月21日至25日，全国
            # 政协二届一次会议举行。" — the reviewer was asked to verify a fact it
            # was never shown, and the segment was reworked twice for nothing.
            narration = str(shot.get("narration") or "").strip()
            if narration:
                lines.append(f"    旁白（成片逐句烧成硬字幕）：{_clip(narration, 140)}")
            dialogue = [
                f"{_clip(_as_dict(x).get('character'), 16)}「{_clip(_as_dict(x).get('line'), 60)}」"
                for x in (shot.get("dialogue") or [])
                if _as_dict(x).get("line")
            ]
            if dialogue:
                lines.append("    对白（同样烧成硬字幕）：" + "；".join(dialogue[:3]))
        lines.append(
            "判定口径：本系统的成片会把每镜的旁白/对白逐句烧成硬字幕，而视频模型"
            "画不出规整的中文字。因此「字幕标明 / 明确标明 / 说明」这类验收条件，"
            "只要该信息在某一镜的旁白或对白里**逐字出现**，即视为已满足，"
            "**不得**再要求画内字幕卡、横幅文字或可读的文件文字。"
            "只有旁白/对白/画内文字全都没有承载时，才算缺失。"
        )

    if stage in ("VIDEO_REVIEW", "FINAL_REVIEW"):
        assets = ctx.state.get("shot_assets") or []
        if (
            media_mode.prompt_only()
            or not ctx.state.get("generate_videos")
            or prompt_review_only
        ):
            # Saying "0/16 成功" without saying why made the reviewer read a
            # prompt-only run as a total render failure: it raised a P0 ("所有视频
            # 镜头生成任务均失败"), sent the stage back, and the rework produced the
            # same empty result until the run died on "未产出成片" — a working
            # configuration reported as broken.
            prompts = [
                t for t in (ctx.state.get("video_generation_tasks") or [])
                if str(t.get("final_video_prompt") or "").strip()
            ]
            lines.append(
                f"视频提示词预审：本轮尚未渲染，已产出 {len(prompts)} 条镜头提示词。"
                "必须等全片所有视频提示词均完成并通过审查后，才会开始串行生成。"
                "「没有视频文件」是本模式的预期结果，**不得**据此判定失败或打回；"
                "请只审查提示词质量、分镜可执行性与前后一致性。"
            )
        else:
            lines.append(f"生成结果：{sum(1 for a in assets if a.get('ok'))}/{len(assets)} 成功")
            for a in assets[:8]:
                if not a.get("ok"):
                    lines.append(f"- {a.get('shot_id')} 失败：{_clip(a.get('error'), 60)}")

    if stage == "ASSET_REVIEW":
        manifest = (ctx.state.get("asset_index") or {}).get(str(board.get("segment_id") or "")) or {}
        # This review now runs before any picture exists: the run draws the whole
        # film's assets afterwards, in one serial queue, so that two segments
        # cannot draw the same person twice. Saying "参考图 无" here would read as
        # a render failure and send the stage back for something it cannot fix.
        lines.append(
            "本轮审查发生在出图之前：参考图会在提示词全部通过后由全片串行队列统一生成，"
            "**不得**因缺少参考图判定失败或要求重画；只审提示词是否可执行、是否互相矛盾。"
        )
        lines.append(
            f"资产：人物 {len(manifest.get('character_assets') or [])}，"
            f"场景 {len(manifest.get('scene_assets') or [])}，"
            f"道具 {len(manifest.get('prop_assets') or [])}，"
            f"声音 {len(manifest.get('voice_profiles') or [])}"
        )
        # Ids travel with every asset line, and the reviewer is asked to cite them.
        # Without an id a finding can only say "one character has no asset", which
        # forces a full re-generation of the segment to fix a single portrait —
        # the cost this field removes. The id is also what the consistency stage
        # reuses to keep every asset that was not named.
        #
        # An empty prompt is a P0 now, not a note: the prompt is the deliverable
        # in prompt mode, and an empty string must never reach the image endpoint
        # (it is charged and returns an unrelated picture).
        for c in (manifest.get("character_assets") or [])[:8]:
            has_prompt = bool(c.get("four_view_prompt"))
            lines.append(
                f"- [{c.get('character_id') or '?'}] {c.get('name')}："
                f"四视图提示词 {'有' if has_prompt else '缺失（P0）'}"
                + ("，参考图 已有" if c.get("image_path") else "")
            )
        for sc in (manifest.get("scene_assets") or [])[:6]:
            lines.append(f"- [{sc.get('scene_id') or '?'}] 场景 {sc.get('name')}："
                         f"提示词 {'有' if sc.get('prompt') else '缺失'}")
        for pr in (manifest.get("prop_assets") or [])[:6]:
            lines.append(f"- [{pr.get('prop_id') or '?'}] 道具 {pr.get('name')}："
                         f"提示词 {'有' if pr.get('prompt') else '缺失'}")
        lines.append(
            "报告资产问题时，必须在 shot_id 之外再给出 asset_id（上面方括号里的编号），"
            "否则只能整段重做。"
        )

    # A review must contain the complete deliverable. Prefix clipping creates
    # false "required content missing" verdicts, which are worse than a failed
    # review because they trigger destructive rework of correct material. Fields
    # above are already individually compacted, so these limits remain bounded
    # while leaving room for every section and every (at most 24) shot.
    if stage == "STORYBOARD_REVIEW":
        limit = 16000
    elif stage == "SCRIPT_REVIEW":
        limit = 12000
    else:
        limit = 4000
    return _clip("\n".join(lines), limit)



# ============================================================================
# 1. Director — 输入识别 + 工作流选择 + 分段
# ============================================================================

PRODUCTION_PLAN_SCHEMA = """{
  "project_id": "",
  "input_type": "FULL_SCRIPT|ARTICLE|HISTORY_MATERIAL|STORY_SUMMARY|CREATIVE_IDEA|VISUAL_REFERENCE|AUDIO_REFERENCE|EXISTING_PROJECT|UNKNOWN",
  "operation": "CREATE|CONTINUE|MODIFY|QUERY|REPAIR",
  "workflow": "A|B|C|D|E|F",
  "review_mode": "STRICT|NORMAL|FAST",
  "content_ok": true,
  "content_problems": [],
  "advice": "",
  "segments": [{"id": "seg1", "title": "", "text": "<这一段对应的原文，必须逐字取自输入，不要改写、不要概括、不要留空>", "order": 0}],
  "requirements": [
    {"requirement_id": "REQ-001",
     "requirement": "<必须让观众看到的一件事或一个画面，如「望志路石库门会议」>",
     "kind": "event|visual|fact",
     "segments": ["seg1"],
     "required": true,
     "acceptance": "<怎样算拍到了，如「黑漆大门铜环特写 + 代表落座」>"}
  ],
  "target_duration": 0,
  "aspect_ratio": "16:9",
  "style": "",
  "notes": ""
}"""


def _parse_requirements(
    raw: Any, segments: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], list[str]]:
    """Whole-film requirements, plus the per-segment split the agents read.

    Three shapes come out of this, and the split is the point:

      · `requirements` — what the finished film must contain, once each.
      · `by_segment` — what each segment is accountable for. A requirement
        assigned to seg2 appears only under seg2; it is never copied to every
        segment, which is what produced the endless rework loop.
      · `unassigned` — requirements no segment claimed. These are reported, not
        pushed onto everybody: "unknown owner" is not "everyone's job".

    Ids the director invented that match no segment are dropped from the split
    but kept in `unassigned`, so a typo shows up as a warning instead of silently
    landing on segment 1.
    """
    known = [str(s.get("id") or "") for s in segments]
    out: list[dict[str, Any]] = []
    by_seg: dict[str, list[dict[str, Any]]] = {}
    unassigned: list[str] = []

    for i, item in enumerate(_as_list(raw, 24)):
        d = _as_dict(item)
        text = _clip(d.get("requirement"), 120)
        if not text:
            continue
        rid = _clip(d.get("requirement_id"), 24) or f"REQ-{i + 1:03d}"
        owners = [
            str(x) for x in _as_list(d.get("segments"), 8)
            if str(x) in known
        ]
        # A requirement may legitimately span segments; it may not span all of
        # them by default. Bare "all" is refused here for the same reason.
        req = {
            "requirement_id": rid,
            "requirement": text,
            "kind": _clip(d.get("kind"), 16) or "event",
            "segments": owners,
            "required": bool(d.get("required", True)),
            "acceptance": _clip(d.get("acceptance"), 120),
        }
        out.append(req)
        if not owners:
            unassigned.append(rid)
        for sid in owners:
            by_seg.setdefault(sid, []).append(req)
    return out, by_seg, unassigned


async def director_six_agent(
    ctx: AgentContext, definition: AgentDefinition, options: dict[str, Any]
) -> dict[str, Any]:
    """Identify the input, choose a workflow, and cut the source into segments.

    Segmentation stays the director's job even though the script adaptor writes the
    script. The reason is concurrency: segments have no dependency between them, so
    they run in parallel, and that parallelism is a property of how the source was
    cut. Moving the cut into the adaptor would make the pipeline serial and cost the
    wall-clock win on every run.

    `workflow` is recorded but not yet acted on. The spec describes six paths
    (A–F); the pipeline currently implements B/C (adapt → shoot). Recording which
    one was chosen means the gap is visible in the run record rather than silently
    assumed.
    """
    text = _chapter_text(ctx)
    target_seconds = int(options.get("target_seconds") or ctx.state.get("target_seconds") or 60)

    # The model is asked for the input type, the workflow and the cut in one call.
    # Three separate calls would let them disagree — a cut of six segments produced
    # by a model that had decided this was a "creative idea" is exactly the
    # mismatch this prevents.
    raw = await ctx.agent_ask(
        "director",
        system=_system("director"),
        user=_json_only(
            f"""请判断下面这段输入属于什么类型，选择工作流，并切成可独立处理的段落。

正文（{len(text)} 字）：
{text[:2600]}

材料体量：约 {target_seconds} 秒的旁白量（按字数估算，仅供参考，以内容为准）

切分要求：
- 按时间、事件或论点自然切分，每段 200-500 字
- 每段要能独立成戏（有人、有场景、有事）
- 段数控制在 1-6 段；内容过少时宁可只有 1 段
- 不要为了凑时长而拆碎或合并段落，每段只按它自己的叙事单元切
- id 从 seg1 开始递增

必现项分配（requirements 字段，这一步必须做）：
- 先列出**整部片子**必须让观众看到的事或画面，每条一个 requirement，全片只写一次
- 然后给每条填 `segments`：它由哪一段负责。一段**只负责自己那段该出现的东西**
- 绝不能把同一条同时分配给所有段。比如「望志路石库门会议」只归开会那一段，
  「南湖游船续会」只归南湖那一段；撤离段两者都不归
- 确实判断不出归属的，`segments` 留空数组，不要随便塞给某一段
- `acceptance` 写清"拍到什么算完成"，审查会照着它验收""",
            PRODUCTION_PLAN_SCHEMA,
        ),
        # 6000, not 4000: the reply now carries the cut **and** the requirement
        # split. Truncating it surfaces as "无法解析的内容" — a truncation reported
        # as a formatting failure.
        max_tokens=6000,
    )

    # ---- input classification ----
    input_type = _clip(raw.get("input_type"), 30) or "UNKNOWN"
    operation = _clip(raw.get("operation"), 20) or "CREATE"
    workflow = _clip(raw.get("workflow"), 2)

    # ---- review mode ----
    #
    # Precedence, and why it is this order:
    #
    #   1. What the user asked for in `options` — an explicit request wins, always.
    #   2. What the model proposed — it read the material and the task text, so it
    #      may know that this is a historical piece needing careful checking.
    #   3. What the deployment configured (`STUDIO_REVIEW_MODE`).
    #   4. NORMAL.
    #
    # The reviewer may never change it afterwards: `quality_review` reads this value
    # and is explicitly told not to derive its own.
    requested_mode = str(options.get("review_mode") or "").strip()
    proposed_mode = _clip(raw.get("review_mode"), 12)
    if requested_mode:
        decided_mode = review_mode.normalise(requested_mode)
    elif proposed_mode:
        decided_mode = review_mode.normalise(proposed_mode)
    else:
        decided_mode = review_mode.from_env()
    ctx.state["review_mode"] = decided_mode

    # A workflow the model omitted is inferred, not left blank.
    #
    # The rejected run displayed "工作流 ?" because the reply carried no `workflow`
    # field. A `?` is worse than a guess here: the workflow only records which path
    # was taken, and the pipeline implements one path anyway. Deriving it from the
    # input type states something true, while an empty value reads as "the system
    # does not know what it is doing" — which is what the user concluded.
    #
    # The field is recorded, not acted on, so a wrong inference costs nothing.
    if not workflow:
        workflow = _infer_workflow(input_type, operation)
        ctx.trace_step(
            "decide",
            f"工作流未给出，按输入类型推断为 {workflow}",
            f"输入类型 {input_type}、操作 {operation}；该字段只记录走了哪条路径，不影响执行",
        )

    jobs_text = {
        "A": "完整脚本 → 整理 → 审查 → 分镜",
        "B": "资料 → 改编 → 审查 → 分镜",
        "C": "梗概 → 创作 → 审查 → 分镜",
        "D": "只生成角色/场景图",
        "E": "已有分镜，只生成视频",
        "F": "修改已有项目",
    }
    ctx.trace_step(
        "decide",
        f"输入类型：{input_type}（{operation}）",
        f"工作流 {workflow or '未判定'}"
        + (f"：{jobs_text[workflow]}" if workflow in jobs_text else ""),
    )

    # ---- content check ----
    # The most common real failure is source material that cannot become a
    # character video: many events, no people. Saying so up front is more useful
    # than rendering a slideshow.
    problems = [_clip(x, 80) for x in _as_list(raw.get("content_problems"), 6)]
    content_ok = bool(raw.get("content_ok", True))
    advice = _clip(raw.get("advice"), 160)

    # ---- segments ----
    segments: list[dict[str, Any]] = []
    for i, item in enumerate(_as_list(raw.get("segments"), 6)):
        d = _as_dict(item)
        seg_text = _clip(d.get("text"), 1200)
        if not seg_text:
            # A segment with no body is not skipped, and the whole text is never
            # silently dropped.
            #
            # Live failure this fixes: the director returned
            #     [{"id":"seg1","title":"全篇","order":0}]
            # with no `text` field. Every segment was then skipped, so the run
            # proceeded with an EMPTY segment. The adaptor wrote a script from
            # nothing, and the reviewer honestly reported "剧本在'凡是符合条件的
            # 人、农民、上山下乡和回'处中断" — it had been handed a fragment.
            #
            # Skipping was the wrong response to a missing field. The material
            # exists; the model simply failed to copy it. Falling back to the
            # source keeps the run honest: the segment carries the real text, and
            # the defect is recorded instead of propagating as an empty script.
            # How many segments the model returned, and which one this is. Using
            # `len(segments)` (the ones accepted so far) instead of the total made
            # the condition depend on earlier iterations: a second segment lacking a
            # body would take the "single segment" branch whenever the first had
            # been accepted.
            raw_segments = _as_list(raw.get("segments"), 6)

            if len(raw_segments) == 1:
                # A single un-bodied segment: the whole chapter is this segment.
                # The cap scales with the material instead of a fixed 1200, because
                # cutting a long chapter mid-sentence is what produced
                # "剧本在…处中断" in review: the adaptor was writing from a fragment.
                budget = max(1200, min(6000, len(text)))
                seg_text = _clip(text, budget)
            else:
                # Several segments but one without a body: it cannot be guessed,
                # so it is dropped and the loss is recorded rather than hidden.
                ctx.trace_step(
                    "warn",
                    f"第 {i + 1} 段没有正文，已跳过",
                    "模型未把材料分配到这一段；其余段落照常处理",
                    ok=False,
                )
                continue

            if not seg_text:
                # The source itself was empty — nothing to fall back to.
                continue
            ctx.trace_step(
                "warn",
                f"第 {i + 1} 段没有正文，已用原材料补全",
                "模型未把正文放进分段；用用户原文兜底，保证下游拿到真实内容",
                ok=False,
            )
        segments.append(
            {
                # The id is generated, not taken from the model: every downstream
                # stage keys artifacts by it, and a model changing seg1 to seg-1
                # between calls would orphan the whole segment.
                "id": f"seg{i + 1}",
                "title": _clip(d.get("title"), 40) or f"第 {i + 1} 段",
                "text": seg_text,
                "order": i,
                "duration_hint": target_seconds if len(segments) == 0 and not raw.get("segments") else 0,
            }
        )

    if not segments:
        # A single segment is always valid: the pipeline can still run, and failing
        # here would reject a legitimate short piece.
        segments = [{"id": "seg1", "title": "全篇", "text": text[:1200], "order": 0}]
        ctx.trace_step("warn", "模型未给出分段，按单段处理", "", ok=False)

    # ---- what the finished film must contain, and who owes it ----
    #
    # Recorded on the run before any segment starts, so every downstream agent
    # reads the same split. Deriving it per segment (which is what happened when
    # each adaptor wrote its own "era visuals") produced four overlapping
    # whole-film lists and a rework loop nothing could exit.
    requirements, by_segment, unassigned = _parse_requirements(
        raw.get("requirements"), segments
    )
    ctx.state["requirements"] = requirements
    ctx.state["requirements_by_segment"] = by_segment
    ctx.state["requirements_unassigned"] = unassigned

    if requirements:
        ctx.trace_step(
            "decide",
            f"必现项 {len(requirements)} 条，已分配到 {len(by_segment)} 段",
            (
                "；".join(
                    f"{sid}: {len(v)} 条" for sid, v in sorted(by_segment.items())
                )
                or "没有任何一条分配到分段"
            ),
            ok=not unassigned,
        )
        if unassigned:
            ctx.trace_step(
                "warn",
                f"{len(unassigned)} 条必现项没有分配到任何分段",
                "、".join(unassigned[:8]) + "｜不会下发到各段，最终覆盖检查会报缺",
                ok=False,
            )
    else:
        # No split from the director. Say so and let each segment fall back to its
        # own era list — the old behaviour, which at least runs. Silence here would
        # look like "all requirements covered" in the final report.
        ctx.trace_step(
            "warn",
            "导演没有给出必现项分配，退回按段自查",
            "各段将只使用自己改编时给出的时代约束；全片覆盖情况无法核对",
            ok=False,
        )

    plan = {
        "project_id": ctx.brief.get("project_id") or ctx.run_id,
        "task_id": format_id("task", 1),
        "input_type": input_type,
        "operation": operation,
        "workflow": workflow,
        "source_content": text,
        "target_duration": target_seconds,
        "aspect_ratio": _clip(raw.get("aspect_ratio"), 10) or "16:9",
        "style": _clip(raw.get("style"), 80),
        "segments": [{"id": s["id"], "title": s["title"], "order": s["order"]} for s in segments],
        "notes": _clip(raw.get("notes"), 120),
    }

    ctx.state["production_plan"] = plan
    ctx.state.setdefault("source_content", text)

    # The material verdict is what the UI shows as a banner. Its absence would
    # make a run on unsuitable material look like it simply produced a dull film.
    material_check = {
        "ok": content_ok,
        "problems": problems,
        "advice": advice,
    }

    if not content_ok:
        ctx.trace_step(
            "warn",
            "材料提示：" + (advice or problems[0] if problems else "可能不适合做成人物视频"),
            "；".join(problems[:3]),
            ok=False,
        )

    return {
        "segments": segments,
        "production_plan": plan,
        "material_check": material_check,
        "director_summary": (
            f"{input_type} / 工作流 {workflow or '?'}：切 {len(segments)} 段"
            + ("" if content_ok else "（材料提示见下）")
        ),
    }


def _chapter_text(ctx: AgentContext, limit: int = 6000) -> str:
    """The material the director reads, in full up to a sanity ceiling.

    2600 was tuned for an average pasted chapter and became a silent cut-off for
    knowledge-base material: an entry that runs to 4000 characters lost its last
    third — usually the "结果与影响" part — before the director ever saw it, so a
    more detailed entry produced exactly the same film as a short one. The ceiling
    still exists (director latency scales with input), but it now sits above a
    detailed entry rather than inside it.
    """
    for key in ("chapter_text", "segment_text", "chapter"):
        value = ctx.state.get(key) or ctx.input.get(key)
        if isinstance(value, str) and value.strip():
            return value[:limit]
    raise ValueError("没有收到正文")
