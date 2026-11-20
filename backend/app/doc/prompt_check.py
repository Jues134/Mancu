"""Validation of shot prompts before they are generated.

Purpose: a shot prompt is the only input to a paid, slow generation call, so a
bad prompt wastes both. The visual stage checks it first and reports problems
instead of rendering — and the report is what the critic uses to send the
storyboard back for a rewrite.

The checks, and why each exists (all observed as real failure modes):

  1. empty          — a shot with no visual description renders as the previous
                      shot again, which reads as a duplicate in the film.
  2. too short      — under ~15 characters the model has too little to draw and
                      invents: a "长衫" prompt produced a generic modern man.
  3. forbidden      — an era prohibition appearing in the prompt is a hard
                      history defect, and the critic already scores for it.
  4. era conflict   — a word naming another era ("手机", "西装", "汽车") in a
                      镜头 set in 1919. Distinct from `forbidden`: this catches
                      words the historian did not list.
  5. no subject     — neither a person nor an action nor a place. A prompt made of
                      mood words only ("苍凉的氛围") yields an empty frame.
  6. no timeline    — a shot of 4 seconds or more written as one continuous
                      paragraph. The model then spreads the seconds by instinct
                      and performs everything in the first one; beats ("0–2 秒…")
                      are what keep the action distributed across the shot.
  7. name missing   — a character declared in `characters` that the prompt never
                      names. The reference legend is written from those names, so
                      the sheet arrives with nothing to bind it to: the picture is
                      paid for and ignored, and the shot invents a face.
  8. inner transition — a dissolve, wipe or fade-to-black written inside one shot.
                      Those are edits. An i2v model asked for 叠化 cross-fades every
                      frame and the middle of the clip turns to mush.
  9. same opener    — many shots starting with the same phrase. Invisible to the
                      per-shot checks above, because no single prompt is wrong;
                      only the list as a whole is.

Each problem is returned as a `PromptIssue` with a fix hint, so a rejection is
actionable rather than a bare score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


#: Below this length the prompt is treated as under-specified. Measured against
#: the storyboard's own outputs: a usable shot description runs 30-60 characters.
MIN_PROMPT_CHARS = 15

#: A time range such as "0–1.5 秒", "2-4秒" or "3 至 5 秒". Matched loosely on
#: purpose: the point is that the prompt divides the shot into beats, not that it
#: uses one particular dash.
TIMELINE_RE = re.compile(r"\d+(?:\.\d+)?\s*[–\-~—至到]\s*\d+(?:\.\d+)?\s*秒")

#: Words that name a period later than the subjects of this project's material.
#: Not exhaustive by design — the historian's `forbidden` list is authoritative,
#: and this is the safety net for words it did not enumerate.
ERA_CONFLICT_WORDS = (
    "手机", "电脑", "电视", "空调", "冰箱", "洗衣机", "微波炉",
    "汽车", "轿车", "公交车", "地铁", "高铁", "飞机",
    "西装", "领带", "牛仔裤", "T恤", "运动鞋", "手表", "腕表",
    "电灯开关", "水泥大楼", "玻璃幕墙", "霓虹灯", "广告牌", "二维码",
)

#: Signals that a prompt describes something drawable: a person, a body part, an
#: action, or a place. Presence of any one is enough — a close-up of hands is as
#: valid as a wide shot of a street.
SUBJECT_WORDS = (
    "人", "男", "女", "青年", "老人", "学生", "官员", "孩童",
    "脸", "手", "眼", "背影", "身影",
    "坐", "站", "走", "举", "握", "望", "转", "推", "读", "写", "谈",
    "街", "屋", "室", "楼", "院", "门", "窗", "桌", "椅", "灯", "墙", "道",
)

#: Words that ask the model to perform an edit inside one shot. All three are
#: transitions, not content: an image-to-video model asked for 叠化 does not cut,
#: it cross-fades every frame into the next and the middle of the clip turns to
#: mud. Observed on a merged 10-second shot that dissolved from a street to a
#: document on a desk.
INTERNAL_TRANSITION_WORDS = ("叠化", "淡出至黑", "渐暗至黑", "黑屏", "划像", "渐隐")

#: Words that move the camera to a *different place* inside one shot. Distinct
#: from the transitions above: 叠化 is an edit instruction, 「画面切入室内」 is a
#: silent change of space. A real prompt read "0–3 秒 门外铜环特写 … 6–9 秒 画面
#: 切入室内，代表落座" — nine seconds, two spaces. No i2v model cuts, so it
#: morphs the door into a table. This one is blocking, not a warning: the shot
#: cannot come out at all, only produce thirty frames of porridge.
SPACE_CUT_WORDS = (
    "画面切入", "切入室内", "切入屋内", "切至室内", "切到室内",
    "镜头一转", "镜头切换", "画面一转", "画面转到", "场景切换",
    "另一处", "转至室内", "转到室内", "切至室外", "切到室外",
)

#: Action verbs counted to estimate how much a shot asks for. Overlap with
#: SUBJECT_WORDS is intentional: that list asks "is there anything to draw", this
#: one asks "how many things happen".
ACTION_WORDS = (
    "走", "坐", "站", "举", "握", "望", "转", "推", "读", "写", "谈",
    "放下", "抬起", "低头", "抬头", "伸手", "挥手", "递", "点头", "摇头",
    "合上", "展开", "落座", "起身", "回头", "俯身", "弯腰", "敲", "按",
    "撕", "折", "翻", "点燃", "吹", "擦", "扶", "拉", "提",
)

#: How much a shot can carry, as two separate densities rather than one sum.
#: Adding them together punished ordinary dialogue — two people pushing a paper
#: back and forth is five verbs in eight seconds, which is not "overloaded", it is
#: just a scene. The two failure modes are different and are measured apart:
#   人太多 → 观众分不清谁是谁（10 秒 5 个人已经是上限）
#   事太多 → 每个动作只来得及扫过（每秒 1.6 个动作基本等于什么都没看清）
PEOPLE_PER_SECOND_FAIL = 0.5
ACTIONS_PER_SECOND_WARN = 1.0
ACTIONS_PER_SECOND_FAIL = 1.6

#: Purely atmospheric words. A prompt made only of these has no subject.
MOOD_WORDS = (
    "氛围", "气息", "感觉", "意境", "苍凉", "压抑", "沉重", "希望", "光明",
    "悲壮", "肃穆", "温暖", "冰冷", "时代感",
)


@dataclass
class PromptIssue:
    """One defect in one shot prompt."""

    shot_id: str
    code: str
    detail: str
    fix_hint: str
    severity: str = "fail"    # fail | warn

    def to_critic_issue(self) -> dict[str, Any]:
        """Shape used by the critic's error log, so a rejection is one object."""
        return {
            "shot_id": self.shot_id,
            "kind": self.code,
            "severity": self.severity,
            "detail": self.detail,
            "fix_hint": self.fix_hint,
            "stage": "storyboard",
        }

    def line(self) -> str:
        return f"  [{self.severity}] {self.shot_id} {self.code}：{self.detail} → {self.fix_hint}"


@dataclass
class PromptCheck:
    """The result of checking every shot in a storyboard."""

    issues: list[PromptIssue] = field(default_factory=list)
    checked: int = 0

    @property
    def blocking(self) -> list[PromptIssue]:
        return [i for i in self.issues if i.severity == "fail"]

    @property
    def ok(self) -> bool:
        """Usable if no shot has a blocking defect."""
        return not self.blocking

    def summary(self) -> str:
        if not self.issues:
            return f"检查 {self.checked} 条提示词：全部合理"
        fails = len(self.blocking)
        warns = len(self.issues) - fails
        parts = []
        if fails:
            parts.append(f"{fails} 条需打回")
        if warns:
            parts.append(f"{warns} 条建议修改")
        return f"检查 {self.checked} 条提示词：" + "，".join(parts)


def check_shot(
    shot: dict[str, Any],
    *,
    forbidden: list[str] | None = None,
    era: str = "",
) -> list[PromptIssue]:
    """Check one shot's visual prompt against the five rules."""
    # The storyboard calls this field `shot_id`; older boards and hand-built
    # fixtures use `id`. Reading only `id` stamped every finding with "?", so a
    # batch of identical defects from different shots rendered as one repeated
    # line and the reader could not tell which shot to fix.
    shot_id = str(shot.get("shot_id") or shot.get("id") or "?")
    # Check the text that actually reaches the video model. `video_prompt` is what
    # gets sent; `visual_description` is the shorter human-facing summary. Judging
    # the summary rejected shots whose real prompt was fine, and — worse — passed
    # shots whose rewritten prompt had a defect the summary did not show.
    prompt = (shot.get("video_prompt") or shot.get("visual_description") or "").strip()
    issues: list[PromptIssue] = []

    # 1. empty
    if not prompt:
        issues.append(
            PromptIssue(
                shot_id, "prompt_empty",
                "画面描述为空，生成时会退化成上一个镜头",
                "补写这个镜头拍什么：谁、在哪、在做什么",
            )
        )
        return issues      # nothing else can be judged

    # 2. too short
    if len(prompt) < MIN_PROMPT_CHARS:
        issues.append(
            PromptIssue(
                shot_id, "prompt_too_short",
                f"画面描述只有 {len(prompt)} 字（下限 {MIN_PROMPT_CHARS}），模型容易自由发挥",
                "补充场景、人物动作、光线，写到 30 字左右",
            )
        )

    # 3. forbidden, from the historian's list
    for item in forbidden or []:
        bare = str(item).split("（")[0].strip()
        if bare and bare in prompt:
            issues.append(
                PromptIssue(
                    shot_id, "prompt_forbidden",
                    f"画面描述出现禁现元素「{bare}」",
                    f"删掉「{bare}」，换成同时代的陈设",
                )
            )

    # 4. era conflict, words the historian may not have listed
    hits = [w for w in ERA_CONFLICT_WORDS if w in prompt]
    if hits:
        issues.append(
            PromptIssue(
                shot_id, "prompt_era_conflict",
                f"画面描述出现时代错位词「{'、'.join(hits)}」"
                + (f"（本片为{era}）" if era else ""),
                "改用该时代存在的物品，或把描述改到不涉及该物件",
                severity="warn" if len(hits) == 1 else "fail",
            )
        )

    # 5. no drawable subject
    has_subject = any(w in prompt for w in SUBJECT_WORDS)
    mood_only = any(w in prompt for w in MOOD_WORDS) and not has_subject
    if not has_subject:
        issues.append(
            PromptIssue(
                shot_id, "prompt_no_subject",
                "画面描述只有情绪词，没有可画的人物或场景"
                if mood_only else "画面描述缺少人物、动作或场景，无法构图",
                "至少写出一个主体：谁在做什麼，或在什么地方",
            )
        )

    # 6. no timeline
    #
    # A shot written as one paragraph lets the model distribute the seconds by
    # instinct: it performs the whole action in the first second and then holds a
    # near-still frame for the rest. The prompt now asks for beats
    # ("0–1.5 秒：…"), so a prompt without any time range is the same defect
    # coming back through a rework. Warn, not fail: a two-second shot with one
    # action legitimately needs no beat split.
    if shot.get("duration") and float(shot.get("duration") or 0) >= 4 and not TIMELINE_RE.search(prompt):
        issues.append(
            PromptIssue(
                shot_id, "prompt_no_timeline",
                f"{shot.get('duration')} 秒的镜头没有时间轴，模型会自己分配节奏"
                "（常见结果是前一秒演完、后面几秒几乎不动）",
                "按 0–2 秒 / 2–4 秒 这样分 2～3 拍，每拍写清动作、表情、环境变化",
                severity="warn",
            )
        )

    # 7. a cut written inside a single shot
    #
    # Merging two shots is right; merging two *spaces* is not. A merged 10-second
    # shot came back as "street … 画面逐渐叠化 … a treaty lying on a desk": the
    # model was asked to perform a dissolve, which no i2v model does — it produces
    # a mush in the middle of the shot instead of a clean cut. Dissolves, fades to
    # black and wipes belong to the edit, so they are asked for here by mistake.
    # `MONTAGE` is the one shot type that legitimately holds discontinuous images.
    if str(shot.get("link_type") or "").upper() != "MONTAGE":
        hit = [w for w in INTERNAL_TRANSITION_WORDS if w in prompt]
        if hit:
            issues.append(
                PromptIssue(
                    shot_id, "prompt_internal_transition",
                    f"单镜里写了转场动作「{hit[0]}」：叠化/黑场/划像属于剪辑，"
                    "模型只会把画面演成一团糊",
                    "一镜只演一个空间；要换场就切一镜，把转场交给合成",
                    severity="warn",
                )
            )

    # 8. a character the shot declares, but the prompt never names
    #
    # The reference legend is built from these names ("图片2为老周的人物设定"), so a
    # body that only says 「他」「那位师傅」「姑娘」 gives the model no way to bind the
    # sheet to a person. The picture is sent, paid for, and ignored — the shot then
    # invents a face while every log line looks normal. Cheap to check, expensive
    # to miss.
    declared = [str(n).strip() for n in (shot.get("characters") or []) if str(n).strip()]
    unnamed = [n for n in declared if n not in prompt]
    if unnamed:
        issues.append(
            PromptIssue(
                shot_id, "prompt_name_missing",
                f"本镜标了人物「{'、'.join(unnamed)}」，但正文里没有出现这个名字，"
                "参考图注对不上人",
                "正文第一次提到该人物时写全名（与 characters 逐字一致），之后可以用简称",
                severity="warn",
            )
        )

    # 9. two spaces inside one shot
    #
    # A nine-second prompt read "门外铜环特写 … 画面切入室内，代表落座". The first
    # six seconds are outdoors, the last three are in the room, and there is no cut
    # in between — because the model cannot cut. It blends the door into the table.
    # Blocking: unlike a clumsy description, this shot has no version that works.
    if str(shot.get("link_type") or "").upper() != "MONTAGE":
        cut = [w for w in SPACE_CUT_WORDS if w in prompt]
        if cut:
            issues.append(
                PromptIssue(
                    shot_id, "prompt_space_cut",
                    f"一个镜头里换了空间（「{cut[0]}」）：前一半在别处，后一半在新地方，"
                    "中间没有剪辑点，模型只会把两处糊成一团",
                    "一镜只演一个空间。要换场就切成两个镜头，把转换交给合成",
                )
            )

    # 10. too much asked for in the time available
    #
    # A ten-second shot carried thirteen named delegates, two foreigners' faces,
    # two empty seats and a letter — five things in three beats. Each gets scanned,
    # none is seen. The count is crude on purpose: it only needs to catch the shots
    # that are obviously overloaded, not grade the rest.
    seconds = float(shot.get("duration") or 0)
    if seconds >= 3:
        actions = sum(1 for w in ACTION_WORDS if w in prompt)
        people_rate = len(declared) / seconds
        action_rate = actions / seconds
        detail = ""
        severity = ""
        if people_rate >= PEOPLE_PER_SECOND_FAIL:
            detail = f"{seconds:g} 秒里 {len(declared)} 个人，观众分不清谁是谁"
            severity = "fail"
        elif action_rate >= ACTIONS_PER_SECOND_FAIL:
            detail = f"{seconds:g} 秒里 {actions} 个动作，每个只来得及扫过"
            severity = "fail"
        elif action_rate >= ACTIONS_PER_SECOND_WARN:
            detail = f"{seconds:g} 秒里 {actions} 个动作，偏挤"
            severity = "warn"
        if severity:
            issues.append(
                PromptIssue(
                    shot_id, "prompt_overloaded", detail,
                    "砍掉次要人物和次要动作，或把这个镜头拆成两镜；"
                    "观众一眼能记住的东西，一镜只放一件",
                    severity=severity,
                )
            )

    return issues


def check_storyboard(
    board: dict[str, Any],
    *,
    forbidden: list[str] | None = None,
    era: str = "",
    max_seconds: float = 10.0,
) -> PromptCheck:
    """Check every shot in a storyboard."""
    shots = (board or {}).get("shots") or []
    result = PromptCheck(checked=len(shots))
    for shot in shots:
        result.issues.extend(check_shot(shot, forbidden=forbidden, era=era))
    result.issues.extend(_check_repeated_openers(shots))
    result.issues.extend(_check_fragments(shots, max_seconds=max_seconds))
    return result


#: How many characters at the start of a prompt are compared. Six is enough to
#: catch "1977年深秋傍晚" / "1977年深秋，" while leaving room for genuinely
#: different openings that happen to start with the same shot size.
OPENER_CHARS = 6

#: How many shots must share an opening before it counts as boilerplate. Two is
#: a coincidence ("中景" opens many shots); three in one segment is a template.
OPENER_REPEATS = 3


def _check_repeated_openers(shots: list[dict[str, Any]]) -> list[PromptIssue]:
    """Flag prompts that all open with the same phrase.

    A real storyboard was returned with 24 shots, every one of them starting
    "1977年深秋傍晚。". Nothing was wrong with any single prompt — each was
    readable and drawable — so the per-shot checks passed and the defect only
    showed up when a human read the list top to bottom.

    It matters for generation, not just for reading: a video model weights the
    opening heavily, so twelve clips that begin with the same words come back
    with the same framing and the same first second. Variety at the start of the
    prompt is what makes the cut look shot rather than looped.
    """
    heads: list[tuple[str, str]] = []
    for shot in shots:
        text = str(shot.get("video_prompt") or shot.get("visual_description") or "")
        # Punctuation and spaces are dropped so "1977年深秋傍晚。" and
        # "1977年深秋傍晚，" count as the same opening.
        bare = re.sub(r"[\s，。、；：,..;:！!？?]", "", text)
        heads.append((str(shot.get("shot_id") or "?"), bare[:OPENER_CHARS]))

    counts: dict[str, int] = {}
    for _, head in heads:
        if head:
            counts[head] = counts.get(head, 0) + 1

    issues: list[PromptIssue] = []
    for shot_id, head in heads:
        if head and counts.get(head, 0) >= OPENER_REPEATS:
            issues.append(
                PromptIssue(
                    shot_id, "prompt_same_opener",
                    f"开场「{head}…」与另外 {counts[head] - 1} 个镜头相同",
                    "换成本镜独有的景别或动作开头；"
                    "时间地点只在场景变化时交代一次",
                    severity="warn",
                )
            )
    return issues


#: A shot this short with nothing to say is almost always half of a shot.
#: Two-and-a-half seconds is long enough to be a deliberate insert and short
#: enough that a run of them reads as a stutter in the finished film.
FRAGMENT_SECONDS = 2.5


def _check_fragments(shots: list[dict[str, Any]], *, max_seconds: float = 10.0) -> list[PromptIssue]:
    """Flag shots that look like they were split off a neighbour.

    A real board came back with eleven shots for 33 seconds of narration, among
    them 5s of a worker tying a thread and then a 2s shot of the same workshop's
    window: same scene, same locked-off camera, the action never stopped. One 7s
    shot was written as two, and the cut is visible in the finished film as a jump
    that means nothing.

    Two shapes, both warnings — the storyboard can be right and this heuristic
    wrong, so it must not block a render:

      · **mergeable pair** — two consecutive shots in one scene whose declared link
        is continuous, totalling no more than the model's own per-shot ceiling.
      · **fragment** — a very short shot with no narration and no dialogue. It has
        no reason to exist on its own; it belongs inside its neighbour.
    """
    issues: list[PromptIssue] = []
    chainable = {"CONTINUOUS_ACTION", "SAME_SCENE_CUT"}

    for index, shot in enumerate(shots):
        duration = float(shot.get("duration") or 0)
        words = len(str(shot.get("narration") or "")) + sum(
            len(str(line.get("text") or line.get("line") or "")) if isinstance(line, dict)
            else len(str(line))
            for line in (shot.get("dialogue") or [])
        )
        if 0 < duration <= FRAGMENT_SECONDS and words == 0:
            issues.append(
                PromptIssue(
                    str(shot.get("shot_id") or "?"), "shot_fragment",
                    f"{duration:g} 秒的空镜，没有旁白也没有台词，单独成镜没有理由",
                    "并进相邻镜头：写进那一镜的「环境随时间变化」，"
                    "或作为它最后 1～1.5 秒的画面",
                    severity="warn",
                )
            )

        if index == 0:
            continue
        prev = shots[index - 1]
        same_scene = (
            str(shot.get("scene_id") or "")
            and str(shot.get("scene_id")) == str(prev.get("scene_id"))
        )
        link = str(shot.get("link_type") or "")
        total = float(prev.get("duration") or 0) + duration
        if same_scene and (link in chainable or not link) and 0 < total <= max_seconds:
            issues.append(
                PromptIssue(
                    str(shot.get("shot_id") or "?"), "shot_mergeable",
                    f"与上一镜 {prev.get('shot_id')} 同场景、动作连续，"
                    f"合计 {total:g} 秒，本可以是一个镜头",
                    "合并成一镜，用时间轴分 2～3 拍；"
                    "景别不同就用缓慢推拉完成变化，而不是切一个镜头",
                    severity="warn",
                )
            )
    return issues
