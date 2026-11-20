"""How each video model wants its reference images *talked about*.

The same shot has to be described differently depending on the model, and not in
a cosmetic way:

  · **Seedance** treats an image as part of the narration. "图片一为上一镜的最后
    一帧……本视频从图片一的画面直接开始，向下延续" is what makes the first frame
    an anchor instead of a mood board. Omit that sentence and the image is just
    another reference; add it where the model expects it and the shot continues
    from the previous one.

  · **MiniMax** takes the first frame as a *parameter*. Its prompt describes
    action only, with bracketed camera commands (`[Push in]`). Telling it 「图片一
    是上一帧」 puts a picture of a picture into the video.

  · **A gateway** (ComfyUI style) uses tags: `<Picture 1>` … `<Picture 9>`.

Writing one prompt for all three is how you get a prompt that works on exactly
one of them. So the pipeline hands over *facts* — which images, in what order,
what each one is for — and a dialect renders them into the sentence that model
understands. Facts are shared; the wording is not.

This module is deliberately pure: no network, no disk, no state. It can be
tested, and it can be swapped, without a backend being configured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

#: How a model reads the pictures handed to it.
BY_PARAMETER = "by_parameter"        # MiniMax: the image is an API field
BY_NUMBERED_PROMPT = "by_numbered_prompt"  # Seedance: the prompt names 图片N
BY_TAG = "by_tag"                    # gateway: <Picture N> tags

#: Facts about one image. `kind` decides the sentence template; `label` and
#: `detail` fill it. `detail` is short on purpose — a full character sheet pasted
#: into a 5-second action prompt competes with the action for the model's
#: attention, and the action is what has to survive.
@dataclass
class ImageFact:
    kind: str                 # first_frame | character | scene | prop
    label: str = ""
    detail: str = ""
    path: str = ""
    #: True when this picture is the frame the shot continues from. A cut to a new
    #: scene also starts from a picture, but it re-composes rather than continues —
    #: telling the model 「向下延续」 there makes it drag the old room into the new one.
    continuation: bool = False
    #: 1-based position in the request, assigned by `order_images`.
    index: int = 0


#: Characters first, then the places and things they act on. A first frame, when
#: there is one, is always #1: Seedance's "图片一为上一帧" is a contract, and the
#: pipeline must not reorder it away.
KIND_ORDER = {"first_frame": 0, "character": 1, "scene": 2, "prop": 3}

KIND_LABEL = {
    "first_frame": "上一镜的最后一帧",
    "character": "人物设定",
    "scene": "场景参考",
    "prop": "道具参考",
}


def order_images(
    first_frame: "ImageFact | None" = None,
    references: Iterable["ImageFact"] = (),
) -> list[ImageFact]:
    """Assign the 1-based index each model will refer to.

    Order is a contract, not a preference: 「图片二为老周」 only means something
    if the second file really is 老周's sheet. Within a kind the caller's order is
    kept, because that order is what the prompt's own wording was written against.
    """
    facts: list[ImageFact] = []
    if first_frame is not None and first_frame.path:
        facts.append(first_frame)
    facts.extend(r for r in references if r.path)
    facts.sort(key=lambda f: KIND_ORDER.get(f.kind, 9))
    for i, fact in enumerate(facts, start=1):
        fact.index = i
    return facts


class ImageOrderError(RuntimeError):
    """The prompt and the files disagree about which picture is which.

    Raised instead of rendered, on purpose. A misalignment does not fail — it
    produces a plausible shot of the wrong person, and the mistake surfaces only
    when someone notices a face changed three shots later. Refusing the shot costs
    one retry; shipping it costs the whole film.
    """


@dataclass
class ImagePlan:
    """The one list both the prompt and the upload read from.

    `facts` and `files` are the same length, same order, by construction. The
    pipeline may only hand the backend `files`, and may only let a dialect render
    `facts`; nothing else is allowed to renumber or re-sort.
    """

    facts: list[ImageFact] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    #: How many usable references did not fit under the backend's limit.
    dropped: int = 0


def plan_images(
    first_frame: "ImageFact | None" = None,
    references: Iterable["ImageFact"] = (),
    *,
    max_images: int = 1,
) -> ImagePlan:
    """Decide, once, which pictures are sent and which number each one gets.

    Every reference without a real path is skipped *before* numbering, so a
    missing portrait cannot shift the rest by one — that shift is exactly the bug
    that makes 「图片二为老周」 point at the courtyard.
    """
    limit = max(0, int(max_images))
    wanted: list[ImageFact] = []
    if (
        limit >= 1
        and first_frame is not None
        and str(first_frame.path or "").strip()
    ):
        wanted.append(first_frame)
    usable = [r for r in references if str(r.path or "").strip()]
    dropped = 0
    room = limit - len(wanted)
    if len(usable) > max(0, room):
        dropped = len(usable) - max(0, room)
        usable = usable[: max(0, room)]
    wanted.extend(usable)

    facts = order_images(None, wanted)
    plan = ImagePlan(facts=facts, files=[str(f.path) for f in facts], dropped=dropped)
    verify_plan(plan)
    return plan


def verify_plan(plan: ImagePlan) -> None:
    """Assert the invariant before a shot is paid for.

    Cheap, and it runs on every shot: an off-by-one here is invisible in the logs
    and obvious only in the finished video.
    """
    if len(plan.files) != len(plan.facts):
        raise ImageOrderError(
            f"提示词引用了 {len(plan.facts)} 张图，实际要传 {len(plan.files)} 张"
        )
    for expected, fact in enumerate(plan.facts, start=1):
        if fact.index != expected:
            raise ImageOrderError(f"图序错位：第 {expected} 张被编成 #{fact.index}")
        if str(fact.path or "") != str(plan.files[expected - 1] or ""):
            raise ImageOrderError(f"图序错位：#{expected} 的说明与实际文件不是同一张")
        if not str(fact.path or "").strip():
            raise ImageOrderError(f"#{expected} 没有文件，却占了一个号")


def _character_sentence(fact: ImageFact) -> str:
    name = fact.label or "主要人物"
    look = f"：{fact.detail}" if fact.detail else ""
    return (
        f"图片{fact.index}为{name}的人物设定{look}。"
        f"全片保持这张脸、发型与服装不变。"
    )


def _first_frame_sentence(fact: ImageFact) -> str:
    what = fact.label or fact.detail or "本镜的起始画面"
    if fact.continuation:
        return (
            f"图片{fact.index}为{what}。"
            f"本视频从图片{fact.index}的画面直接开始，"
            f"保持人物、服装、光线与构图不变，向下延续。"
        )
    # A cut to a new scene also gets a picture, but it must not inherit the old
    # room: same sentence shape, different instruction.
    return (
        f"图片{fact.index}为{what}。"
        f"本镜以图片{fact.index}的场景布局、光线方向与色调为准，"
        f"人物按后文的描述重新进入画面。"
    )


def _scene_sentence(fact: ImageFact) -> str:
    what = fact.detail or ""
    look = f"：{what}" if what else ""
    return (
        f"图片{fact.index}为{fact.label or '本镜场景'}的场景参考{look}。"
        f"保持环境布局与光线方向不变，画面里的人物按后文描述表演。"
    )


def _prop_sentence(fact: ImageFact) -> str:
    what = fact.detail or ""
    look = f"：{what}" if what else ""
    return (
        f"图片{fact.index}为{fact.label or '本镜道具'}的道具参考{look}。"
        f"保持它的形状、材质与颜色不变。"
    )


SENTENCE_OF = {
    "first_frame": _first_frame_sentence,
    "character": _character_sentence,
    "scene": _scene_sentence,
    "prop": _prop_sentence,
}


def render_numbered(body: str, images: list[ImageFact]) -> str:
    """Seedance:每张图一段话讲清身份与用途，图在前，戏在后。

    The intro block is what turns a reference image into an anchor. It has to come
    before the action, because it sets the terms the action is read under — placed
    after, the model has already decided what the frame looks like.
    """
    if not images:
        return body.strip()
    lines = [SENTENCE_OF.get(f.kind, _scene_sentence)(f) for f in images]
    return "\n".join(lines) + "\n" + body.strip()


def render_tag(body: str, images: list[ImageFact]) -> str:
    """Gateway style: a legend line, then the body that refers to `<Picture N>`."""
    if not images:
        return body.strip()
    legend = "；".join(
        f"<Picture {f.index}> = {f.label or KIND_LABEL.get(f.kind, '参考图')}"
        + (f"（{f.detail}）" if f.detail else "")
        for f in images
    )
    return f"参考图：{legend}。\n{body.strip()}"


def render_parameter(body: str, images: list[ImageFact]) -> str:
    """MiniMax: the pictures travel as fields, so the prompt stays action-only.

    Returns the body untouched — including when images exist. Any mention of
    「图片一」 here would be read as scene content and produce a shot of a photo.
    """
    return body.strip()


RENDERERS = {
    BY_NUMBERED_PROMPT: render_numbered,
    BY_TAG: render_tag,
    BY_PARAMETER: render_parameter,
}


def render(
    body: str,
    images: list[ImageFact],
    *,
    dialect: str = BY_PARAMETER,
) -> str:
    """Render a shot prompt in the model's own way of talking about pictures.

    An unknown dialect falls back to `by_parameter`: it is the only one that never
    adds text, so a misconfigured model gets a plain prompt rather than a broken
    one. The images are still sent as fields where the backend supports them.
    """
    return RENDERERS.get(dialect, render_parameter)(body, images)


def describe(images: list[ImageFact]) -> str:
    """The fact list as a human reads it — used by the trace, not by a model."""
    if not images:
        return "无参考图"
    parts = []
    for f in images:
        kind = KIND_LABEL.get(f.kind, f.kind)
        # A first frame's kind label and its own label are the same words; saying
        # 「上一镜的最后一帧·上一镜的最后一帧」 reads like two different things.
        parts.append(f"#{f.index} {kind}" if f.label in ("", kind) else f"#{f.index} {kind}·{f.label}")
    return "；".join(parts)
