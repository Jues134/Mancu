"""Produce copyable prompt blocks, in both Chinese and English.

Requirement from the user: the deliverable is a prompt they copy and paste into
another platform. So each block must be

  · complete — a whole prompt, not a fragment the user must assemble
  · self-contained — including the negative list and the technical prefix
  · bilingual — Chinese for 即梦 (Seedance), English for MiniMax/Flux/MJ

Why two versions rather than a translation: measured earlier on image prompts, the
Chinese and English forms steer a model very differently (the Chinese one produced
a Western face and anime styling; the English one produced the described East Asian
actor). Translating one into the other would inherit that gap. So each is written
for its target rather than converted.

Why the assembly lives here rather than in the agents: the prompt is the product, so
its shape deserves one definition that every stage shares. It also keeps the format
stable if a stage's internal representation changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


#: Fixed technical suffix for photorealistic film look. English because that is
#: what steers the model; Chinese style wording was measured to yield anime.
FILM_SUFFIX_EN = (
    "photorealistic cinematic film still, 35mm film grain, muted color grading, "
    "natural skin texture, period-accurate costume"
)

#: The Chinese equivalent, for platforms whose prompts are Chinese.
FILM_SUFFIX_ZH = "写实电影剧照质感，35mm 胶片颗粒，低饱和调色，真实的皮肤质感，符合时代的服饰"

#: Negative terms that keep faces from collapsing into one template — the user's
#: explicit requirement about per-character facial structure.
FACE_NEGATIVE_EN = (
    "identical faces, copy-pasted features, pointed chin, oversized glassy eyes, "
    "doll face, influencer template, over-smoothed skin, plastic skin"
)
FACE_NEGATIVE_ZH = "雷同脸、复制粘贴式五官、尖下巴、超大玻璃眼、幼态娃娃脸、网红脸、过度磨皮、塑料皮肤"

#: Era prohibitions are added per run from the content constraints.
DEFAULT_NEGATIVE_EN = "anime, illustration, cartoon, 2D, deformed, cropped, text, watermark"
DEFAULT_NEGATIVE_ZH = "动漫、插画、卡通、2D、畸形、裁切、文字、水印"


@dataclass
class PromptBlock:
    """One copyable prompt, in both languages."""

    title: str
    en: str
    zh: str
    negative_en: str = ""
    negative_zh: str = ""
    #: Optional note shown next to the block, e.g. "switch off, prompt only".
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "en": self.en,
            "zh": self.zh,
            "negative_en": self.negative_en,
            "negative_zh": self.negative_zh,
            "note": self.note,
        }


def _join(parts: list[str], sep: str = ", ") -> str:
    return sep.join(p.strip().rstrip(",") for p in parts if p and p.strip())


def _negative(era_forbidden: list[str] | None, *, chinese: bool) -> str:
    base = DEFAULT_NEGATIVE_ZH if chinese else DEFAULT_NEGATIVE_EN
    face = FACE_NEGATIVE_ZH if chinese else FACE_NEGATIVE_EN
    parts = [base, face]
    if era_forbidden:
        parts.append("、".join(str(x) for x in era_forbidden[:8]) if chinese
                     else ", ".join(str(x) for x in era_forbidden[:8]))
    return _join(parts, "，" if chinese else ", ")


def character_block(
    *,
    name: str,
    spec: dict[str, str],
    style_en: str = "",
    style_zh: str = "",
    era_forbidden: list[str] | None = None,
    note: str = "",
) -> PromptBlock:
    """Build the four-view character sheet prompt, in both languages.

    The layout sentence is included even though this project's current image model
    cannot render four panels: the block is for copying elsewhere, and a capable
    model needs the instruction present.

    Face attributes are listed before clothing because image models weight early
    tokens most; putting costume first measurably weakens the face.
    """
    keys = ("age", "face_shape", "forehead", "jaw", "eyes", "brows", "nose",
            "mouth", "hair", "build", "clothing", "distinctive")
    spec = {k: str(spec.get(k) or "").strip() for k in keys}

    # English: one sentence per group, so a reader can also read it.
    face_en = _join([spec["age"], spec["face_shape"], spec["forehead"], spec["jaw"],
                     spec["eyes"], spec["brows"], spec["nose"], spec["mouth"]])
    body_en = _join([spec["hair"], spec["build"], spec["clothing"], spec["distinctive"]])
    en = (
        "four-view character reference sheet of one single person. "
        "Layout: left a head-and-shoulders close-up of the face; right three "
        "full-body views of the same person — front view, side view, back view. "
        + (f"Face: {face_en}. " if face_en else "")
        + (f"Hair, build and clothing: {body_en}. " if body_en else "")
        + "Same face, same hair and same clothing in all four panels. "
        "Plain light grey seamless background, full body visible head to shoes, "
        "natural body proportions, no scenery. "
        + (style_en or FILM_SUFFIX_EN)
    )

    # Chinese: the same content, phrased the way 即梦 reads best — one flowing
    # description rather than labelled groups.
    face_zh = "，".join(v for v in (spec["age"], spec["face_shape"], spec["forehead"],
                                  spec["jaw"], spec["eyes"], spec["brows"],
                                  spec["nose"], spec["mouth"]) if v)
    body_zh = "，".join(v for v in (spec["hair"], spec["build"], spec["clothing"],
                                  spec["distinctive"]) if v)
    zh = (
        f"角色四视图设定图，同一人。左侧为头肩部面部特写，右侧为正面、侧面、背面全身三视图。"
        + (f"面部：{face_zh}。" if face_zh else "")
        + (f"发型与穿着：{body_zh}。" if body_zh else "")
        + "四视图保持同一张脸、同一发型、同一服装。浅灰纯色无缝背景，全身完整入画含鞋底，自然人体比例，无场景元素。"
        + (style_zh or FILM_SUFFIX_ZH)
    )

    return PromptBlock(
        title=f"角色四视图 · {name}",
        en=en,
        zh=zh,
        negative_en=_negative(era_forbidden, chinese=False),
        negative_zh=_negative(era_forbidden, chinese=True),
        note=note,
    )


def shot_block(
    *,
    shot_id: str,
    seconds: Any,
    shot_type: str,
    camera_move: str,
    visual_prompt: str,
    narration: str = "",
    era_forbidden: list[str] | None = None,
    note: str = "",
) -> PromptBlock:
    """Build a video-shot prompt, in both languages.

    The English side follows MiniMax-H3's documented shape (duration and aspect
    first, then what happens, then camera); the Chinese side is a flowing
    description for 即梦, which reads Chinese natively and has no bracket syntax.
    """
    from .video_prompt import bracket_for, clamp_seconds

    secs = clamp_seconds(seconds)
    bracket = bracket_for(camera_move)
    camera_zh = {
        "[Push in]": "镜头缓慢推进",
        "[Pull out]": "镜头缓慢拉远",
        "[Pan left]": "镜头向左摇",
        "[Pan right]": "镜头向右摇",
        "[Truck left]": "镜头向左平移",
        "[Truck right]": "镜头向右平移",
        "[Pedestal up]": "镜头升高",
        "[Pedestal down]": "镜头降低",
        "[Tilt up]": "镜头仰起",
        "[Tilt down]": "镜头俯下",
        "[Zoom in]": "镜头推近",
        "[Zoom out]": "镜头拉远",
        "[Shake]": "手持轻微晃动",
        "[Tracking shot]": "镜头跟随主体",
        "[Static shot]": "固定镜头",
    }.get(bracket, "固定镜头")

    # `visual_prompt` is already the rewritten, model-facing text, and it may
    # itself start with `5s, 16:9.` — which this builder adds again. Strip the
    # whole prefix (duration + ratio), not just the duration, or the aspect repeats.
    core = (visual_prompt or "").strip()
    prefix = re.compile(rf"^{secs}\s*s\s*,\s*\d+\s*:\s*\d+\s*[.、]?\s*", re.IGNORECASE)
    core = prefix.sub("", core).strip()

    en = (
        f"{secs}s, 16:9. {core} "
        f"Camera: {shot_type or 'medium'} shot, {bracket}. "
        f"{FILM_SUFFIX_EN}"
    )
    zh = (
        f"{secs} 秒，16:9。{core} "
        f"{shot_type or '中景'}，{camera_zh}。"
        f"{FILM_SUFFIX_ZH}"
    )
    return PromptBlock(
        title=f"镜头 {shot_id}",
        en=en,
        zh=zh,
        negative_en=_negative(era_forbidden, chinese=False),
        negative_zh=_negative(era_forbidden, chinese=True),
        note=note,
    )


def collect(blocks: list[PromptBlock]) -> dict[str, Any]:
    """Package blocks for the UI, plus a single copyable text."""
    text_en = "\n\n".join(
        f"### {b.title}\n{b.en}\n\nNegative: {b.negative_en}" for b in blocks
    )
    text_zh = "\n\n".join(
        f"### {b.title}\n{b.zh}\n\n负面词：{b.negative_zh}" for b in blocks
    )
    return {
        "blocks": [b.to_dict() for b in blocks],
        "text_en": text_en,
        "text_zh": text_zh,
        "count": len(blocks),
    }
