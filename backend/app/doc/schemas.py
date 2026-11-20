"""Typed hand-offs for the active documentary production pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ============================================================================
# 0. Director — scheduling only
# ============================================================================


class SegmentStatus(str, Enum):
    NEW = "new"
    ANNOTATED = "annotated"      # 考据完成
    STORYBOARDED = "storyboarded"
    DESIGNED = "designed"        # 角色/画风已定
    GENERATED = "generated"      # 镜头已生成
    VOICED = "voiced"
    APPROVED = "approved"
    REJECTED = "rejected"


class StorySegment(BaseModel):
    """One independently schedulable unit of the source text.

    Splitting the input is the director's first act, because it is the only way
    it can keep working while a sub-agent is busy. One chapter cannot be
    annotated and storyboarded at once; four sections can.
    """

    id: str
    title: str = ""
    text: str
    #: Position in the source, for ordering the final cut.
    order: int = 0
    status: SegmentStatus = SegmentStatus.NEW
    #: Attempts at each stage, so the director can stop a segment that keeps failing.
    attempts: dict[str, int] = Field(default_factory=dict)


class MaterialCheck(BaseModel):
    """Verdict on whether the source can become a character-driven video.

    This exists because it is the most common real failure: a passage with many
    events and no people yields a slideshow of narration over generic scenery.
    Reporting that up front is more useful than producing it.
    """

    ok: bool = True
    #: Events per 100 characters — high density means the text is a chronicle, not
    #: a story, and a character video has nothing to show.
    event_density: float = 0.0
    has_characters: bool = False
    has_scenes: bool = False
    problems: list[str] = Field(default_factory=list)
    advice: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "event_density": round(self.event_density, 2),
            "has_characters": self.has_characters,
            "has_scenes": self.has_scenes,
            "problems": self.problems,
            "advice": self.advice,
        }


# ============================================================================
# 1. Historian — constraints, not prose
# ============================================================================


class EraConstraint(BaseModel):
    """What must not appear in the imagery for this period.

    Framed as prohibitions rather than descriptions because that is how it gets
    used: passed verbatim into the image and video prompts as a negative list.
    A positive description ("wears a robe") is easily ignored; an explicit
    "no modern objects" is what stops an anachronism reaching the render.
    """

    era: str = ""
    year_range: str = ""
    #: Clothing/hair/props that make a shot read as authentic.
    required_visuals: list[str] = Field(default_factory=list)
    #: Things that must never appear (anachronisms).
    forbidden_visuals: list[str] = Field(default_factory=list)
    #: How historical figures may be characterised, to avoid misattribution.
    character_rules: list[str] = Field(default_factory=list)
    color_palette: str = ""
    note: str = ""


class HistoryConstraint(BaseModel):
    segment_id: str
    era: EraConstraint
    #: Key people who must appear, with a one-line identity.
    key_figures: list[dict[str, str]] = Field(default_factory=list)
    #: Facts that must not be contradicted by the imagery.
    hard_facts: list[str] = Field(default_factory=list)


# ============================================================================
# 2. Storyboard — the shot list
# ============================================================================


class Shot(BaseModel):
    """One cut. Everything downstream keys off this."""

    id: str
    segment_id: str
    index: int = 0
    #: Narration spoken over this shot.
    narration: str = ""
    #: What the camera sees, in concrete visual terms (who, where, doing what).
    visual_description: str = ""
    #: Characters present, by name — the visual agent uses these to pick reference
    #: portraits, so a typo here means a wrong face on screen.
    characters: list[str] = Field(default_factory=list)
    shot_type: str = "medium"         # wide | medium | close | detail
    camera_move: str = "static"       # static | push_in | pull_out | pan | handheld
    emotion: str = ""
    seconds: float = 5.0
    #: Negatives injected into the generation prompt.
    avoid: list[str] = Field(default_factory=list)
    #: Model-facing prompt, rewritten for the target video model.
    #:
    #: Kept separate from `visual_description` on purpose: the storyboard describes
    #: what the audience sees, while this carries the duration, the subject action
    #: and the bracket camera command that MiniMax-H3 actually scores. Only this
    #: field is sent to the video endpoint.
    video_prompt: str = ""


class Storyboard(BaseModel):
    segment_id: str
    shots: list[Shot] = Field(default_factory=list)
    #: Total planned runtime.
    total_seconds: float = 0.0
    style_note: str = ""


# ============================================================================
# 3. Character / art direction — the consistency anchor
# ============================================================================


class CharacterSheet(BaseModel):
    """One cast member's locked appearance."""

    name: str
    #: The prompt used to generate the reference portrait. Written in English
    #: photographic terms on purpose: Chinese "写实风格" produced anime art in
    #: measurement, while English cinematography terms produced film stills.
    portrait_prompt: str = ""
    #: Physical description, reused in every shot prompt.
    appearance: str = ""
    #: Local path of the generated reference image.
    portrait_path: str = ""
    #: Same, retrievable by the image-to-video call.
    portrait_url: str = ""
    in_shots: list[str] = Field(default_factory=list)


class StyleBible(BaseModel):
    """Global look, locked once and applied everywhere."""

    #: English photography terms — the difference between a film still and an
    #: illustration, measured on the same prompt in both languages.
    style_prompt: str = (
        "photorealistic cinematic film still, side lighting, shallow depth of field, "
        "35mm film grain, muted color grading, no illustration, no anime"
    )
    negative_prompt: str = (
        "illustration, anime, cartoon, painting, 3d render, watermark, text, "
        "modern objects, distorted face"
    )
    palette: str = ""
    aspect_ratio: str = "16:9"


class ArtBible(BaseModel):
    segment_id: str
    style: StyleBible = Field(default_factory=StyleBible)
    characters: list[CharacterSheet] = Field(default_factory=list)


# ============================================================================
# 4. Visual generation
# ============================================================================


class ShotAsset(BaseModel):
    """A generated shot and its provenance."""

    shot_id: str
    segment_id: str
    path: str = ""
    url: str = ""
    #: Which reference portrait the image-to-video call used. Recorded because it
    #: is the evidence for the consistency claim.
    reference_used: str = ""
    seconds: float = 0.0
    prompt: str = ""
    ok: bool = False
    error: str = ""
    cost_note: str = ""


# ============================================================================
# 5. Voiceover
# ============================================================================


class VoiceClip(BaseModel):
    shot_id: str
    path: str = ""
    seconds: float = 0.0
    text: str = ""
    voice: str = ""
    ok: bool = False
    error: str = ""


# ============================================================================
# 6. Compositor
# ============================================================================


class FinalCut(BaseModel):
    path: str = ""
    seconds: float = 0.0
    shot_count: int = 0
    has_audio: bool = False
    has_subtitles: bool = True
    note: str = ""


# ============================================================================
# 7. Critic — the self-loop
# ============================================================================


class CriticIssue(BaseModel):
    """A rejection reason that a producer can act on."""

    shot_id: str = ""
    stage: str = ""          # historian | storyboard | character | visual
    kind: str = ""           # anachronism | inconsistency | logic | quality
    severity: str = "warn"   # warn | fail
    detail: str = ""
    fix_hint: str = ""


class CriticVerdict(BaseModel):
    approved: bool = False
    issues: list[CriticIssue] = Field(default_factory=list)
    #: The agent the director should re-dispatch to; empty means "nothing to redo".
    rework_target: str = ""
    #: Shot ids that must be regenerated.
    reshoot: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    summary: str = ""
