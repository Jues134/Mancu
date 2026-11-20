"""Tell the user exactly what this run can and cannot produce, before and after it.

The rule: **a capability that is missing must be stated, not silently degraded.**
Three situations produce silently wrong deliverables, and all three are failures
of reporting rather than of execution:

  1. **The run is in prompt-only mode.** It produces prompts and no media. That is
     a legitimate mode, but a user who expected a film must be told which mode they
     are in — otherwise the missing film looks like a bug.
  2. **A model cannot do something.** A video model that cannot produce audio, an
     image model that cannot do four views, a provider with no video model at all.
     The stage degrades gracefully, which is correct, but the user has to know the
     result is a fallback rather than the model's output.
  3. **Nothing at all is configured.** No image model, no video model. The run still
     produces prompts, and the user must not think generation simply failed.

Design: one report, built from what the gates already record (`__media_skip` and
the capability descriptors), produced at the start of a run so the user knows what
to expect, and again at the end so the record shows what actually happened.

Why a dedicated module rather than a sentence in a stage: the same facts are needed
by the run banner, the conversation view, the capability endpoint and the review
stage. Building it once keeps those four from disagreeing, which is exactly how
"the UI says it can generate audio but the film is silent" happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


#: Severity of a limitation. `info` is a mode the user chose; `warn` is a
#: degradation they should know about; `fail` means the deliverable is materially
#: different from what the run appeared to promise.
INFO = "info"
WARN = "warn"
FAIL = "fail"


@dataclass
class Limitation:
    """One thing this run cannot do, in words the user can act on."""

    #: Which capability: "video", "audio", "image", "four_view", "text".
    capability: str
    severity: str
    #: A full sentence. Names the model where that helps the user switch it.
    message: str
    #: What the run does instead. Empty when the answer is "nothing".
    fallback: str = ""
    #: The model this concerns, for the record.
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "severity": self.severity,
            "message": self.message,
            "fallback": self.fallback,
            "model": self.model,
        }


@dataclass
class CapabilityReport:
    """What this run can produce, and what it cannot."""

    prompt_only: bool = False
    limitations: list[Limitation] = field(default_factory=list)
    #: The one-line banner the UI shows at the top of a run.
    banner: str = ""
    #: Everything the user must be told, most severe first.
    notices: list[str] = field(default_factory=list)

    def worst(self) -> str:
        """The most severe limitation's severity, or "" when there are none."""
        for level in (FAIL, WARN, INFO):
            if any(x.severity == level for x in self.limitations):
                return level
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_only": self.prompt_only,
            "worst": self.worst(),
            "banner": self.banner,
            "notices": self.notices,
            "limitations": [x.to_dict() for x in self.limitations],
        }


def build_report(
    *,
    prompt_only_mode: bool,
    generate_images: bool,
    generate_videos: bool,
    image_model: str = "",
    video_model: str = "",
    video_caps: dict[str, Any] | None = None,
    image_caps: dict[str, Any] | None = None,
) -> CapabilityReport:
    """Assemble the report from the switches and the model descriptors.

    Every branch that appends a limitation also sets a `fallback`, because "cannot
    do X" without "here is what happens instead" leaves the user unable to judge
    whether the result is usable.
    """
    caps_v = video_caps or {}
    caps_i = image_caps or {}
    report = CapabilityReport(prompt_only=prompt_only_mode)

    def add(capability: str, severity: str, message: str, fallback: str = "", model: str = "") -> None:
        report.limitations.append(
            Limitation(
                capability=capability,
                severity=severity,
                message=message,
                fallback=fallback,
                model=model,
            )
        )

    # ---- 1. prompt-only mode ----
    #
    # Not a limitation of the model — a mode the user selected. Still reported,
    # because "no film" must never be discovered by looking for one.
    if prompt_only_mode:
        add(
            "video", INFO,
            "当前为提示词模式：本轮只产出提示词与校验，不生成图片和视频。",
            "产出完整的图片/视频提示词，可直接复制到其他工具执行。",
        )
        # In prompt-only mode the per-model gaps are moot: nothing is being
        # generated, so listing them would bury the one fact that matters.
        report.banner = _banner(report)
        report.notices = _notices(report)
        return report

    # ---- 2. video generation ----
    if not generate_videos:
        add(
            "video", INFO,
            "未勾选「出视频」：镜头只产出提示词，不渲染视频。",
            "每个镜头都生成完整的视频提示词。",
        )
    elif not video_model:
        add(
            "video", FAIL,
            "没有配置可用的视频模型，本轮无法生成视频。",
            "镜头只产出提示词，不会渲染成片；请在设置中配置视频模型。",
        )

    # ---- 3. audio ----
    #
    # The video model producing sound is a separate capability from producing
    # pictures, and it is the one most often absent. Stated here as well as in
    # `audio_plan` so the banner and the stage agree.
    if generate_videos and video_model:
        native_audio = caps_v.get("native_audio")
        if native_audio is False:
            add(
                "audio", WARN,
                f"当前视频模型（{video_model}）没有生成音频的能力。",
                (
                    "旁白由本地语音合成（edge-tts / Windows SAPI）配音。"
                    if _tts_ok()
                    else "本机语音合成也不可用，成片将没有声音。"
                ),
                model=video_model,
            )
        elif native_audio is None:
            add(
                "audio", WARN,
                f"无法确认视频模型（{video_model}）是否具备音频生成能力。",
                (
                    "已按不支持处理：旁白由本地语音合成配音。"
                    "如该模型实际支持原生音频，可在模型配置中声明。"
                    if _tts_ok()
                    else "已按不支持处理，且本机语音合成不可用，成片将没有声音。"
                ),
                model=video_model,
            )

    # ---- 4. image generation ----
    if not generate_images:
        add(
            "image", INFO,
            "未勾选「出图」：人物与场景只产出提示词。",
            "四视图与场景提示词照常产出，可直接复制执行。",
        )
    elif not image_model:
        add(
            "image", FAIL,
            "没有配置可用的生图模型，本轮无法生成定妆照与场景图。",
            "只产出提示词；人物一致性失去定妆照作为锚点。",
        )
    else:
        # Four-view generation is the one image capability whose absence changes
        # the deliverable rather than just its quality.
        four_view = caps_i.get("four_view")
        if four_view is False:
            add(
                "four_view", WARN,
                f"当前生图模型（{image_model}）不支持一次生成四视图。",
                "改为提供可分开执行的四张独立提示词，需分别生成。",
                model=image_model,
            )

    # ---- 5. text ----
    #
    # Without an LLM nothing runs at all, so this is a hard failure rather than a
    # limitation of output quality.
    if caps_v.get("llm_missing"):
        add(
            "text", FAIL,
            "未配置大模型 Key，无法推理与调度。",
            "无；请设置 STUDIO_LLM_API_KEY 后重新开始。",
        )

    report.banner = _banner(report)
    report.notices = _notices(report)
    return report


def _tts_ok() -> bool:
    """Whether local synthesis can actually produce a file on this machine."""
    try:
        from .audio_plan import tts_available

        return tts_available()
    except Exception:  # noqa: BLE001 - a probe failure means "assume not"
        return False


def _banner(report: CapabilityReport) -> str:
    """The single line shown above a run.

    Empty when there is nothing to say — a fully capable run should not carry
    a banner explaining that everything works.
    """
    if not report.limitations:
        return ""

    worst = report.worst()
    # Failures get the headline; degradations get a count. Listing every item in
    # the banner would make it a paragraph nobody reads.
    failures = [x for x in report.limitations if x.severity == FAIL]
    if failures:
        return "；".join(x.message for x in failures[:2])

    warns = [x for x in report.limitations if x.severity == WARN]
    if warns:
        return "；".join(x.message for x in warns[:2])

    infos = [x for x in report.limitations if x.severity == INFO]
    return infos[0].message if infos else ""


def _notices(report: CapabilityReport) -> list[str]:
    """Every limitation as one sentence: what is missing, and what happens instead."""
    ordered = sorted(
        report.limitations,
        key=lambda x: {FAIL: 0, WARN: 1, INFO: 2}.get(x.severity, 3),
    )
    out: list[str] = []
    for item in ordered:
        text = item.message
        if item.fallback:
            text = f"{text} 替代方案：{item.fallback}"
        out.append(text)
    return out


def merge_audio_notice(report: CapabilityReport, audio_notice: str) -> CapabilityReport:
    """Fold the audio stage's own notice into the run report.

    `audio_plan` decides at generation time what the model actually did, which can
    differ from what was predicted (a model may refuse audio, or TTS may be missing
    on this machine). Adding it here keeps one place for the user to look rather
    than two partially-overlapping banners.
    """
    text = (audio_notice or "").strip()
    if not text:
        return report
    if any(item.capability == "audio" and text in item.fallback for item in report.limitations):
        return report
    if text not in report.notices:
        report.notices.append(text)
    return report
