"""How a run's audio will be produced, and what to tell the user about it.

The production spec requires "视频及声音生成", but whether the *video model* can
produce sound is a property of the model, not of the pipeline. Three outcomes are
possible, and they need different handling:

  1. **The model generates audio natively.** Its clips arrive with sound.
  2. **The model cannot.** Then something else must voice the narration, or the
     film is silent — and silent narration over historical footage reads as a bug,
     not a style. The fallback is the local TTS chain (edge-tts → Windows SAPI).
  3. **Nobody knows whether it can.** The honest answer is "unknown", which must
     be treated as "cannot" for planning: assuming a capability that is absent
     produces a silent film nobody warned about.

The rule this file encodes is therefore: **when the model cannot be confirmed to
produce audio, say so explicitly and fall back to TTS.** The message is part of
the deliverable — a user who is told "this model has no audio capability, narration
was dubbed locally" knows what they got. A user who is not told believes the model
generated that voice.

Why the message names the model: "音频不可用" alone gives nobody a way to act on it.
"This model appears to have no audio capability (minimax-h3)" tells the user either
to switch models or to accept the dubbed version knowingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


#: The three answers for "can this model make sound".
AUDIO_NATIVE = "native"      # the model produces audio itself
AUDIO_ABSENT = "absent"      # confirmed it cannot
AUDIO_UNKNOWN = "unknown"    # not established either way

#: How the run's audio was actually produced.
VIA_MODEL = "model"
VIA_TTS = "tts"
VIA_USER_SAMPLE = "user_sample"


@dataclass
class AudioPlan:
    """What will voice this film, and what the user must be told."""

    #: What the video model can do, one of the three constants above.
    model_capability: str = AUDIO_UNKNOWN
    #: Whether narration will be synthesised locally.
    needs_tts: bool = True
    #: Characters whose voice comes from a user-uploaded sample.
    user_sampled: int = 0
    #: A message fit for the UI and the run record. Empty when nothing surprising
    #: happened — a native-audio model with no caveat needs no banner.
    notice: str = ""
    #: The same fact in machine-readable form, for the trace and the review stage.
    detail: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.detail is None:
            self.detail = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_capability": self.model_capability,
            "needs_tts": self.needs_tts,
            "user_sampled": self.user_sampled,
            "notice": self.notice,
            "detail": self.detail,
        }


def plan_audio(
    *,
    model: str,
    capability: str,
    user_sampled: int = 0,
    tts_available: bool = True,
) -> AudioPlan:
    """Decide how audio is made, and what to say about it.

    `capability` comes from the video-model descriptor, which returns UNKNOWN for
    any model it has not been established for. UNKNOWN is treated as ABSENT for
    planning while still being reported as unknown, so the notice is accurate:
    "the model's audio capability is unknown, so narration was dubbed locally" is
    a different claim from "the model cannot produce audio".
    """
    needs_tts = capability != AUDIO_NATIVE

    if capability == AUDIO_NATIVE:
        notice = ""
        detail = {"reason": "模型原生生成音频"}
    elif capability == AUDIO_ABSENT:
        notice = (
            f"当前视频模型（{model or '未配置'}）没有生成音频的能力，"
            "旁白由本地语音合成（edge-tts / Windows SAPI）配音。"
        )
        detail = {"reason": "模型不支持原生音频"}
    else:
        notice = (
            f"无法确认视频模型（{model or '未配置'}）是否具备音频生成能力，"
            "已按不支持处理：旁白由本地语音合成配音。"
            "如该模型实际支持原生音频，可在模型配置中声明。"
        )
        detail = {"reason": "能力未知，按不支持处理"}

    if needs_tts and not tts_available:
        # Both fallbacks are gone. Saying so now is better than a silent film
        # that looks like the audio simply failed.
        notice += " 本地语音合成也不可用，成片将没有声音。"
        detail["tts_available"] = False

    if user_sampled:
        detail["user_sampled_voices"] = user_sampled

    return AudioPlan(
        model_capability=capability,
        needs_tts=needs_tts and tts_available,
        user_sampled=user_sampled,
        notice=notice,
        detail=detail,
    )


def capability_from_descriptor(caps: dict[str, Any]) -> str:
    """Read an audio capability out of a model descriptor.

    The descriptor is what `_video_capabilities()` builds: a small dict of what
    the model is known to do. Absence of a key is UNKNOWN, not False — a model
    nobody has probed is not the same as one that has been proven incapable.
    """
    value = caps.get("native_audio")
    if value is True:
        return AUDIO_NATIVE
    if value is False:
        return AUDIO_ABSENT
    # Some descriptors express it as a list of channels they can produce.
    channels = caps.get("audio_channels")
    if isinstance(channels, list) and channels:
        return AUDIO_NATIVE
    return AUDIO_UNKNOWN


def tts_available() -> bool:
    """Whether any local synthesis path actually works on this machine.

    `list_supported` now probes real synthesis rather than checking whether
    `powershell.exe` is on PATH: presence is not capability, and reporting a
    machine as able to speak when it cannot is how a silent film ships with no
    warning.
    """
    try:
        from .tts import list_supported

        supported = list_supported()
        return bool(supported.get("edge_tts") or supported.get("sapi"))
    except Exception:  # noqa: BLE001 - a missing probe must not break planning
        return False
