"""Accepted upload formats, in one place.

The studio accepts three kinds of reference material — images, audio and video —
and the rules differ per kind in ways that matter:

  · **An audio upload is a voice sample bound to a character.** The user is telling
    the system "this is what 陈独秀 sounds like". It is not a generic file; without
    a character binding it cannot be used at all, because nothing knows whose voice
    it is meant to be.
  · **Format detection is by extension AND declared MIME.** Neither alone is
    reliable: a browser reports `.wav` as `audio/wav` but some do not; a `.mp3`
    renamed to `.txt` still declares `audio/mpeg`. Requiring both to agree rejects
    honest uploads on some browsers and still lets a mislabelled file through.
    The rule used here is: the extension must be allowed, and the declared MIME
    must belong to the same kind. That is stricter than trusting one, and looser
    than demanding an exact pair.

  · **Common formats only.** Everything below is what a normal user has on disk.
    Exotic or container formats (`.flac`, `.ogg`, `.mkv`, `.mov`) are excluded
    deliberately: the video providers do not accept all of them, and a file that
    uploads but is rejected later by the model is worse than one rejected now
    with a clear reason.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FileKind:
    """One category of uploadable file."""

    name: str
    #: Allowed extensions, lower-case, with the leading dot.
    extensions: frozenset[str]
    #: MIME types treated as belonging to this kind.
    mime_types: frozenset[str]
    #: Size ceiling for this kind, in bytes.
    max_bytes: int
    #: How the UI should describe it, so the wording lives next to the rule.
    hint: str


IMAGE = FileKind(
    name="image",
    extensions=frozenset({".jpg", ".jpeg", ".png", ".webp"}),
    mime_types=frozenset({"image/jpeg", "image/png", "image/webp"}),
    # Generous: a 4K reference still is large and there is no benefit to rejecting
    # it early — the provider downsamples anyway.
    max_bytes=20 * 1024 * 1024,
    hint="JPG / PNG / WebP",
)

AUDIO = FileKind(
    name="audio",
    # `mp3` and `wav` cover virtually every voice sample a user has. `m4a` is
    # included because that is what a phone records.
    extensions=frozenset({".mp3", ".wav", ".m4a"}),
    mime_types=frozenset({
        "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave",
        "audio/mp4", "audio/x-m4a", "audio/m4a",
    }),
    # A voice sample only needs to be long enough to clone from; 20 MB is minutes
    # of audio and keeps a slow upload from blocking the request.
    max_bytes=20 * 1024 * 1024,
    hint="MP3 / WAV / M4A",
)

VIDEO = FileKind(
    name="video",
    extensions=frozenset({".mp4", ".webm"}),
    mime_types=frozenset({"video/mp4", "video/webm"}),
    # Larger than the others: a reference clip is a few seconds of footage.
    max_bytes=64 * 1024 * 1024,
    hint="MP4 / WebM",
)

#: Every kind the API accepts, keyed by the name used in the request payload.
KINDS: dict[str, FileKind] = {k.name: k for k in (IMAGE, AUDIO, VIDEO)}


class UploadRejected(ValueError):
    """Raised when a file is not acceptable, with a reason fit for the user."""


def infer_kind(filename: str, mime_type: str) -> FileKind:
    """Decide which kind a file is, or raise with a reason a user can act on.

    Order of checks is deliberate: extension first, because that is what the user
    can see and correct; then MIME, because a mismatch means the file was renamed
    or mis-declared, and either way it should not reach a provider.
    """
    from pathlib import Path

    suffix = Path(filename or "").suffix.lower()
    if not suffix:
        raise UploadRejected(
            f"文件「{filename}」没有扩展名，无法判断类型。"
            f"支持：图片 {IMAGE.hint}；音频 {AUDIO.hint}；视频 {VIDEO.hint}"
        )

    for kind in KINDS.values():
        if suffix in kind.extensions:
            declared = (mime_type or "").lower().strip()
            # An empty MIME is common from `curl`-style clients and from some
            # browsers; the extension is then the only evidence, so accept it.
            if not declared:
                return kind
            if declared not in kind.mime_types:
                raise UploadRejected(
                    f"「{filename}」的扩展名是 {suffix}，但声明的类型是 {mime_type}。"
                    f"{kind.name} 允许 {kind.hint}，请确认文件没有被改名。"
                )
            return kind

    raise UploadRejected(
        f"不支持的文件类型 {suffix}。"
        f"支持：图片 {IMAGE.hint}；音频 {AUDIO.hint}；视频 {VIDEO.hint}"
    )


def describe_accepted() -> dict[str, dict[str, object]]:
    """The accepted formats, for the UI so it can state the rules up front."""
    return {
        kind.name: {
            "extensions": sorted(kind.extensions),
            "mime_types": sorted(kind.mime_types),
            "max_mb": round(kind.max_bytes / (1024 * 1024), 1),
            "hint": kind.hint,
        }
        for kind in KINDS.values()
    }
