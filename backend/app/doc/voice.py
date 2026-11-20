"""Voice assets: a user-uploaded sample bound to one character, or the model's own.

The rule from the production spec, in plain terms:

  · The user may upload a voice sample and say which character it belongs to.
  · Uploaded voices are stored as assets with a `voice_profile_id`, so the video
    stage can reference them by id rather than by "the file the user gave us".
  · If no sample was uploaded, the video model generates the voice itself.
  · The two paths must be recorded differently, because "the model spoke in a voice
    we invented" and "the model reproduced a voice the user supplied" are
    different deliverables — one is reproducible, the other is not.

Why binding is mandatory: an audio file with no character is unusable. Nothing
downstream can tell whose voice it is, so the video model would either ignore it
or apply one voice to everyone. Rejecting an unbound sample at upload time is
kinder than silently mis-assigning it later.

Why a sample never silently overrides the model: if a character has a bound
sample, the video stage must use it and must report a failure if the model cannot
accept reference audio, rather than quietly falling back to a generated voice.
A wrong voice is the kind of defect a reviewer hears immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ids import format_id


#: How a voice for a character was decided. Recorded per character so a report can
#: separate reproducible deliverables from model-invented ones.
SOURCE_UPLOADED = "uploaded"
SOURCE_MODEL = "model"


@dataclass
class VoiceProfile:
    """One character's voice decision."""

    voice_profile_id: str
    character_id: str
    character_name: str = ""
    #: `uploaded` or `model`.
    source: str = SOURCE_MODEL
    #: Path or URI of the user's sample. Empty for model-generated voices.
    asset_uri: str = ""
    #: What the sample is meant to convey. A voice reference is not just "the
    #: sound" — pitch, pace and accent may be inherited selectively.
    reference_role: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "voice_profile_id": self.voice_profile_id,
            "character_id": self.character_id,
            "character_name": self.character_name,
            "source": self.source,
            "asset_uri": self.asset_uri,
            "reference_role": self.reference_role,
            "notes": self.notes,
        }


def build_voice_profile(
    *,
    character_id: str,
    character_name: str,
    voice_number: int,
    sample_uri: str = "",
    reference_role: str = "",
) -> VoiceProfile:
    """Create the voice record for one character.

    `source` is derived from whether a sample was supplied, never passed in: a
    caller that could set it independently could claim "uploaded" for a file that
    does not exist, and the whole point of the field is to be trustworthy.
    """
    return VoiceProfile(
        voice_profile_id=format_id("audio", voice_number),
        character_id=character_id,
        character_name=character_name,
        source=SOURCE_UPLOADED if sample_uri else SOURCE_MODEL,
        asset_uri=sample_uri,
        reference_role=reference_role or ("人物音色、语速与口音" if sample_uri else ""),
    )


@dataclass
class VoicePlan:
    """Every character's voice for one project, plus what the video stage must do."""

    profiles: list[VoiceProfile] = field(default_factory=list)

    def bound(self) -> list[VoiceProfile]:
        """Characters whose voice came from an uploaded sample."""
        return [p for p in self.profiles if p.source == SOURCE_UPLOADED]

    def generated(self) -> list[VoiceProfile]:
        """Characters whose voice the model will produce."""
        return [p for p in self.profiles if p.source == SOURCE_MODEL]

    def to_dicts(self) -> list[dict[str, Any]]:
        """Every profile as a plain dict, for the asset manifest."""
        return [p.to_dict() for p in self.profiles]

    def for_video_model(self) -> dict[str, Any]:
        """What the video stage needs to know, in the terms its prompt uses.

        Two separate lists rather than one: a model that accepts reference audio
        uses `reference_audio`, while `generate_audio` covers the rest. Merging
        them would lose the distinction between "use this sample" and "make a
        voice", which is the decision that actually matters downstream.
        """
        refs = [
            {
                "voice_profile_id": p.voice_profile_id,
                "character_id": p.character_id,
                "asset_uri": p.asset_uri,
                "reference_role": p.reference_role,
            }
            for p in self.bound()
        ]
        return {
            "reference_audio": refs,
            "generate_audio": bool(self.generated()),
            "generated_voices": [
                {"voice_profile_id": p.voice_profile_id, "character_id": p.character_id}
                for p in self.generated()
            ],
        }


def plan_voices(
    characters: list[dict[str, Any]],
    uploads: list[dict[str, Any]] | None = None,
    *,
    start_at: int = 1,
) -> VoicePlan:
    """Assign a voice to every character that needs one.

    `characters` come from the consistency agent's `character_assets`; `uploads`
    are the user's audio uploads, each carrying `character_id` (or
    `character_name` when no id exists yet).

    Nothing is invented: a character with no matching upload gets the model
    source. That is the documented default, not an error.
    """
    seq = start_at
    plan = VoicePlan()

    # Match by id first, then by name. The consistency agent may not have assigned
    # ids when the user uploaded (the upload happens before any run), so a name
    # match is the normal path for a first project.
    by_id = {str(u.get("character_id") or ""): u for u in (uploads or []) if u.get("character_id")}
    by_name = {str(u.get("character_name") or ""): u for u in (uploads or []) if u.get("character_name")}

    for character in characters:
        cid = str(character.get("character_id") or character.get("id") or "")
        cname = str(character.get("name") or character.get("character_name") or "")
        upload = by_id.get(cid) or by_name.get(cname) or {}

        plan.profiles.append(
            build_voice_profile(
                character_id=cid,
                character_name=cname,
                voice_number=seq,
                sample_uri=str(upload.get("uri") or upload.get("asset_uri") or ""),
                reference_role=str(upload.get("reference_role") or ""),
            )
        )
        seq += 1

    return plan


def unbound_uploads(
    characters: list[dict[str, Any]], uploads: list[dict[str, Any]] | None
) -> list[str]:
    """Uploads that name a character nobody recognised.

    Returned rather than ignored: the user believes they supplied a voice, and if
    the name was a typo the video will come back in the model's voice with no
    explanation. The caller turns these into a warning on the run.
    """
    known_ids = {
        str(c.get("character_id") or c.get("id") or "") for c in characters
    } - {""}
    known_names = {
        str(c.get("name") or c.get("character_name") or "") for c in characters
    } - {""}

    leftovers: list[str] = []
    for u in uploads or []:
        cid = str(u.get("character_id") or "")
        cname = str(u.get("character_name") or "")
        if cid and cid in known_ids:
            continue
        if cname and cname in known_names:
            continue
        leftovers.append(cname or cid or str(u.get("filename") or "未命名音频"))
    return leftovers
