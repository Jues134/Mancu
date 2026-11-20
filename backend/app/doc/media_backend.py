"""The one interface the pipeline uses to render media, and who implements it.

Why a protocol rather than "just call whatever the config returns":

Three backends must be interchangeable, and each speaks a different dialect:

  · **Local ComfyUI gateway** — asynchronous jobs with a queue, reference audio,
    a reference *clip*, several reference pictures. Polls `/api/jobs`.
  · **Hosted OpenAI-compatible provider** — synchronous image call plus a
    asynchronous video task polled by id. Takes one reference image, no audio.
  · **A hosted key bought later** — same as above, but the endpoint and dialect
    are whatever the vendor sells. The user's expectation is one URL, one key,
    and it works.

Without a single declared shape, the failure is structural and silent: the video
stage was written against `render_video`, the hosted client only offers
`image_to_video`, and the mismatch only surfaces as an AttributeError mid-run —
after the prompts, the storyboard and the assets have all been paid for.

So the pipeline depends on this module's three functions, never on a concrete
client. Each backend adapts to it, and a backend that *cannot* do something says
so through a capability dict rather than pretending.

The split between `render_image` / `render_video` and the capability record is
deliberate: what a backend can do is data (so the UI can warn before a run), while
how to make it do a thing is code. Mixing the two is how "the model cannot make
sound" became a silent film.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MediaBackend(Protocol):
    """What the pipeline requires of any renderer.

    Structural, not nominal: a class satisfies this by having the three methods,
    which is what keeps the hosted client and the local gateway interchangeable
    without either importing the other.
    """

    #: Named in the trace and the capability report, so a run record states which
    #: backend produced its media. Required: an unattributed deliverable is
    #: impossible to reproduce.
    backend_name: str

    async def render_image(
        self,
        *,
        prompt: str,
        out_dir: Any,
        reference_image: str = "",
        size: str = "",
        steps: int = 0,
    ) -> Any:
        """Return an object with `.url`, `.local_path`, `.prompt`, `.seconds`."""

    async def render_video(
        self,
        *,
        prompt: str,
        out_dir: Any,
        seconds: float = 5.0,
        reference_image: str = "",
        reference_audio: str = "",
        reference_video: str = "",
        size: str = "",
        steps: int = 0,
    ) -> Any:
        """Return an object with `.url`, `.local_path`, `.duration`.

        `reference_audio` and `reference_video` are optional for the *caller* to
        supply but not for the backend to honour: one that cannot accept them must
        say so through `capabilities()`, and the caller then decides whether to
        degrade or refuse. Silently ignoring them produced a film with the right
        picture and the wrong (or absent) voice.
        """


#: Fields every backend must report, with the conservative answer for anything
#: unproven. Absence is "unknown", never "yes".
CAPABILITY_KEYS = (
    "image",            # can it render a still at all
    "video",            # can it render a moving shot
    "first_frame",      # can a shot be anchored to a given first frame
    "last_frame",       # …and a given last frame
    "reference_images", # several pictures as identity/style references
    "reference_audio",  # a voice or sound sample as a reference
    "reference_video",  # a clip whose motion is to be followed
    "native_audio",     # the video model produces its own soundtrack
    "max_seconds_per_shot",
    #: How many pictures one request may carry, first frame included. Not a
    #: preference: a prompt that cites 「图片二」 while only one file is sent is a
    #: wrong-person shot, so the citation list is capped by this number.
    "max_reference_images",
    #: How this backend wants its pictures *described*: `by_parameter` (the image
    #: travels as a field, the prompt never mentions it), `by_numbered_prompt`
    #: (Seedance: 「图片一引用的是【老周】的资产图」), or `by_tag` (gateway
    #: `<Picture 1>`). Wrong choice here is not a cosmetic difference: MiniMax
    #: reads 「图片一」 as scene content and renders a shot *of* a photo.
    "image_dialect",
    "backend",          # which adapter this is
    "model",            # which model, for the record
)


def empty_capabilities() -> dict[str, Any]:
    """The conservative record: nothing claimed, nothing denied."""
    return {
        "image": False,
        "video": False,
        "first_frame": False,
        "last_frame": False,
        "reference_images": False,
        "reference_audio": False,
        "reference_video": False,
        # Deliberately absent rather than False: "unknown audio" and "no audio"
        # produce different notices, and claiming the former as the latter is a lie
        # about what was verified.
        "max_seconds_per_shot": 5.0,
        # One picture is the floor every backend meets; a model that takes more
        # says so. `by_parameter` adds no text, so an adapter that forgets to
        # declare a dialect still gets a usable prompt.
        "max_reference_images": 1,
        "image_dialect": "by_parameter",
        "backend": "",
        "model": "",
    }


@dataclass
class BackendSelection:
    """Which backend a run will use, and why.

    `reason` is not decoration: when no backend can render, the user needs to know
    which of the three was missing — no gateway configured, no hosted media model,
    or a model that exists but has no video route.
    """

    backend: Any = None
    name: str = ""
    reason: str = ""
    capabilities: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.capabilities is None:
            self.capabilities = empty_capabilities()

    @property
    def usable(self) -> bool:
        return self.backend is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "usable": self.usable,
            "reason": self.reason,
            "capabilities": dict(self.capabilities or {}),
        }


def describe(backend: Any) -> dict[str, Any]:
    """Read a backend's capability record, tolerating a backend without one.

    A backend that does not implement `capabilities()` is treated as capable of
    what its two render methods imply and nothing more. That is the safe reading:
    it lets an older adapter work while never overstating what it can do.
    """
    caps = empty_capabilities()
    if backend is None:
        return caps

    name = str(getattr(backend, "backend_name", "") or type(backend).__name__)
    caps["backend"] = name

    getter = getattr(backend, "capabilities", None)
    if callable(getter):
        try:
            reported = getter() or {}
        except Exception:  # noqa: BLE001 - a broken report must not break a run
            reported = {}
        for key in CAPABILITY_KEYS:
            if key in reported:
                caps[key] = reported[key]

    # Fall back to what the methods themselves prove: having render_video means
    # video works, whether or not the record said so.
    if hasattr(backend, "render_image") and "image" not in caps:
        caps["image"] = True
    if hasattr(backend, "render_video"):
        caps.setdefault("video", True)

    model = str(getattr(backend, "video_model", "") or getattr(backend, "image_model", ""))
    if model:
        caps["model"] = model
    return caps



def select_backend(
    model_settings: Any = None, *, capability: str = "", agent_id: str = ""
) -> BackendSelection:
    """Pick the media backend for a run: gateway first, hosted second.

    Order and why:

      1. **Local gateway** — no hosted key needed, runs on this machine's GPU, and
         it is the only backend whose video route accepts a reference audio track.
         When it is configured it is what the user intended.
      2. **Hosted provider** — a key plus a URL the user bought. This is the path
         the compatibility layer exists for: it must work with nothing but two
         values entered in the settings dialog.
      3. **Nothing** — the reason is recorded so the capability report can say
         which of the two was missing instead of "no media".

    `models_config.load()` is consulted lazily so a test can pass its own settings
    without touching the on-disk file.
    """
    assigned_id = ""
    assigned_profile = None
    if model_settings is not None and agent_id:
        raw = str((getattr(model_settings, "assignments", {}) or {}).get(agent_id) or "")
        assigned_id = raw.split(":", 1)[0]
        assigned_profile = next(
            (
                p for p in (getattr(model_settings, "providers", []) or [])
                if str(getattr(p, "id", "") or "") == assigned_id
            ),
            None,
        )

    def is_gateway(profile: Any) -> bool:
        if profile is None:
            return False
        return (
            str(getattr(profile, "kind", "") or "").lower() == "gateway"
            or str(getattr(profile, "preset", "") or "").lower() == "gateway"
            or str(getattr(profile, "id", "") or "").lower() in ("gateway", "comfyui")
        )

    # ---- 1. explicitly assigned gateway, or legacy gateway fallback ----
    # A hosted Agent assignment suppresses the generic gateway fallback: the
    # selection in Agent settings is authoritative for this capability.
    try:
        from .gateway_generator import gateway_from_settings

        gateway = (
            gateway_from_settings(model_settings, preferred_provider_id=assigned_id)
            if is_gateway(assigned_profile)
            else None if assigned_profile is not None
            else gateway_from_settings(model_settings)
        )
    except Exception as exc:  # noqa: BLE001 - a broken gateway must not mask hosted
        gateway = None
        gateway_error = str(exc)[:120]
    else:
        gateway_error = ""

    if gateway is not None:
        return BackendSelection(
            backend=gateway,
            name=str(getattr(gateway, "backend_name", "comfyui-gateway")),
            reason="已配置本地 ComfyUI 网关",
            capabilities=describe(gateway),
        )

    # ---- 2. hosted provider ----
    try:
        from .generator import Generator

        hosted = Generator.from_settings(model_settings)
    except Exception as exc:  # noqa: BLE001 - reported as the reason below
        detail = str(exc)[:120]
        return BackendSelection(
            backend=None,
            name="",
            reason=(
                f"未配置可用的媒体后端：{detail}"
                + (f"（网关配置有误：{gateway_error}）" if gateway_error else "")
            ),
        )

    return BackendSelection(
        backend=hosted,
        name=str(getattr(hosted, "backend_name", "hosted")),
        reason="已配置云端的生图/生视频供应商",
        capabilities=describe(hosted),
    )
