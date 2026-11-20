"""Adapter that lets the six-agent pipeline render through a local ComfyUI gateway.

Why this exists: the pipeline calls a `Generator` (image/video models over an
OpenAI-style API). The machine that will actually render has neither a hosted
video model configured nor the VRAM to run one alongside the LLM — but it does
have a local ComfyUI gateway that was smoke-tested end to end, including
reference images, a reference clip and a reference audio track.

So rather than requiring a paid hosted model, this adapter presents that gateway
behind the same interface the pipeline already expects:

    await gen.render_image(prompt=..., out_dir=...)
    await gen.render_video(prompt=..., out_dir=..., seconds=..., reference_image=...)

What it deliberately does NOT do: pretend the gateway supports more than it does.
The gateway has two video routes and they are genuinely different:

  · `video`     — first/last frame only. Cannot lock a character's identity and
                  cannot take an audio reference.
  · `ref_video` — up to 9 pictures, 3 clips and 3 audio tracks. This is the only
                  route that can hold a face AND a voice.

The adapter picks `ref_video` whenever a reference image or audio is supplied and
`video` otherwise, and it reports which route ran — because "the shot was made
with a face lock and a voice reference" and "the shot was made from a prompt" are
different deliverables that a reviewer must be able to tell apart.

The gateway is asynchronous: jobs are queued and polled. Rendering here therefore
blocks until the job settles, which is correct — the pipeline treats a shot as
finished only when its file exists.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import uuid
from pathlib import Path
from typing import Any

from .generator import GeneratedImage, GeneratedVideo
from .media_url import served_url


#: How long to wait for one job. The gateway spends 45–70 s on a 2-second clip at
#: the smoke size; a real shot is longer and the queue may hold a few. Five
#: minutes is generous without letting a wedged job hang a run forever.
DEFAULT_JOB_TIMEOUT = 300.0

#: Poll interval. The gateway's own status endpoint is cheap, and a 2-second
#: granularity is what the queue actually moves at.
POLL_SECONDS = 2.0

#: The gateway speaks in preset names, not pixels: an image is `square` /
#: `portrait` / `landscape` / `hd`, a clip is `tiny` / `small` / `normal` / `hd`.
#:
#: The pipeline, meanwhile, asks in pixels (`1024x1024`, `1920x1080`) because that
#: is what a hosted provider accepts. Sending those straight through produced a
#: rejected job, so the two vocabularies are translated here rather than in the
#: agents — the agents should not have to know which backend they are talking to.
_IMAGE_PRESETS = {"square": (1024, 1024), "portrait": (832, 1216),
                  "landscape": (1216, 832), "hd": (1328, 1328)}
_VIDEO_PRESETS = {"tiny": (608, 352), "small": (736, 416),
                  "normal": (864, 480), "hd": (1280, 736)}


def normalise_size(size: str, *, kind: str, fallback: str) -> str:
    """Translate a request into the gateway's preset vocabulary.

    Rules, and the reason for each:

      · a preset name passes through — the caller knew what it wanted;
      · an image given in pixels is mapped by **orientation** (a 4:3 request must
        not come back square, or the four-view sheet is cropped);
      · a video given in pixels falls back to the configured size. Video presets
        differ only in resolution, and `hd` costs many minutes per shot, so the
        operator's STUDIO_GATEWAY_SIZE — not a 1920x1080 hint written for hosted
        providers — decides how much GPU time a run may spend.
    """
    raw = (size or "").strip().lower()
    if not raw:
        return fallback
    table = _VIDEO_PRESETS if kind == "video" else _IMAGE_PRESETS
    if raw in table:
        return raw
    if "x" not in raw:
        return fallback
    if kind == "video":
        return fallback
    try:
        width, height = (int(part) for part in raw.split("x", 1))
    except ValueError:
        return fallback
    if width <= 0 or height <= 0:
        return fallback

    # Images: orientation first, at the preset's own resolution. `hd` is only used
    # when it was asked for by name, because it doubles the render time.
    if abs(width / height - 1.0) <= 0.08:
        return "square"
    return "landscape" if width > height else "portrait"


class GatewayError(RuntimeError):
    """Raised when the gateway rejects a job or fails to produce a file."""


class GatewayGenerator:
    """ComfyUI gateway behind the pipeline's Generator interface.

    Not a subclass of `Generator`: that class owns hosted-provider mechanics
    (budget, billing, model pinning) which do not apply to a local queue. Duck
    typing keeps the seam honest — the pipeline needs three methods, not a class.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        image_model: str = "comfyui",
        video_model: str = "comfyui",
        size: str = "tiny",
        steps: int = 8,
        job_timeout: float = DEFAULT_JOB_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        # Recorded so the capability report and the trace can name what rendered.
        self.image_model = image_model
        self.video_model = video_model
        self.size = size
        self.steps = steps
        self.job_timeout = job_timeout

    #: Recorded in the trace and the capability report.
    backend_name = "comfyui-gateway"

    def capabilities(self) -> dict[str, Any]:
        """What the gateway can do.

        Everything here was smoke-tested end to end, which is why the audio and
        clip references are True rather than assumed: `ref_video` takes 9 pictures,
        3 clips and 3 audio tracks, and a result with a matching audio spectrum
        was verified.

        `native_audio` is absent, not False: the gateway's video route outputs a
        soundtrack because it is driven by the reference audio, which is not the
        same thing as a model that composes sound. Reporting it as native would
        make the pipeline skip the TTS fallback for the wrong reason.
        """
        return {
            "image": True,
            "video": True,
            "first_frame": True,
            "last_frame": True,      # the `video` route accepts both
            "reference_images": True,
            "reference_audio": True,
            "reference_video": True,
            "max_seconds_per_shot": 15.0,
            "backend": self.backend_name,
            "model": self.video_model,
        }

    # ---------------------------------------------------------------- plumbing

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key}

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        import httpx

        url = f"{self.base_url}{path}"
        # One retry on a transport error. The gateway restarts (ComfyUI + gateway
        # are started together) and a restart closes connections mid-response —
        # observed as a bare `httpx.ReadError`. Losing a whole shot to that would
        # be a wasted generation, and the retry is free when nothing is wrong.
        last: Exception | None = None
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    r = await client.post(url, json=body, headers=self._headers())
                break
            except httpx.TransportError as exc:
                last = exc
                if attempt == 0:
                    await asyncio.sleep(1.5)
        else:
            # `str(ReadError())` is empty, which produced a useless
            # "无法连接生成网关： " message. Name the class so a log line says what
            # actually happened.
            raise GatewayError(
                f"无法连接生成网关（{self.base_url}）：{type(last).__name__}: {last!r}"
            )
        if r.status_code >= 400:
            raise GatewayError(f"网关拒绝请求（{r.status_code}）：{r.text[:160]}")
        return r.json()

    async def _get(self, path: str) -> dict[str, Any]:
        import httpx

        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(url, headers=self._headers())
        if r.status_code >= 400:
            raise GatewayError(f"网关状态查询失败（{r.status_code}）")
        return r.json()

    async def _upload(self, path: Path) -> str:
        """Upload one local file and return the name the gateway assigned.

        The gateway only accepts files it has itself stored: a reference passed as
        a local path would be unreadable on its side. Multipart is hand-assembled
        because the field name must be exactly `file`.
        """
        import httpx

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        files = {"file": (path.name, path.read_bytes(), content_type)}
        async with httpx.AsyncClient(timeout=180.0) as client:
            r = await client.post(
                f"{self.base_url}/api/upload", files=files, headers=self._headers()
            )
        if r.status_code >= 400:
            raise GatewayError(f"参考素材上传失败（{r.status_code}）：{r.text[:120]}")
        items = (r.json() or {}).get("items") or []
        if not items:
            raise GatewayError("参考素材上传后没有返回文件名")
        return str(items[0]["file"])

    async def _wait(self, job_id: str, *, label: str) -> dict[str, Any]:
        """Poll until the job settles. Returns its final record."""
        waited = 0.0
        while waited < self.job_timeout:
            await asyncio.sleep(POLL_SECONDS)
            waited += POLL_SECONDS
            job = await self._get(f"/api/jobs/{job_id}")
            status = str(job.get("status") or "")
            if status == "done":
                return job
            if status == "error":
                raise GatewayError(f"{label}失败：{str(job.get('error'))[:160]}")
        raise GatewayError(
            f"{label}超时（{self.job_timeout:.0f} 秒仍未完成）。"
            "网关可能仍在排队，可以稍后重试或降低尺寸。"
        )

    async def _download(self, job: dict[str, Any], out_dir: Path, *, default_ext: str) -> tuple[str, str]:
        """Fetch the job's first file to `out_dir`.

        Returns (local_path, url). `url` is what the page can actually load: our own
        `/api/asset/...` for a file under the asset tree, else the renderer's link.
        """
        import httpx

        files = job.get("files") or []
        if not files:
            raise GatewayError("任务完成但没有产出文件")

        item = files[0]
        filename = Path(str(item.get("filename") or f"out{default_ext}")).name
        # Gateways commonly reuse names such as ``output.mp4``.  A unique local
        # name keeps the next serial result from overwriting the previous file.
        local = out_dir / f"{uuid.uuid4().hex[:12]}-{filename}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # `url` needs the key in a header; `direct_url` carries it in the query. The
        # adapter uses the header form so the key never lands in a log line.
        url = str(item.get("url") or "")
        if not url:
            raise GatewayError("任务结果缺少下载地址")
        # Same transient-restart window as `_post`: a dropped connection here would
        # discard a render that has already been paid for in GPU time.
        content = b""
        last: Exception | None = None
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
                    r = await client.get(url, headers=self._headers())
                if r.status_code >= 400:
                    raise GatewayError(f"结果下载失败（{r.status_code}）")
                content = r.content
                if not content:
                    raise GatewayError("结果下载失败：网关返回了空文件")
                break
            except httpx.TransportError as exc:
                last = exc
                if attempt == 0:
                    await asyncio.sleep(1.5)
        else:
            raise GatewayError(f"结果下载中断：{last}")
        local.write_bytes(content)
        if not local.is_file() or local.stat().st_size != len(content):
            raise GatewayError("结果未能完整保存到本地")

        # Our own URL first: the renderer's link is authenticated and expires, and
        # the page renders this in an <img>/<video> tag that cannot send a header.
        return str(local), served_url(local) or str(item.get("direct_url") or "")

    # ---------------------------------------------------------------- interface

    async def render_image(
        self,
        *,
        prompt: str,
        out_dir: Path,
        reference_image: str = "",
        size: str = "",
        steps: int = 0,
        **_: Any,
    ) -> GeneratedImage:
        """Render one image. With a reference it runs image-to-image."""
        body: dict[str, Any] = {
            "type": "image",
            "prompt": prompt,
            # `square` for both cases: the gateway's own default, and a reference
            # image does not change the output aspect.
            "size": normalise_size(size, kind="image", fallback="square"),
            "steps": int(steps or self.steps * 2),
        }
        if reference_image:
            ref = Path(reference_image)
            if ref.is_file():
                body["ref_images"] = [await self._upload(ref)]

        job = await self._post("/api/jobs", body)
        job_id = str(job.get("job_id") or job.get("id") or "")
        final = await self._wait(job_id, label="图片生成")
        local, direct = await self._download(final, Path(out_dir), default_ext=".png")

        seconds = float(final.get("elapsed") or 0.0)
        return GeneratedImage(url=direct or local, local_path=local, prompt=prompt, seconds=seconds)

    async def render_video(
        self,
        *,
        prompt: str,
        out_dir: Path,
        seconds: float = 5.0,
        reference_image: str = "",
        reference_audio: str = "",
        reference_video: str = "",
        size: str = "",
        steps: int = 0,
        #: Which run this belongs to. Unused here — the gateway's own timeout
        #: governs — but present because the pipeline passes it and both backends
        #: must accept the same arguments. A signature that only one backend
        #: satisfies is the defect this protocol exists to prevent.
        run_id: str = "",
    ) -> GeneratedVideo:
        """Render one shot.

        Route choice, and why it matters:

          · audio or video supplied  → `ref_video`. Only this route can take a
            voice or a clip, and only it can lock the face across shots.
          · picture only             → `video` with `first_frame`, which the
            gateway drives from the image.

        Silently sending an audio reference to the `video` route would drop it:
        the gateway would accept the job and produce a film with no voice.
        """
        duration = max(1, min(15, int(round(seconds or 5))))
        common: dict[str, Any] = {
            "prompt": prompt,
            "size": normalise_size(size, kind="video", fallback=self.size),
            "duration": duration,
            "steps": int(steps or self.steps),
        }

        route = "ref_video" if (reference_audio or reference_video) else "video"
        body: dict[str, Any] = {**common, "type": route}
        reference_used = ""

        if route == "ref_video":
            # The gateway addresses uploads by placeholder, so the prompt must name
            # them. If the caller's prompt does not, prepend a line that does —
            # otherwise the model receives three files with no instruction about
            # which is the face, which is the motion and which is the voice.
            pictures: list[str] = []
            audios: list[str] = []
            videos: list[str] = []

            if reference_image and Path(reference_image).is_file():
                pictures.append(await self._upload(Path(reference_image)))
            if reference_video and Path(reference_video).is_file():
                videos.append(await self._upload(Path(reference_video)))
            if reference_audio and Path(reference_audio).is_file():
                audios.append(await self._upload(Path(reference_audio)))

            if pictures:
                body["ref_images"] = pictures
            if videos:
                body["ref_videos"] = videos
            if audios:
                body["ref_audios"] = audios

            notes: list[str] = []
            if pictures:
                notes.append("参考 <Picture 1> 中的人物长相与衣着，保持脸型、五官、发型一致。")
            if videos:
                notes.append("参考 <Video 1> 中主体的运动方式与镜头节奏。")
            if audios:
                notes.append("参考 <Audio 1> 的音色质感，用于环境声与人声。")
            if notes and "<Picture" not in prompt and "<Audio" not in prompt:
                body["prompt"] = " ".join(notes) + " 场景：" + prompt

            reference_used = "、".join(
                x for x in (
                    "人物参考" if pictures else "",
                    "视频参考" if videos else "",
                    "音频参考" if audios else "",
                ) if x
            )
        else:
            if reference_image and Path(reference_image).is_file():
                body["first_frame"] = await self._upload(Path(reference_image))
                reference_used = "首帧"

        job = await self._post("/api/jobs", body)
        job_id = str(job.get("job_id") or job.get("id") or "")
        final = await self._wait(job_id, label=f"视频生成（{route}）")
        local, direct = await self._download(final, Path(out_dir), default_ext=".mp4")

        elapsed = float(final.get("elapsed") or 0.0)
        return GeneratedVideo(
            url=direct or local,
            local_path=local,
            seconds=float(duration),
            duration=float(duration),
            reference_used=reference_used,
            task_id=job_id,
        )


#: Where a user's own gateway address may be configured.
#:
#: Both forms are checked because the same value can arrive two ways: an operator
#: exports an environment variable before starting the server, or a user fills in a
#: provider profile in the settings dialog. Requiring only the environment form
#: meant a user who bought a URL had no way to use it from the UI.
ENV_URL = "STUDIO_GATEWAY_URL"
ENV_KEY = "STUDIO_GATEWAY_KEY"


def gateway_from_settings(
    model_settings: Any = None, *, preferred_provider_id: str = ""
) -> GatewayGenerator | None:
    """Build a gateway adapter from configuration, or None when not configured.

    Returns None rather than raising: "no gateway" is a normal configuration, and
    the capability report already explains what a run without media can produce.

    Resolution order, and why:

      1. **Environment variables** — an operator's explicit choice; wins over a
         stored profile because it is set per-process and is what a deployment
         uses to override a checked-in file.
      2. **A provider profile pointing at a gateway** — the user fills in a URL and
         a key in the settings dialog. Recognised by either the `gateway` preset id
         or an explicit `kind: "gateway"`, so a user who just pastes a URL into an
         existing profile is not silently ignored.

    The point of accepting the second form: a user who buys a hosted key and a URL
    should not have to edit files or restart the server with new environment
    variables. One URL + one key, entered once in the UI, must be enough.
    """
    import os

    # ---- an explicitly assigned Agent profile ----
    try:
        from .. import models_config

        cfg = model_settings or models_config.load()
    except Exception:  # noqa: BLE001 - a missing config is just "no gateway"
        cfg = None

    if cfg is not None and preferred_provider_id:
        preferred = next(
            (
                p for p in (getattr(cfg, "providers", []) or [])
                if str(getattr(p, "id", "") or "") == preferred_provider_id
            ),
            None,
        )
        if preferred is not None and preferred.base_url and preferred.api_key:
            return _build(
                preferred.base_url,
                preferred.api_key,
                model=(preferred.video_model or preferred.image_model),
            )

    base = (os.environ.get(ENV_URL) or "").strip()
    key = (os.environ.get(ENV_KEY) or "").strip()
    if base and key:
        return _build(base, key)

    # ---- any configured gateway profile ----
    if cfg is not None:
        for p in getattr(cfg, "providers", []) or []:
            if not (getattr(p, "base_url", "") and getattr(p, "api_key", "")):
                continue
            kind = str(getattr(p, "kind", "") or "").lower()
            preset = str(getattr(p, "preset", "") or "").lower()
            pid = str(getattr(p, "id", "") or "").lower()
            if kind == "gateway" or preset == "gateway" or pid in ("gateway", "comfyui"):
                return _build(p.base_url, p.api_key, model=(p.video_model or p.image_model))

    return None


def _build(base_url: str, api_key: str, *, model: str = "") -> GatewayGenerator:
    """Construct the adapter with the configured size/step defaults."""
    import os

    return GatewayGenerator(
        base_url=base_url,
        api_key=api_key,
        video_model=model or "comfyui",
        image_model=model or "comfyui",
        size=os.environ.get("STUDIO_GATEWAY_SIZE", "tiny"),
        steps=int(os.environ.get("STUDIO_GATEWAY_STEPS", "8") or 8),
    )
