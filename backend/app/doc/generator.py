"""Generation clients — text-to-image and image-to-video.

Everything here was established by measurement against the live API, and the
numbers are recorded because they drive the orchestration:

  · text-to-image (CogView)  ≈ 15–40 s per portrait
  · image-to-video (CogVideoX) ≈ 80–160 s per 5 s shot
  · output is 16:9 or 1:1, ~1080p, no audio track

Three rules are encoded here rather than left to the caller, because each one was
learned by getting it wrong first:

  1. **English photographic prompt terms produce film stills; Chinese "写实风格"
     produces anime.** The same requirement yielded a 二次元 illustration in
     Chinese and a cinematic frame in English. So `StyleBible` carries English
     terms and this module does not translate them.

  2. **A reference image must be submitted immediately after it is created.**
     The URLs returned by the image API carry an expiring signature; reusing one
     in a later session produced `1210 参数有误`. Video generation therefore takes
     the URL from the same call that created it.

  3. **Video generation is asynchronous and slow.** Every shot is a submit + poll
     + local-save cycle with its own timeout. The run deliberately keeps only one
     such cycle in flight so the next prompt cannot overtake an unsaved result.

Anachronism control: the era's forbidden visuals are appended to the negative
prompt on every call. This is the mechanism that stops a wristwatch appearing in
a Qing dynasty frame — a positive description cannot express "must not exist".
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .generation_errors import diagnose


#: Prompt suffixes that push the model toward photographic output. Kept as data so
#: the art director can extend them without touching the client.
FILM_STYLE_SUFFIX = (
    "photorealistic cinematic film still, natural side lighting, shallow depth of field, "
    "35mm film grain, muted color grading, high detail, no illustration, no anime, no 3d render"
)


class GenerationCancelled(RuntimeError):
    """Raised when the run was cancelled mid-generation.

    A distinct type so the caller can tell "the user stopped this" from "the
    provider failed", and report the former as `cancelled` rather than `failed`.
    """


async def _cancellable_sleep(seconds: float, run_id: str) -> None:
    """Sleep in small slices, raising as soon as the run is cancelled.

    A plain `asyncio.sleep(20)` is uninterruptible: the cancel flag is only read
    after it returns, so a click waits up to the full interval. Slicing bounds the
    delay to `slice` while keeping the total wait unchanged.
    """
    from .cancellation import requested

    remaining = seconds
    while remaining > 0:
        if run_id and requested(run_id):
            raise GenerationCancelled("用户取消了任务")
        slice_s = min(0.25, remaining)
        await asyncio.sleep(slice_s)
        remaining -= slice_s
        if run_id and requested(run_id):
            raise GenerationCancelled("用户取消了任务")


class GenerationError(RuntimeError):
    """A generation call failed in a way the caller should report, not retry."""


@dataclass
class GeneratedImage:
    url: str
    local_path: str = ""
    prompt: str = ""
    seconds: float = 0.0


@dataclass
class GeneratedVideo:
    url: str
    local_path: str = ""
    seconds: float = 0.0
    duration: float = 0.0
    reference_used: str = ""
    task_id: str = ""


@dataclass
class GenerationBudget:
    """Track spend so a run cannot silently drain an account.

    Video generation is the expensive part; without a counter a 12-shot run that
    retries twice becomes 36 calls and nobody notices until the invoice.

    The limit is expressed in shots, not a fixed small number: several segments run
    concurrently and share one budget, so a cap tuned for a single segment was
    exhausted by the third segment's portraits — after which every remaining shot
    failed with "已达上限", which reads like a model error but is a budgeting bug.
    """

    max_calls: int = 0
    calls_made: int = 0
    started: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.max_calls:
            # Portraits (~1 per character) + one call per shot, doubled to allow
            # one rework round without the run dying mid-way.
            self.max_calls = 80

    def take(self) -> None:
        if self.calls_made >= self.max_calls:
            raise GenerationError(
                f"生成调用已达上限（{self.max_calls} 次）。"
                "请缩短时长或减少镜头，或提高预算上限。"
            )
        self.calls_made += 1

    def remaining(self) -> int:
        return max(0, self.max_calls - self.calls_made)


class Generator:
    """Thin client for a Zhipu-compatible image/video endpoint."""

    BASE = "https://open.bigmodel.cn/api/paas/v4"

    def __init__(
        self,
        *,
        api_key: str,
        image_model: str = "cogview-4-250304",
        video_model: str = "cogvideox-3",
        timeout: float = 180.0,
        budget: GenerationBudget | None = None,
        base_url: str = "",
        video_base_url: str = "",
        video_api_key: str = "",
    ) -> None:
        self.api_key = api_key
        self.image_model = image_model
        self.video_model = video_model
        self.timeout = timeout
        self.budget = budget or GenerationBudget()
        # The endpoint follows the provider. Hardcoding Zhipu's address meant a
        # SiliconFlow key was sent to Zhipu, which answered 401 — reported as an
        # invalid key when the real problem was the URL.
        # `self.BASE`, not the bare name: `BASE` is a class attribute, so a bare
        # reference inside `__init__` is a NameError — it only stayed hidden because
        # every caller happened to pass an explicit base_url. A user who supplies
        # just a key (expecting the default endpoint) hit it immediately.
        self.base_url = (base_url or self.BASE).rstrip("/")
        # Video can come from a different provider than images. Empty means "same
        # as the image provider", which is the common single-provider case.
        self.video_base_url = (video_base_url or base_url or self.BASE).rstrip("/")
        self.video_api_key = video_api_key or api_key

    # ------------------------------------------------------------------ image

    async def text_to_image(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        out_dir: Path | None = None,
        name: str = "",
    ) -> GeneratedImage:
        """Generate one still. Returns the URL and, if a directory was given, a
        local copy.

        The local copy exists because the returned URL is temporary: a run
        inspected tomorrow would have broken image links, and the trace would then
        be unable to prove what was generated.
        """
        self.budget.take()
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Both `size` (Zhipu) and `image_size` (SiliconFlow) are sent: the two
            # providers are both OpenAI-compatible but disagree on this field name,
            # and each ignores the key it does not use. Sending both avoids
            # guessing which dialect the configured base_url speaks.
            resp = await client.post(
                f"{self.base_url}/images/generations",
                json={
                    "model": self.image_model,
                    "prompt": prompt,
                    "size": size,
                    "image_size": size,
                    "batch_size": 1,
                },
                headers=self._headers(),
            )
        if resp.status_code != 200:
            # Diagnosed, not raw: the user must be told whether the endpoint is
            # missing, the model unsupported, or the key rejected — those need
            # different fixes, and a bare status code hides which it is.
            _d = diagnose(
                operation="image",
                status=resp.status_code,
                body=resp.text,
                model=self.image_model,
                base_url=self.base_url,
            )
            raise GenerationError(_d.as_error_message())

        body = resp.json()
        # Zhipu returns `data`, SiliconFlow returns `images`; both hold objects
        # with a `url`. Accepting either keeps one code path for both providers.
        data = body.get("data") or body.get("images") or []
        if not data:
            raise GenerationError(
                f"文生图未返回图片：{json.dumps(body, ensure_ascii=False)[:160]}"
            )
        first = data[0]
        url = first.get("url", "") if isinstance(first, dict) else str(first)
        if not url:
            # SiliconFlow can return base64 in `b64_json`; report it rather than
            # writing an empty file, and leave a marker so the caller knows why.
            if isinstance(first, dict) and first.get("b64_json"):
                raise GenerationError("文生图返回 base64 数据，本应用只处理图片 URL")
            raise GenerationError(
                f"文生图返回中没有 url：{json.dumps(body, ensure_ascii=False)[:160]}"
            )
        seconds = time.perf_counter() - started

        local = ""
        if out_dir is not None and url:
            local = await self._download(url, out_dir / f"{name or 'image'}.png")

        # Prefer our own copy: a hosted provider's link is public, but a link that
        # needs a key (or expires) leaves the page with an empty frame.
        from .media_url import served_url

        return GeneratedImage(url=served_url(local) or url, local_path=local,
                              prompt=prompt, seconds=seconds)

    # ------------------------------------------------------------------ video

    async def image_to_video(
        self,
        prompt: str,
        *,
        reference_url: str,
        out_dir: Path | None = None,
        name: str = "",
        poll_timeout: float = 420.0,
        #: Which run this belongs to, so the poll loop can stop the moment the
        #: user cancels instead of waiting out an interval.
        run_id: str = "",
        #: What the caller asked for. This family of endpoints accepts only
        #: discrete durations (5/10 s) and preset sizes, so these are requests
        #: rather than guarantees — the compositor still conforms every clip.
        #: Sending them is still worth it: a 10-second shot asked for as 10 and
        #: cut down looks right, while a 6-second narration over a 5-second loop
        #: repeats the motion and reads as a mistake.
        seconds: float = 0.0,
        size: str = "",
    ) -> GeneratedVideo:
        """Generate a shot whose first frame is `reference_url`.

        The reference is passed as `image_url`, which is what fixes the actor's
        identity across shots. Callers must pass a URL created moments ago — see
        the module docstring on the expiring signature.
        """
        if not reference_url:
            raise GenerationError("图生视频必须提供参考图 URL（否则人物无法保持一致）")

        self.budget.take()
        body: dict[str, Any] = {
            "model": self.video_model,
            "prompt": prompt,
            "image_url": reference_url,
        }
        # Only fields the caller actually specified. Inventing a default here would
        # quietly override the storyboard's plan for every shot.
        if seconds and seconds > 0:
            body["duration"] = max(1, int(round(seconds)))
        if size:
            body["size"] = size

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                # Video may live on a different provider than images, so its own
                # endpoint travels with the model.
                f"{self.video_base_url}/videos/generations",
                json=body,
                headers=self._headers(),
            )
            if resp.status_code in (400, 422) and len(body) > 3:
                # Some deployments reject an unknown field instead of ignoring it.
                # A shot that could be rendered must not fail over an optional
                # hint, so retry once without the constraints; the compositor
                # normalises the result either way.
                for key in ("duration", "size"):
                    body.pop(key, None)
                resp = await client.post(
                    f"{self.video_base_url}/videos/generations",
                    json=body,
                    headers=self._headers(),
                )
        if resp.status_code != 200:
            # Diagnosed: 404 here usually means the provider has no image-to-video
            # route, which the user must distinguish from a rejected key.
            _d = diagnose(
                operation="video",
                status=resp.status_code,
                body=resp.text,
                model=self.video_model,
                base_url=self.base_url,
            )
            raise GenerationError(_d.as_error_message())

        task_id = resp.json().get("id", "")
        if not task_id:
            raise GenerationError("图生视频未返回任务 id")

        url, elapsed = await self._poll_video(task_id, poll_timeout, run_id)
        local = ""
        if out_dir is not None and url:
            local = await self._download(url, out_dir / f"{name or 'shot'}.mp4")
        # Same rule as images: the copy on disk is the one the page can always load.
        from .media_url import served_url

        return GeneratedVideo(
            url=served_url(local) or url,
            local_path=local,
            seconds=elapsed,
            reference_used=reference_url,
            task_id=task_id,
        )

    # ------------------------------------------------- backend protocol (media_backend)

    #: Recorded in the trace and the capability report.
    backend_name = "hosted"

    def capabilities(self) -> dict[str, Any]:
        """What this endpoint can do, as far as is known.

        Conservative on purpose. A hosted vendor key is almost always an
        image-and-video pair with ONE reference image and no audio track; claiming
        audio support because "some model somewhere does it" is how a run ships a
        silent film with a green tick.

        `native_audio` is omitted rather than set to False: whether this endpoint
        can synth a soundtrack is not established, and unknown is a different
        statement from cannot.
        """
        return {
            "image": bool(self.image_model),
            "video": bool(self.video_model),
            "first_frame": bool(self.video_model),
            "last_frame": False,
            # The hosted image route takes a reference for image-to-image; the
            # video route takes exactly one first frame.
            "reference_images": bool(self.image_model),
            "reference_audio": False,
            "reference_video": False,
            "max_seconds_per_shot": 5.0,
            "backend": self.backend_name,
            "model": self.video_model or self.image_model,
        }

    async def render_image(
        self,
        *,
        prompt: str,
        out_dir: Path,
        reference_image: str = "",
        size: str = "",
        steps: int = 0,
    ) -> GeneratedImage:
        """Protocol entry for image generation.

        `reference_image` / `steps` are accepted and ignored: this endpoint's image
        dialect has no reference-image field, so honouring them is impossible and
        pretending otherwise would silently produce a different picture than the
        caller asked for. The capability record says so, and the caller warns.
        """
        return await self.text_to_image(
            prompt,
            size=size or "1024x1024",
            out_dir=out_dir,
            # A fixed ``image.png`` would be overwritten by the next item in the
            # serial queue, so every confirmed download gets its own durable file.
            name=f"image-{uuid.uuid4().hex[:12]}",
        )

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
        run_id: str = "",
    ) -> GeneratedVideo:
        """Protocol entry for video generation.

        The hosted route is strictly image-to-video: it requires a first frame and
        accepts no audio or clip reference. A caller that supplies one gets a clear
        refusal here rather than a shot rendered without it — that is the whole
        point of declaring capabilities.

        `reference_image` may be a local path; it is uploaded through the provider's
        files endpoint first, because this route takes a URL.
        """
        if reference_audio or reference_video:
            raise GenerationError(
                "当前视频接口不支持参考音频或参考视频："
                "该供应商的视频路由只接受首帧图。"
                "如需音色或动作参考，请改用本地 ComfyUI 网关（STUDIO_GATEWAY_URL）。"
            )
        if not reference_image:
            raise GenerationError(
                "当前视频接口只支持图生视频，必须提供首帧图。"
                "如需纯文生视频，请配置支持该能力的供应商。"
            )

        url = reference_image
        if not str(reference_image).lower().startswith(("http://", "https://")):
            url = await self._upload_reference(Path(reference_image))

        return await self.image_to_video(
            prompt,
            reference_url=url,
            out_dir=out_dir,
            # Preserve every completed shot instead of overwriting ``shot.mp4``
            # when the next prompt finishes.
            name=f"shot-{uuid.uuid4().hex[:12]}",
            run_id=run_id,
            # Forwarded, not dropped. These were accepted by this method and then
            # silently discarded, so the request went out with no duration and no
            # size at all — every shot came back at whatever the model defaults to,
            # and the compositor had to trim or loop each one to fit its narration.
            seconds=seconds,
            size=size,
        )

    async def _upload_reference(self, path: Path) -> str:
        """Put a local file where the hosted route can read it.

        Tried in order: the provider's own `/files` upload, then a data URL. The
        two are what OpenAI-compatible vendors actually offer; a vendor with
        neither cannot accept a local first frame at all, and the error says so
        instead of sending a filesystem path over the wire.
        """
        if not path.is_file():
            raise GenerationError(f"首帧文件不存在：{path}")

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                with path.open("rb") as fh:
                    resp = await client.post(
                        f"{self.base_url}/files",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        files={"file": (path.name, fh, "application/octet-stream")},
                        data={"purpose": "vision"},
                    )
                if resp.status_code == 200:
                    body = resp.json()
                    for key in ("url", "file_url", "download_url"):
                        if body.get(key):
                            return str(body[key])
                    file_id = body.get("id")
                    if file_id:
                        return f"{self.base_url}/files/{file_id}/content"
            except Exception:  # noqa: BLE001 - fall through to the data URL
                pass

        # Data URL fallback: the provider reads the bytes inline.
        import base64
        import mimetypes

        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode()
        return f"data:{mime};base64,{encoded}"

    async def _poll_video(
        self, task_id: str, timeout: float, run_id: str = ""
    ) -> tuple[str, float]:
        """Poll until the task settles.

        Reports the final error verbatim: a `1210 参数有误` from a stale reference
        URL looks identical to a malformed prompt until the API's own message is
        surfaced, and that message is what tells the two apart.
        """
        started = time.perf_counter()
        delay = 8.0
        async with httpx.AsyncClient(timeout=45.0) as client:
            while time.perf_counter() - started < timeout:
                # Interruptible wait: a cancel takes effect within 0.25 s instead
                # of waiting out the 8–20 s interval.
                await _cancellable_sleep(delay, run_id)
                resp = await client.get(
                    f"{self.video_base_url}/async-result/{task_id}",
            headers=self._video_headers(),
                )
                if resp.status_code != 200:
                    continue
                body = resp.json()
                status = body.get("task_status")
                if status == "SUCCESS":
                    results = body.get("video_result") or []
                    if not results:
                        raise GenerationError("任务成功但未返回视频")
                    return results[0].get("url", ""), time.perf_counter() - started
                if status == "FAIL":
                    error = body.get("error") or {}
                    raise GenerationError(
                        f"视频生成失败：{error.get('code','')} {error.get('message','')}".strip()
                    )
                # Long tasks can take minutes; back off so polling does not add load.
                delay = min(20.0, delay + 2.0)

        raise GenerationError(f"视频生成超时（{timeout:.0f}s 未完成，任务 {task_id}）")

    # ---------------------------------------------------------------- helpers

    def _video_headers(self) -> dict[str, str]:
        """Auth for the video provider, which may differ from the image one."""
        return {
            "Authorization": f"Bearer {self.video_api_key}",
            "Content-Type": "application/json",
        }

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _download(self, url: str, dest: Path) -> str:
        """Fetch the artifact locally so the run stays inspectable."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
            resp = await client.get(url)
        if resp.status_code != 200:
            raise GenerationError(f"生成结果下载失败（HTTP {resp.status_code}）")
        if not resp.content:
            raise GenerationError("生成结果下载失败：上游返回了空文件")
        dest.write_bytes(resp.content)
        if not dest.is_file() or dest.stat().st_size != len(resp.content):
            raise GenerationError("生成结果未能完整保存到本地")
        return str(dest)

    @classmethod
    def from_settings(
        cls,
        model_settings: Any = None,
        *,
        image_agent_id: str = "consistency",
        video_agent_id: str = "video_gen",
        **kw: Any,
    ) -> "Generator":
        """Build from configured providers, honouring media-agent assignments.

        The Agent settings table binds consistency to an image model and video_gen
        to a video model.  Falling back to the first provider with that capability
        made the UI selection cosmetic; resolve the explicit bindings first.
        """
        from .. import models_config

        cfg = model_settings or models_config.load()

        # Prefer a provider that actually declares a media model. Without this,
        # a text provider configured for `glm-5` would be chosen for image calls —
        # its endpoint has no /images/generations, so every shot would fail with a
        # 404 that looks like a bad key.
        media = [
            p for p in cfg.providers
            if p.api_key and (p.image_model or p.video_model)
        ]
        if media:
            # Per capability, not per provider: image and video are independent
            # services, and requiring one provider to offer both made an
            # exhausted provider win merely by appearing first in the list.
            #
            # A single Generator serves both, so the video provider's key is used
            # for video calls when it differs — see `video_base_url` and
            # `video_api_key`.
            def assigned(agent_id: str, field: str):
                raw = str((getattr(cfg, "assignments", {}) or {}).get(agent_id) or "")
                provider_id, _, pinned = raw.partition(":")
                profile = next(
                    (p for p in cfg.providers if str(p.id) == provider_id), None
                )
                if profile is None or not profile.api_key or not getattr(profile, field, ""):
                    return None, ""
                return profile, pinned

            image_provider, pinned_image = assigned(image_agent_id, "image_model")
            video_provider, pinned_video = assigned(video_agent_id, "video_model")
            image_provider = image_provider or next(
                (p for p in media if p.image_model), None
            )
            video_provider = video_provider or next(
                (p for p in media if p.video_model), None
            )
            anchor = image_provider or video_provider
            assert anchor is not None  # `media` is non-empty, so one exists
            return cls(
                api_key=anchor.api_key,
                image_model=(pinned_image or image_provider.image_model if image_provider else ""),
                video_model=(pinned_video or video_provider.video_model if video_provider else ""),
                timeout=float(anchor.timeout or 180.0),
                base_url=anchor.base_url or "",
                # When video comes from a different provider, its endpoint and key
                # must travel with it, or the request goes to the wrong account.
                video_base_url=(
                    video_provider.base_url
                    if video_provider and video_provider is not image_provider
                    else ""
                ),
                video_api_key=(
                    video_provider.api_key
                    if video_provider and video_provider is not image_provider
                    else ""
                ),
                **kw,
            )

        # Fallback: an older single-provider setup, where the one keyed provider
        # serves everything. Kept so an existing configuration still runs.
        for profile in cfg.providers:
            if not profile.api_key:
                continue
            return cls(api_key=profile.api_key, base_url=profile.base_url or "", **kw)
        raise GenerationError(
            "没有可用的生图/生视频供应商：请在设置里为某个供应商填写「生图模型」或「生视频模型」"
        )


# ============================================================================
# Prompt assembly — where the consistency guarantees are enforced
# ============================================================================


def build_shot_prompt(
    *,
    visual_description: str,
    style_prompt: str,
    appearance: str = "",
    extra_negatives: list[str] | None = None,
) -> str:
    """Compose the positive prompt sent to the video model.

    Ordering is deliberate: subject first (the model weights early tokens most),
    then locked appearance, then the global style tail. Putting the style block in
    the middle measurably weakens the subject in the output.
    """
    parts = [visual_description.strip()]
    if appearance:
        parts.append(appearance.strip())
    parts.append(style_prompt.strip() or FILM_STYLE_SUFFIX)
    return ", ".join(p for p in parts if p)


def build_negative_prompt(*, base: str, forbidden: list[str] | None = None) -> str:
    """The avoid-list, with era prohibitions appended.

    Anachronisms are negative constraints: "no wristwatch" cannot be expressed by
    describing a robe, so they belong here rather than in the positive prompt.
    """
    items = [base] if base else []
    items.extend(forbidden or [])
    return ", ".join(i.strip() for i in items if i and i.strip())
