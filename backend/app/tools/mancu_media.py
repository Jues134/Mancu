"""Client for the optional media production backend.

Image/video generation and composition are delegated to its HTTP API.

The client adds two things a raw call lacks:
  * availability detection, so the studio degrades to "prompts only" mode with a
    clear message instead of failing confusingly when the service is not running;
  * bounded polling for async compose, with a real timeout.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from ..config import MancuMediaConfig


class MancuMediaError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, detail: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


class MancuMediaClient:
    def __init__(self, config: MancuMediaConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ health

    async def health(self) -> dict[str, Any]:
        """Probe the instance; returns {"ok": bool, "detail": str}."""
        url = self.config.base_url.rstrip("/") + "/api/health"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
            db_status = (data.get("db") or {}).get("status")
            ok = db_status == "ok"
            return {
                "ok": ok,
                "detail": "漫簇媒体后端可用" if ok else f"漫簇媒体后端数据库异常：{db_status}",
                "base_url": self.config.base_url,
                "ffmpeg": (data.get("ffmpeg") or {}).get("exists"),
            }
        except Exception as exc:  # noqa: BLE001 - probe never raises
            return {
                "ok": False,
                # Deliberately generic: the media backend is an optional external
                # service, not a folder in this repository. Pointing at a path here
                # would imply a dependency the studio does not have.
                "detail": (
                    f"未连接媒体后端（{self.config.base_url}）。"
                    "这是可选的外部服务：没有它仍可生成事实卡、视觉方案、文案、分镜与提示词；"
                    "如需出图与成片，请部署媒体后端并把 MANCU_MEDIA_BASE_URL 指向它。"
                ),
                "base_url": self.config.base_url,
                "error": str(exc),
            }

    # ------------------------------------------------------------- generation

    async def generate_image(
        self,
        *,
        prompt: str,
        negative_prompt: str = "",
        aspect_ratio: str = "3:4",
        reference_image_path: str | None = None,
    ) -> dict[str, Any]:
        """Generate one image through the media backend's provider abstraction."""
        cfg = self.config
        if not cfg.image_api_key:
            raise MancuMediaError(
                "尚未配置生图 Key：请设置环境变量 STUDIO_IMAGE_API_KEY"
                "（以及可选的 STUDIO_IMAGE_PROVIDER / STUDIO_IMAGE_MODEL）"
            )
        if not cfg.image_model:
            raise MancuMediaError("尚未配置生图模型：请设置 STUDIO_IMAGE_MODEL")

        size = _size_for_ratio(aspect_ratio)
        payload: dict[str, Any] = {
            "provider": cfg.image_provider,
            "model": cfg.image_model,
            "apiKey": cfg.image_api_key,
            "prompt": prompt,
            "mode": "image-to-image" if reference_image_path else "text-to-image",
            "options": {
                "width": size[0],
                "height": size[1],
                **( {"negativePrompt": negative_prompt} if negative_prompt else {} ),
            },
        }
        if cfg.image_base_url:
            payload["baseUrl"] = cfg.image_base_url
        if reference_image_path:
            payload["imageUrl"] = reference_image_path

        url = cfg.base_url.rstrip("/") + "/api/ai/image"
        try:
            async with httpx.AsyncClient(timeout=cfg.timeout_seconds) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            body = ""
            try:
                body = exc.response.text[:300]
            except Exception:  # noqa: BLE001
                pass
            raise MancuMediaError(f"生图失败：HTTP {exc.response.status_code}", status=exc.response.status_code, detail=body) from exc
        except Exception as exc:  # noqa: BLE001
            raise MancuMediaError(f"生图请求失败：{exc}", detail=str(exc)) from exc

        urls = data.get("imageUrls") or data.get("urls") or []
        if not urls:
            raise MancuMediaError("生图接口未返回图片地址", detail=str(data)[:300])
        return {"urls": urls, "model": data.get("modelId") or cfg.image_model, "raw": data}

    # ---------------------------------------------------------------- compose

    async def compose(self, project_id: str, *, poll_interval: float = 3.0, timeout: float = 900.0) -> dict[str, Any]:
        """Trigger a compose and poll to a terminal state."""
        base = self.config.base_url.rstrip("/")
        url = f"{base}/api/project/{project_id}/compose"
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                resp = await client.post(url, json={"freeTts": {"enabled": True}})
                resp.raise_for_status()
                started = resp.json()
                composition_id = started.get("compositionId")
                if not composition_id:
                    raise MancuMediaError("合成接口未返回 compositionId", detail=str(started)[:300])

                waited = 0.0
                while waited < timeout:
                    await asyncio.sleep(poll_interval)
                    waited += poll_interval
                    q = await client.get(f"{base}/api/project/{project_id}/compose?compositionId={composition_id}")
                    q.raise_for_status()
                    comp = (q.json() or {}).get("composition") or {}
                    status = comp.get("status")
                    if status == "done":
                        return comp
                    if status == "failed":
                        raise MancuMediaError("视频合成失败", detail=str(comp)[:300])
                raise MancuMediaError(f"合成超时（超过 {int(timeout)} 秒）")
        except MancuMediaError:
            raise
        except httpx.HTTPStatusError as exc:
            raise MancuMediaError(f"合成失败：HTTP {exc.response.status_code}", status=exc.response.status_code) from exc
        except Exception as exc:  # noqa: BLE001
            raise MancuMediaError(f"合成请求失败：{exc}", detail=str(exc)) from exc


def _size_for_ratio(ratio: str) -> tuple[int, int]:
    """Map an aspect ratio to a provider-safe pixel size."""
    mapping = {
        "1:1": (1024, 1024),
        "3:4": (896, 1152),
        "4:3": (1152, 896),
        "9:16": (768, 1344),
        "16:9": (1344, 768),
    }
    return mapping.get(ratio, (896, 1152))
