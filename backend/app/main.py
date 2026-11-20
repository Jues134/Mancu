"""FastAPI application: the studio's HTTP surface.

Endpoints are deliberately few. The whole interaction is:
  upload → create run → poll run (steps + artifacts stream in) → intervene.

Runs execute as asyncio background tasks in the same process. That is a
deliberate constraint: SQLite allows one writer and the composing backend is a
separate service, so a distributed queue would add moving parts without adding
throughput. The state machine lives in the database, so a process restart leaves
runs that can be inspected and re-driven rather than lost.

Uploads accept base64 JSON rather than multipart, which keeps the dependency
surface and browser client small.
"""

from __future__ import annotations

import base64
import binascii
import os
import socket
import threading
import time
import webbrowser
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, logs
from .api.diagnostics import router as diagnostics_router
from .api.doc import router as doc_router
from .api.knowledge import router as knowledge_router
from .api.mcp import router as mcp_router
from .api.relay import router as relay_router
from .api.settings import router as settings_router
from .config import CONFIG_DIR, DATA_DIR, UPLOAD_DIR, ensure_dirs, settings
from .tools.mancu_media import MancuMediaClient
from .tools.llm import LLMError


# Before anything else: a crash in the next twenty lines is exactly what someone
# opens 设置 → 日志 to look at, so the file has to exist first.
logs.setup()
logger = logs.get_logger("mancu.app")

ensure_dirs()
app = FastAPI(title="漫簇", version="0.1.0")
app.include_router(settings_router)
app.include_router(relay_router)
app.include_router(doc_router)
app.include_router(knowledge_router)
app.include_router(mcp_router)
app.include_router(diagnostics_router)


def _open_browser_when_serving() -> None:
    """把界面在默认浏览器里打开，等端口真的能连上再开。

    这一步放在服务端而不是 start.ps1：启动脚本里把浏览器“延迟打开”要起一个脱离
    的子进程（`start /b powershell ...`），那串引号在部分 shell 下会解析失败，
    失败时只是静默不开——用户看不出是启动脚本坏了还是网页打不开。服务自己
    知道端口几时开始接受连接，轮询自己的端口再交给系统打开，时序永远是对的。

    只在 start.ps1 通过 `MANCU_OPEN_BROWSER=1` 显式要求时执行：用 uvicorn 直接
    起服务、跑测试、或者容器里部署时，谁都不希望后台突然弹出浏览器。
    """

    def _port() -> int:
        try:
            return int(os.environ.get("STUDIO_PORT") or 8100)
        except ValueError:
            return 8100

    port = _port()
    url = f"http://127.0.0.1:{port}"

    def _open() -> None:
        # startup 钩子跑在开始服务之前，此刻端口往往还没监听，所以先探活。
        # 60 秒后放弃：连不上就是服务没起来，这时打开浏览器只会显示错误页。
        for _ in range(120):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    webbrowser.open(url)
                    return
            except OSError:
                time.sleep(0.5)

    threading.Thread(target=_open, name="open-browser", daemon=True).start()


def _startup_report() -> None:
    """One block per boot: what this process actually is.

    This exists because the first question in every "it doesn't work" thread is
    "what is your setup", and answering it used to mean asking the user to go find
    out. Six lines at startup turn that into a screenshot.

    Every part is guarded: a report that fails must not stop the server, and one
    broken source must not silence the other lines.
    """
    import platform
    import sys

    logger.info(
        "漫簇启动 v%s · Python %s · %s · 端口 %s",
        app.version,
        sys.version.split()[0],
        platform.machine(),
        os.environ.get("STUDIO_PORT") or "8100",
    )
    logger.info("数据目录 %s", DATA_DIR)
    logger.info("配置目录 %s", CONFIG_DIR)
    logger.info("日志 %s（级别 %s）", logs.LOG_DIR, logs.level_name())

    try:
        from . import models_config

        cfg = models_config.load()
        models = sorted({p.model for p in cfg.providers if getattr(p, "model", "")})
        logger.info("供应商 %d 个 · 模型：%s", len(cfg.providers), "、".join(models) or "（未配置）")
    except Exception as exc:  # noqa: BLE001 - a broken config is reported, not fatal
        logger.warning("读取供应商配置失败：%s", exc)

    try:
        from .kb import vector

        health = vector.health()
        logger.info(
            "语义检索 · %s · %s 段 · %s",
            "可用" if health.get("ready") else "不可用",
            health.get("chunks", 0),
            health.get("reason", ""),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("语义检索状态读取失败：%s", exc)

    try:
        removed = logs.prune()
        if removed:
            logger.info("清理过期日志 %d 个文件", removed)
    except Exception:  # noqa: BLE001 - housekeeping is never worth a failed boot
        pass


@app.on_event("startup")
async def _close_interrupted_runs() -> None:
    """Mark runs left non-terminal by a previous process as interrupted.

    This process owns a fresh, empty live-run set, so any row still claiming to be
    `queued`/`running` belongs to a process that no longer exists. Closing those out
    at startup is what keeps a stale flag from turning into a permanently stuck
    history entry — deleting them would otherwise be refused forever.

    Best effort: a failure here must not stop the server from serving.
    """
    _startup_report()
    try:
        closed = db.reconcile_interrupted_runs()
        if closed:
            # The run was killed with the previous process. Saying so is what turns
            # "这次任务怎么没跑完" into a fact instead of a mystery.
            logger.warning("上一个进程留下的未完成任务已标记为中断：%s 个", len(closed))
    except Exception as exc:  # noqa: BLE001 - startup must not fail over bookkeeping
        logger.warning("中断任务对账失败：%s", exc)
    if os.environ.get("MANCU_OPEN_BROWSER", "").strip().lower() in {"1", "true", "yes"}:
        try:
            _open_browser_when_serving()
        except Exception:  # noqa: BLE001 - never block serving over a browser tab
            pass


# Local development: the Vite dev server runs on another port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#: Paths that must not write a line every time they are hit. The run panel polls
#: its status once a second while a task runs and the UI refetches health on load;
#: at INFO those would push every real event out of a 5 MB file within a day.
#: Errors and slow responses on these paths are still recorded — that is the part
#: worth knowing about.
_QUIET_PREFIXES = ("/api/doc/runs/", "/api/health", "/bundle.", "/assets/")

#: Above this, the line is worth reading even when it succeeded. A page that
#: "just felt slow" has no error to search for; it has only a duration.
SLOW_MS = 2000


def _is_quiet(method: str, path: str) -> bool:
    return method == "GET" and path.startswith(_QUIET_PREFIXES)


@app.middleware("http")
async def _record_request(request: Request, call_next):
    """Time every request and put the two facts on disk that a failure needs.

    The id is the point of this function. Without it, the traceback logged five
    frames deep and the HTTP 500 the browser saw are two unrelated lines in a
    file; with it, `req=8f2a` appears on both.
    """
    request_id = db.new_id()[:8]
    started = time.perf_counter()
    quiet = _is_quiet(request.method, request.url.path)
    with logs.request_context(request_id):
        try:
            response = await call_next(request)
        except Exception:
            # Starlette turns this into a 500 further out; log it here so the
            # line carries the request id that the caller will be told about.
            logger.exception("请求处理失败：%s %s", request.method, request.url.path)
            logs.event(
                "http.failed", level="error",
                method=request.method, path=request.url.path,
                ms=int((time.perf_counter() - started) * 1000),
            )
            raise

        elapsed = int((time.perf_counter() - started) * 1000)
        status = response.status_code
        response.headers["X-Request-Id"] = request_id

        if status >= 500:
            logger.error("%s %s → %s（%s ms）", request.method, request.url.path, status, elapsed)
        elif status >= 400:
            logger.warning("%s %s → %s（%s ms）", request.method, request.url.path, status, elapsed)
        elif elapsed >= SLOW_MS:
            logger.warning("慢请求 %s %s → %s（%s ms）", request.method, request.url.path, status, elapsed)
        elif not quiet:
            logger.info("%s %s → %s（%s ms）", request.method, request.url.path, status, elapsed)

        if status >= 400 or elapsed >= SLOW_MS:
            logs.event(
                "http.error" if status >= 400 else "http.slow",
                level="warn",
                method=request.method,
                path=request.url.path,
                status=status,
                ms=elapsed,
            )
        return response


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Last line of defence: anything that escaped a route handler.

    The body names where to look instead of returning a bare 500. A user who sees
    "内部错误" and nothing else has no next step; one who is told the stack is in
    设置 → 日志, under request id `8f2a`, can open it and find the line.
    """
    logger.exception("未处理的异常：%s %s", request.method, request.url.path)
    logs.event(
        "http.unhandled", level="error",
        method=request.method, path=request.url.path, error=f"{type(exc).__name__}: {exc}",
    )
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "detail": "服务内部错误。完整堆栈见「设置 → 日志 → 错误日志」",
            "request_id": logs.current_request_id(),
        },
    )

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
EXT_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}


class UploadFilePayload(BaseModel):
    filename: str = "image.jpg"
    mime_type: str = "image/jpeg"
    data_base64: str
    #: Which character this file belongs to. Required for audio: a voice sample
    #: with no owner cannot be used, because nothing downstream can tell whose
    #: voice it is. Ignored for images and video.
    character_id: str = ""
    character_name: str = ""
    #: What the sample is meant to convey, e.g. "只参考音色，不参考语速".
    reference_role: str = ""


class UploadRequest(BaseModel):
    files: list[UploadFilePayload]


# ============================================================================
# Health / capability
# ============================================================================


@app.get("/api/health")
async def health() -> dict:
    """Report what the studio can actually do right now.

    The UI uses this to explain missing configuration up front, instead of
    letting a run fail halfway with a cryptic error.
    """
    mancu_media = await MancuMediaClient(settings.mancu_media).health()
    return {
        "ok": True,
        "llm": {
            "configured": settings.llm.configured,
            "base_url": settings.llm.base_url,
            "model": settings.llm.model,
        },
        "mancu_media": mancu_media,
        "image_gen": {
            "configured": bool(settings.mancu_media.image_api_key and settings.mancu_media.image_model),
            "provider": settings.mancu_media.image_provider,
            "model": settings.mancu_media.image_model or "(未设置)",
        },
        "budget": {"max_usd_per_run": settings.budget.max_usd_per_run},
    }


# ============================================================================
# Upload
# ============================================================================


@app.post("/api/upload")
async def upload(payload: UploadRequest) -> dict:
    """Accept base64-encoded images, audio and video; persist under data/uploads.

    An audio upload must say which character it belongs to. That is enforced here
    rather than left to the caller: a voice sample with no owner is unusable
    downstream, and silently accepting it would make the user believe they
    supplied a voice that never gets applied.
    """
    from .doc.uploads import UploadRejected, infer_kind

    accepted: list[dict] = []
    for item in payload.files:
        try:
            kind = infer_kind(item.filename, item.mime_type)
        except UploadRejected as exc:
            raise HTTPException(400, str(exc)) from exc

        if kind.name == "audio" and not (item.character_id or item.character_name):
            raise HTTPException(
                400,
                f"音频「{item.filename}」必须指定属于哪个角色"
                "（character_name 或 character_id）。没有归属的声音无法用于配音，"
                "未上传声音的角色会由视频模型生成声音。",
            )

        # Tolerate a data-URL prefix ("data:image/png;base64,....").
        raw = item.data_base64.split(",", 1)[-1] if item.data_base64.startswith("data:") else item.data_base64
        try:
            content = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(400, f"文件数据不是合法的 base64：{item.filename}") from exc

        if len(content) > kind.max_bytes:
            raise HTTPException(
                400,
                f"文件过大：{item.filename}（{kind.name} 上限 "
                f"{round(kind.max_bytes / (1024 * 1024))}MB）",
            )
        if not content:
            raise HTTPException(400, f"文件内容为空：{item.filename}")

        suffix = Path(item.filename or "").suffix.lower()
        target = UPLOAD_DIR / f"{db.new_id()}{suffix}"
        target.write_bytes(content)
        accepted.append(
            {
                "path": str(target),
                "kind": kind.name,
                "filename": item.filename,
                # Only meaningful for audio; echoed so the caller can bind the
                # stored path to a character without re-sending the file.
                "character_id": item.character_id,
                "character_name": item.character_name,
                "reference_role": item.reference_role,
            }
        )

    return {
        "paths": [a["path"] for a in accepted],
        "files": accepted,
    }


@app.get("/api/upload/{name}")
async def get_upload(name: str) -> FileResponse:
    # Path traversal guard: only allow a bare file name.
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "非法文件名")
    path = UPLOAD_DIR / name
    if not path.is_file():
        raise HTTPException(404, "文件不存在")
    return FileResponse(path)


# ============================================================================
# Misc
# ============================================================================


@app.get("/api/llm/test")
async def llm_test() -> dict:
    """One tiny completion to prove the key/model/base URL actually work."""
    from .tools.llm import chat

    try:
        result = await chat(
            settings.llm,
            system="只回复一个词：OK",
            user="ping",
            json_mode=False,
            max_tokens=20,
        )
        return {"ok": True, "model": result.model, "reply": result.text.strip()[:40]}
    except LLMError as exc:
        return JSONResponse(status_code=200, content={"ok": False, "error": str(exc), "detail": exc.detail})


# Serve the built frontend when present.
#
# The frontend is one SPA bundle; the routes below all hand back the same
# index.html and the client picks the page from the path.
#
# A `StaticFiles` mount at "/" would swallow every unknown path with a 404, so
# real assets are served explicitly first and the known page routes fall through
# to index.html.
# __file__ is backend/app/main.py → parents[2] is the project root (backend/app → backend → root).
_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _DIST.is_dir():

    @app.get("/settings", include_in_schema=False)
    async def settings_page() -> FileResponse:
        return FileResponse(_DIST / "index.html")

    @app.get("/relay", include_in_schema=False)
    async def relay_page() -> FileResponse:
        # The relay UI moved into /settings; this route keeps old bookmarks alive
        # and lets the SPA redirect rather than returning a hard 404.
        return FileResponse(_DIST / "index.html")

    #: The bundle is rebuilt during development, so a cached copy hides changes:
    #: "I changed it and the page did not update" was this, twice. `no-cache`
    #: still allows a cached copy to exist but forces a revalidation each time,
    #: which is cheap (a 304) and removes the stale-page class of bug entirely.
    NO_CACHE = {
        "Cache-Control": "no-cache, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    @app.get("/bundle.js", include_in_schema=False)
    async def bundle_js() -> FileResponse:
        return FileResponse(
            _DIST / "bundle.js", media_type="text/javascript", headers=NO_CACHE
        )

    @app.get("/bundle.css", include_in_schema=False)
    async def bundle_css() -> FileResponse:
        return FileResponse(
            _DIST / "bundle.css", media_type="text/css", headers=NO_CACHE
        )

    #: 品牌标志：浏览器标签页图标（index.html 里 <link rel="icon">）和
    #: 收藏夹/桌面快捷方式都会来取。和 bundle 一样 no-cache，换 logo 时
    #: 刷新页面就能看到，不会被缓存钉死。
    @app.get("/logo.svg", include_in_schema=False)
    async def logo_svg() -> FileResponse:
        return FileResponse(
            _DIST / "logo.svg", media_type="image/svg+xml", headers=NO_CACHE
        )

    app.mount("/assets", StaticFiles(directory=str(_DIST)), name="assets")

    @app.get("/", include_in_schema=False)
    async def index_page() -> FileResponse:
        return FileResponse(_DIST / "index.html")
