"""Read the logs back out: the 日志 section of 设置, and the bundle to attach to a bug report.

Two audiences, one module:

  · **The user, while something is wrong.** `GET /api/diagnostics/logs` returns
    the recent events and the tail of both text logs, already filtered. They are
    looking for one line that explains a failure they just watched happen.
  · **Whoever they send it to.** `GET /api/diagnostics/bundle` produces one zip
    containing the logs, the run that failed, and the environment. Asking someone
    to "send me your providers.json and your studio.db" means asking them to send
    their API key and their whole library; the bundle carries what is needed to
    diagnose and nothing that is not.

Every field that could carry a credential is masked on the way out, in the same
place the logs were masked on the way in (`logs.mask`).
"""

from __future__ import annotations

import io
import json
import platform
import sys
import zipfile
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from .. import db, logs
from ..config import CONFIG_DIR, DATA_DIR, DB_PATH, PROVIDERS_PATH

router = APIRouter()


def _masked_providers() -> dict[str, Any]:
    """The provider file with keys replaced.

    Kept as a file in the bundle because "which endpoint, which model, which key
    is it using" is the first thing to check, and a key that starts with the
    wrong prefix is itself the answer — so the prefix survives, the rest does not.
    """
    try:
        raw = json.loads(PROVIDERS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"error": f"读不出来：{exc}", "path": str(PROVIDERS_PATH)}

    for provider in raw.get("providers") or []:
        key = str(provider.get("api_key") or "")
        provider["api_key"] = (
            f"{key[:6]}…{key[-4:]}（已截断，共 {len(key)} 位）" if len(key) > 12 else ("(空)" if not key else "(短)")
        )
    return raw


def _environment() -> str:
    try:
        from importlib.metadata import version

        packages = ", ".join(
            f"{name}={version(name)}"
            for name in ("fastapi", "uvicorn", "httpx", "pydantic", "numpy", "onnxruntime", "tokenizers")
        )
    except Exception:  # noqa: BLE001 - a diagnostic must not fail while collecting diagnostics
        packages = "(未能读取)"
    lines = [
        f"时间        {datetime.now().isoformat(timespec='seconds')}",
        f"Python      {sys.version.split()[0]}",
        f"平台        {platform.platform()}",
        f"架构        {platform.machine()}",
        f"依赖        {packages}",
        f"数据目录    {DATA_DIR}",
        f"配置目录    {CONFIG_DIR}",
        f"数据库      {DB_PATH}",
        f"日志目录    {logs.LOG_DIR}",
    ]
    try:
        lines.append(f"日志级别    {logs.level_name()}")
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)


def _run_snapshot(run_id: str) -> dict[str, Any]:
    record = db.get_run(run_id)
    if record is None:
        raise HTTPException(404, "运行不存在")
    return {
        "run": record,
        "messages": db.list_messages(run_id),
        "reviews": db.list_reviews(run_id),
        "steps": db.list_steps(run_id),
    }


@router.get("/api/diagnostics/logs")
async def get_logs(run_id: str = "", level: str = "", limit: int = 200) -> dict[str, Any]:
    """Recent events plus the tail of both text logs.

    `limit` is capped: the panel renders a list, and a thousand rows is slower to
    read than the twenty that matter. The full history stays on disk.
    """
    capped = max(1, min(int(limit or 200), 500))
    return {
        "events": logs.read_events(limit=capped, run_id=run_id.strip(), level=level.strip()),
        "summary": logs.summary(),
        "errors": logs.tail(logs.ERROR_LOG, 200),
        "app": logs.tail(logs.APP_LOG, 120),
        "run_id": run_id.strip(),
    }


@router.get("/api/diagnostics/bundle")
async def get_bundle(run_id: str = "") -> Response:
    """One zip: logs, environment, masked provider config, and optionally one run.

    Deliberately built in memory and returned as an attachment. Writing it into
    the data directory would leave a copy of the user's keys and book text lying
    around after they thought they had only looked at it.
    """
    buffer = io.BytesIO()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(
            "README.txt",
            "漫簇诊断包\n"
            "==========\n\n"
            "这个包里有什么：\n"
            "  environment.txt       运行环境与依赖版本\n"
            "  summary.json          日志文件清单与体积\n"
            "  app.log               最近的运行记录（尾部 800 行）\n"
            "  error.log             最近的错误（含完整堆栈）\n"
            "  events.jsonl          结构化事件，一行一次调用\n"
            "  providers.masked.json 供应商配置，Key 已截断\n"
            "  run.json              指定任务的完整记录（仅在指定了 run_id 时）\n\n"
            "没有包含什么：\n"
            "  API Key（已截断）、用户数据目录、上传的文件、生成的视频。\n\n"
            "如果要把这个包发给别人，先自己扫一眼 events.jsonl：\n"
            "模型提示词和返回内容的片段会出现在里面。\n",
        )
        bundle.writestr("environment.txt", _environment())
        bundle.writestr("summary.json", json.dumps(logs.summary(), ensure_ascii=False, indent=2))
        bundle.writestr("app.log", logs.tail(logs.APP_LOG, 800))
        bundle.writestr("error.log", logs.tail(logs.ERROR_LOG, 800))
        bundle.writestr(
            "providers.masked.json",
            json.dumps(_masked_providers(), ensure_ascii=False, indent=2),
        )

        events = logs.read_events(limit=3000, run_id=run_id.strip(), days=logs.KEEP_DAYS)
        bundle.writestr(
            "events.jsonl",
            "\n".join(json.dumps(item, ensure_ascii=False) for item in reversed(events)),
        )

        if run_id.strip():
            bundle.writestr(
                "run.json",
                json.dumps(_run_snapshot(run_id.strip()), ensure_ascii=False, indent=2, default=str),
            )

    filename = f"mancu-diagnostics-{stamp}.zip"
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
