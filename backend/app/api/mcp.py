"""MCP connection management.

This is intentionally not a provider marketplace.  The product currently has
one optional read-only external source (Feishu) plus its built-in knowledge
service, so the API exposes exactly those capabilities.
"""

from __future__ import annotations

import json
from html import escape

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..connectors import feishu
from ..kb import store
from ..platform.mcp_tools import outbound_server
from ..platform import tool_switches


router = APIRouter(prefix="/api/mcp")


class FeishuSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=30)


class FeishuFetchRequest(BaseModel):
    url: str = Field(min_length=10, max_length=1000)


class FeishuSettingsRequest(BaseModel):
    app_id: str = Field(min_length=5, max_length=100)
    app_secret: str = Field(default="", max_length=300)
    redirect_uri: str = Field(min_length=10, max_length=1000)


class ToolSwitchRequest(BaseModel):
    enabled: bool


@router.get("/directory")
async def directory() -> dict:
    names = outbound_server().tool_names()
    switches = tool_switches.load()
    tools = []
    if "search_knowledge" in names:
        tools.append({
            "id": "search_knowledge",
            "name": "资料检索",
            "kind": "builtin",
            "description": "检索本地资料库；本地没有时，再按检索设置去找已配置的权威网站。",
            "connected": True,
            "enabled": switches["knowledge"],
        })
    connection = await feishu.status()
    return {
        "tools": tools,
        "entries": store.stats()["total"],
        "feishu": {
            **connection,
            "id": "feishu",
            "name": "飞书云文档",
            "description": "仅在用户明确要求使用飞书资料时，搜索并读取当前账号有权限访问的云文档。",
            "permissions": ["搜索云文档", "读取云文档正文"],
            "enabled": switches["feishu"],
        },
    }


@router.put("/tools/{service}/enabled")
async def set_tool_enabled(service: str, payload: ToolSwitchRequest) -> dict:
    try:
        state = tool_switches.set_enabled(service, payload.enabled)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"service": service, "enabled": state[service], "switches": state}


@router.get("/feishu/status")
async def feishu_status() -> dict:
    return await feishu.status()


@router.put("/feishu/settings")
async def feishu_settings(payload: FeishuSettingsRequest) -> dict:
    try:
        return feishu.save_settings(payload.app_id, payload.app_secret, payload.redirect_uri)
    except feishu.FeishuError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/feishu/connect")
async def feishu_connect() -> dict:
    try:
        tool_switches.set_enabled("feishu", True)
        return feishu.start_login()
    except feishu.FeishuError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/feishu/cancel")
async def feishu_cancel() -> dict:
    """Abandon a pending authorization so the panel stops waiting."""
    try:
        return feishu.cancel_login()
    except feishu.FeishuError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.delete("/feishu")
async def feishu_disconnect() -> dict:
    try:
        return feishu.disconnect()
    except feishu.FeishuError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/feishu/callback", response_class=HTMLResponse)
async def feishu_callback(
    code: str = "",
    state: str = "",
    error: str = "",
    error_description: str = "",
) -> HTMLResponse:
    """Receive Feishu's browser redirect and show the result in place.

    Feishu lands here after the user scans. This page is opened in the *system
    browser*, not a popup, so it must stand on its own: it shows who logged in,
    what was actually granted, and the documents that account can now read.
    It is deliberately NOT auto-closed — the user opened a real browser tab and
    closing it under them would look like the login failed.
    """
    ok = False
    account = ""
    detail: dict = {}
    permissions: list[dict] = []
    documents: list[dict] = []
    documents_note = ""
    message = ""
    if error:
        message = f"飞书授权未完成：{error_description or error}"
    else:
        try:
            result = await feishu.complete_login(code, state)
            ok = bool(result.get("connected"))
            account = str(result.get("account") or "")
            detail = result.get("account_detail") or {}
            permissions = list(result.get("permissions") or [])
            message = "飞书授权成功，可以关闭本页回到主程序" if ok else "飞书授权完成，但连接状态异常"
        except feishu.FeishuError as exc:
            message = str(exc)
    if ok:
        try:
            listing = await feishu.list_documents(12)
            documents = list(listing.get("documents") or [])
            if listing.get("ok") is not True:
                documents_note = str(listing.get("reason") or "")
        except feishu.FeishuError as exc:
            # 登录已经成功，清单拿不到只影响这个页面的展示，不改结论。
            documents_note = str(exc)
    safe_message = escape(message)
    safe_account = escape(account)
    color = "#34d399" if ok else "#f87171"
    mark = "✓" if ok else "!"
    tenant = escape(str(detail.get("tenant") or ""))
    email = escape(str(detail.get("email") or ""))
    sub_line = " · ".join(part for part in (tenant, email) if part)

    permission_items = "".join(
        f"<li><code>{escape(str(item.get('scope') or ''))}</code>"
        f"<span>{escape(str(item.get('label') or ''))}</span></li>"
        for item in permissions
    ) or "<li><span>本次授权没有返回权限清单</span></li>"

    if documents:
        document_items = "".join(
            f"<li><a href='{escape(str(doc.get('url') or '#'))}' target='_blank' rel='noreferrer'>"
            f"{escape(str(doc.get('title') or '未命名文档'))}</a>"
            f"<span>{escape(str(doc.get('type') or ''))}"
            + (f" · {escape(str(doc.get('owner')))}" if doc.get("owner") else "")
            + "</span></li>"
            for doc in documents
        )
    else:
        document_items = f"<li><span>{escape(documents_note) or '暂无可见文档'}</span></li>"

    body_sections = (
        f"<div class='account'>{safe_account}"
        + (f"<small>{sub_line}</small>" if sub_line else "")
        + "</div>"
        "<h3>已授权权限</h3>"
        f"<ul class='list'>{permission_items}</ul>"
        "<h3>可访问的云文档</h3>"
        f"<ul class='list'>{document_items}</ul>"
    ) if ok else ""

    # 开发时回调页在后端（8100），开设置页的可能是 Vite（5173），
    # 所以不能假定同源；postMessage 让已经打开的设置页立刻刷新状态。
    # 消息里不含任何凭证。
    return HTMLResponse(
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>飞书授权</title><style>"
        "body{margin:0;background:#0b0d10;color:#e6e9ee;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;"
        "display:flex;justify-content:center;padding:48px 20px;box-sizing:border-box;min-height:100vh}"
        "main{width:100%;max-width:640px}"
        ".mark{width:64px;height:64px;border-radius:50%;border:2px solid var(--c);display:grid;"
        "place-items:center;font-size:30px;color:var(--c);margin-bottom:22px}"
        "h2{margin:0 0 12px;color:var(--c)}"
        "p{color:#9aa4b2;line-height:1.6;margin:0 0 20px}"
        "h3{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:#6b7280;"
        "margin:26px 0 10px;font-weight:600}"
        ".account{font-size:20px;font-weight:600;padding:16px 18px;border-radius:12px;"
        "background:#141820;border:1px solid #1f2530;display:flex;flex-direction:column;gap:4px}"
        ".account small{font-size:13px;font-weight:400;color:#8b95a5}"
        ".list{list-style:none;margin:0;padding:0;border:1px solid #1f2530;border-radius:12px;overflow:hidden}"
        ".list li{display:flex;justify-content:space-between;align-items:center;gap:16px;"
        "padding:11px 16px;border-top:1px solid #1f2530;font-size:14px;background:#11151c}"
        ".list li:first-child{border-top:none}"
        ".list a{color:#8ab4f8;text-decoration:none;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}"
        ".list a:hover{text-decoration:underline}"
        ".list code{color:#c9d4e4;font-size:12.5px}"
        ".list span{color:#6b7280;font-size:12.5px;flex-shrink:0;max-width:55%;text-align:right;"
        "overflow:hidden;text-overflow:ellipsis;white-space:nowrap}"
        "footer{margin-top:28px;font-size:12.5px;color:#525a68}"
        "</style></head><body><main style='--c:" + color + "'>"
        f"<div class='mark'>{mark}</div>"
        f"<h2>{'飞书登录完成' if ok else '飞书登录未完成'}</h2>"
        f"<p>{safe_message}</p>"
        + body_sections
        + "<footer>本页可以关闭。主程序已自动识别授权结果，无需再点任何按钮。</footer>"
        "</main>"
        f"<script>try{{if(window.opener&&!window.opener.closed)window.opener.postMessage("
        f"{{type:'feishu-oauth',ok:{str(ok).lower()},account:{json.dumps(account, ensure_ascii=False)}}},'*');}}catch(e){{}}"
        "</script></body></html>",
        status_code=200,
    )


@router.get("/feishu/docs")
async def feishu_documents(limit: int = 20) -> dict:
    """列出当前账号可访问的云文档。

    这是「扫码之后我能看到什么」的答案：搜索接口要求先想起关键词，
    刚登录的人需要的是一个清单。权限不足时返回 ok=False + 原因，
    而不是 500——登录本身是成功的，拿不到清单不该让界面以为掉线了。
    """
    try:
        return await feishu.list_documents(limit)
    except feishu.FeishuError as exc:
        raise HTTPException(409, str(exc)) from exc


# Read-only diagnostic endpoints make it possible to verify the connection from
# the settings UI or automated tests without starting a costly video run.
@router.post("/feishu/search")
async def feishu_search(payload: FeishuSearchRequest) -> dict:
    return await outbound_server().call("search_feishu_docs", {"query": payload.query})


@router.post("/feishu/fetch")
async def feishu_fetch(payload: FeishuFetchRequest) -> dict:
    return await outbound_server().call("fetch_feishu_doc", {"url": payload.url})

