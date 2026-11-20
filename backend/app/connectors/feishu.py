"""飞书只读连接器，走浏览器 OAuth 2.0。

用户在飞书网页里扫码或密码登录授权，后端用一次性 code 换 token，
用户 token 不会回传到浏览器。只实现文档检索和正文读取。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlencode, urlparse

import httpx

from ..config import CONFIG_DIR


AUTHORIZE_URL = "https://accounts.feishu.cn/open-apis/authen/v1/authorize"
TOKEN_URL = "https://accounts.feishu.cn/oauth/v3/token"
USER_INFO_URL = "https://open.feishu.cn/open-apis/authen/v1/user_info"
SEARCH_URL = "https://open.feishu.cn/open-apis/search/v2/doc_wiki/search"
#: 用户最近访问的文档。用来回答「扫码之后我到底能看哪些文档」——
#: 搜索接口要求用户先想起关键词，刚登录的人往往想不起。
RECENT_URL = "https://open.feishu.cn/open-apis/drive/v1/files"
WIKI_SPACES_URL = "https://open.feishu.cn/open-apis/wiki/v2/spaces"
API_BASE = "https://open.feishu.cn/open-apis"
REQUIRED_SCOPES = (
    "search:docs:read",
    "docx:document:readonly",
    "wiki:wiki:readonly",
    "drive:drive:readonly",
    "offline_access",
)
#: 用户能看懂的中文说明。登录成功后要和账号一起摆出来：
#: 「授权了哪些权限」比「连接成功」有用得多，也解释了为什么有的文档读不到。
SCOPE_LABELS = {
    "search:docs:read": "搜索云文档",
    "docx:document:readonly": "读取新版文档正文",
    "doc:document:readonly": "读取旧版文档正文",
    "wiki:wiki:readonly": "读取知识库",
    "drive:drive:readonly": "读取云空间文件列表",
    "offline_access": "离线访问（令牌过期后自动续期）",
}
MAX_DOCUMENT_CHARS = 60_000
_CONFIG_PATH = Path(CONFIG_DIR) / "feishu.json"
#: 待授权的 state 存在文件里，不再放内存字典。
#: 后端一重启（uvicorn --reload，或 start.ps1 重启端口），内存里的 state 就没了，
#: 用户扫完码回来正好收到「授权状态无效或已经过期」。
_STATE_PATH = Path(CONFIG_DIR) / "feishu_oauth.json"
STATE_TTL_SECONDS = 600
_ALLOWED_HOST_SUFFIXES = ("feishu.cn", "larksuite.com")
_token_lock = asyncio.Lock()


class FeishuError(RuntimeError):
    """A user-facing OAuth or OpenAPI failure with secrets removed."""


def _safe_error(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(
        r"(?i)(access_token|refresh_token|app_secret|client_secret|code)\s*[:=]\s*[^\s,}\]]+",
        r"\1=[已隐藏]",
        text,
    )
    return text[:800]


def _read_config() -> dict[str, Any]:
    data: dict[str, Any] = {}
    if _CONFIG_PATH.is_file():
        try:
            value = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                data = value
        except (OSError, ValueError):
            data = {}
    return {
        **data,
        "app_id": str(data.get("app_id") or os.environ.get("STUDIO_FEISHU_APP_ID") or "").strip(),
        "app_secret": str(data.get("app_secret") or os.environ.get("STUDIO_FEISHU_APP_SECRET") or "").strip(),
        "redirect_uri": str(data.get("redirect_uri") or os.environ.get("STUDIO_FEISHU_REDIRECT_URI") or "").strip(),
    }


def _write_config(data: dict[str, Any]) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = _CONFIG_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    temporary.replace(_CONFIG_PATH)


def _read_states() -> dict[str, float]:
    """读取待授权的 state。做过期过滤，后端重启后仍在。"""
    if not _STATE_PATH.is_file():
        return {}
    try:
        value = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(value, dict):
        return {}
    now = time.time()
    states: dict[str, float] = {}
    for key, expires in value.items():
        try:
            deadline = float(expires)
        except (TypeError, ValueError):
            continue
        if deadline > now:
            states[str(key)] = deadline
    return states


def _write_states(states: dict[str, float]) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    live = {key: expires for key, expires in states.items() if expires > now}
    temporary = _STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(live, ensure_ascii=False), encoding="utf-8")
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    temporary.replace(_STATE_PATH)


def cancel_login() -> dict[str, Any]:
    """取消登录：清掉所有待授权状态。

    关掉弹窗必须等于「我放弃了」。否则后端还要等十分钟才过期，
    界面一直停在「授权中」，用户只能干等。
    """
    dropped = len(_read_states())
    _write_states({})
    result = public_status()
    result["message"] = (
        "已取消本次飞书登录" if dropped
        else "当前没有进行中的飞书登录"
    )
    return result


def save_settings(app_id: str, app_secret: str, redirect_uri: str) -> dict[str, Any]:
    current = _read_config()
    client_id = str(app_id or "").strip()
    callback = str(redirect_uri or "").strip()
    secret = str(app_secret or "").strip() or str(current.get("app_secret") or "")
    if not client_id.startswith("cli_"):
        raise FeishuError("App ID 格式不正确，应以 cli_ 开头")
    if not secret:
        raise FeishuError("请填写 App Secret")
    parsed = urlparse(callback)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise FeishuError("回调地址必须是完整的 http:// 或 https:// 地址")
    changed = client_id != current.get("app_id") or callback != current.get("redirect_uri")
    saved = {**current, "app_id": client_id, "app_secret": secret, "redirect_uri": callback}
    if changed:
        for key in ("access_token", "refresh_token", "expires_at", "refresh_expires_at", "user", "scopes"):
            saved.pop(key, None)
    _write_config(saved)
    return public_status(saved)


def public_status(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or _read_config()
    configured = bool(cfg.get("app_id") and cfg.get("app_secret") and cfg.get("redirect_uri"))
    expires_at = float(cfg.get("expires_at") or 0)
    has_refresh = bool(cfg.get("refresh_token"))
    connected = bool(cfg.get("access_token")) and (expires_at > time.time() or has_refresh)
    state = "connected" if connected else "not_connected" if configured else "not_configured"
    user = cfg.get("user") if isinstance(cfg.get("user"), dict) else {}
    # 存下来的权限是「这次授权实际拿到的」，不是「我们申请的」。
    # 用户看到一个权限清单，才知道为什么某篇文档读不到。
    granted = [str(item) for item in (cfg.get("scopes") or []) if isinstance(item, str)]
    scopes = granted or ([] if not connected else list(REQUIRED_SCOPES))
    return {
        "configured": configured,
        "state": state,
        "connected": connected,
        "message": (
            "飞书已连接" if connected else
            "尚未连接飞书" if configured else
            "飞书登录服务尚未由管理员配置"
        ),
        "account": str(user.get("name") or user.get("en_name") or user.get("email") or ""),
        "account_detail": {
            "name": str(user.get("name") or ""),
            "en_name": str(user.get("en_name") or ""),
            "email": str(user.get("email") or ""),
            "avatar_url": str(user.get("avatar_url") or ""),
            "tenant": str(user.get("tenant_name") or user.get("tenant_key") or ""),
        },
        "scopes": scopes,
        "permissions": [
            {"scope": scope, "label": SCOPE_LABELS.get(scope, scope)}
            for scope in scopes
        ],
        "app_id": str(cfg.get("app_id") or ""),
        "app_secret_configured": bool(cfg.get("app_secret")),
        "redirect_uri": str(cfg.get("redirect_uri") or ""),
        "required_scopes": list(REQUIRED_SCOPES),
    }


async def status() -> dict[str, Any]:
    result = public_status()
    now = time.time()
    pending = any(expires > now for expires in _read_states().values())
    if pending and not result["connected"]:
        result.update(
            state="authorizing",
            message="等待在飞书网页完成扫码或密码授权",
        )
    return result


def start_login() -> dict[str, Any]:
    cfg = _read_config()
    current = public_status(cfg)
    if not current["configured"]:
        raise FeishuError(
            "飞书登录服务尚未由管理员配置，暂时无法生成官方授权链接。"
            "请由项目管理员在服务端设置飞书 App ID、App Secret 和回调地址。"
        )
    now = time.time()
    # 一次只保留一个待授权状态：新登录顶掉旧的，
    # 避免废弃的尝试越积越多，界面永远停在「授权中」。
    _write_states({})
    state = secrets.token_urlsafe(32)
    _write_states({state: now + STATE_TTL_SECONDS})
    params = {
        "client_id": cfg["app_id"],
        "response_type": "code",
        "redirect_uri": cfg["redirect_uri"],
        "scope": " ".join(REQUIRED_SCOPES),
        "prompt": "consent",
        "state": state,
    }
    return {
        **current,
        "state": "authorizing",
        "connected": False,
        "message": "请在飞书网页完成扫码或密码授权",
        "authorization_url": AUTHORIZE_URL + "?" + urlencode(params),
    }


async def _json_request(
    method: str,
    url: str,
    *,
    token: str = "",
    json_body: dict[str, Any] | None = None,
    form: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.request(method, url, headers=headers, json=json_body, data=form, params=params)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise FeishuError(f"飞书接口返回 HTTP {exc.response.status_code}：{_safe_error(exc.response.text[:500])}") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise FeishuError(f"无法连接飞书开放平台：{_safe_error(exc)}") from exc
    if not isinstance(payload, dict):
        raise FeishuError("飞书接口返回了无法识别的数据")
    code = payload.get("code")
    if code not in (None, 0, "0"):
        message = payload.get("msg") or payload.get("error_description") or payload.get("error") or "请求失败"
        raise FeishuError(f"飞书错误 {code}：{_safe_error(message)}")
    if payload.get("error") and not payload.get("access_token"):
        raise FeishuError(_safe_error(payload.get("error_description") or payload["error"]))
    data = payload.get("data", payload)
    return data if isinstance(data, dict) else {"value": data}


async def complete_login(code: str, state: str) -> dict[str, Any]:
    states = _read_states()
    key = str(state or "")
    expiry = states.pop(key, 0)
    _write_states(states)
    if not expiry or expiry < time.time():
        raise FeishuError("授权状态无效或已经过期，请从设置页重新连接")
    auth_code = str(code or "").strip()
    if not auth_code:
        raise FeishuError("飞书没有返回授权码")
    cfg = _read_config()
    token = await _json_request(
        "POST", TOKEN_URL,
        form={
            "grant_type": "authorization_code", "client_id": cfg["app_id"],
            "client_secret": cfg["app_secret"], "code": auth_code,
            "redirect_uri": cfg["redirect_uri"],
        },
    )
    access_token = str(token.get("access_token") or "")
    if not access_token:
        raise FeishuError("飞书没有返回 user_access_token，请检查应用权限和回调地址")
    try:
        user = await _json_request("GET", USER_INFO_URL, token=access_token)
    except FeishuError:
        user = {}
    now = time.time()
    cfg.update(
        access_token=access_token,
        refresh_token=str(token.get("refresh_token") or ""),
        expires_at=now + max(60, int(token.get("expires_in") or 7200)) - 60,
        refresh_expires_at=now + max(60, int(token.get("refresh_token_expires_in") or 2592000)) - 60,
        user=user,
        # 飞书可能在响应里直接回 scope；没回就按我们申请的记，
        # 权限不足时接口会自己报错，界面不必假装知道得更多。
        scopes=_parse_scopes(token) or list(REQUIRED_SCOPES),
    )
    _write_config(cfg)
    return public_status(cfg)


async def _access_token() -> str:
    async with _token_lock:
        cfg = _read_config()
        if not cfg.get("access_token"):
            raise FeishuError("飞书尚未授权，请在设置中连接飞书")
        if float(cfg.get("expires_at") or 0) > time.time() + 30:
            return str(cfg["access_token"])
        refresh = str(cfg.get("refresh_token") or "")
        if not refresh or float(cfg.get("refresh_expires_at") or 0) <= time.time():
            disconnect()
            raise FeishuError("飞书授权已经过期，请重新连接")
        token = await _json_request(
            "POST", TOKEN_URL,
            form={
                "grant_type": "refresh_token", "client_id": cfg["app_id"],
                "client_secret": cfg["app_secret"], "refresh_token": refresh,
            },
        )
        access = str(token.get("access_token") or "")
        if not access:
            raise FeishuError("飞书令牌刷新失败，请重新连接")
        now = time.time()
        cfg.update(
            access_token=access,
            refresh_token=str(token.get("refresh_token") or refresh),
            expires_at=now + max(60, int(token.get("expires_in") or 7200)) - 60,
            refresh_expires_at=now + max(60, int(token.get("refresh_token_expires_in") or 2592000)) - 60,
        )
        _write_config(cfg)
        return access


def disconnect() -> dict[str, Any]:
    cfg = _read_config()
    for key in ("access_token", "refresh_token", "expires_at", "refresh_expires_at", "user", "scopes"):
        cfg.pop(key, None)
    _write_config(cfg)
    return public_status(cfg)


def _parse_scopes(token: dict[str, Any]) -> list[str]:
    """从令牌响应里取出实际授予的 scope。

    飞书 OAuth v3 把 scope 放在 `scope`（空格分隔）或 `scopes`（数组）里，
    两种都见过，所以都读一遍而不是赌其中一种。
    """
    raw = token.get("scope")
    if isinstance(raw, str) and raw.strip():
        return [item for item in raw.replace(",", " ").split() if item]
    value = token.get("scopes")
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return []


def _normalize_doc_type(value: Any) -> str:
    """把飞书的多种类型写法收敛成 docx/doc/sheet/bitable/wiki/其它。"""
    text = str(value or "").strip().lower()
    if text in {"doc", "docx"}:
        return "docx"
    if text in {"sheet", "spreadsheet", "bitable", "base"}:
        return text
    if text == "wiki":
        return "wiki"
    return text or "file"


def _file_url(token: str, doc_type: str) -> str:
    """按类型拼出可读链接。

    链接默认落在用户自己的域（feishu.cn / larksuite.com 都可能），
    这里不猜租户域名，交给前端展示标题 + 类型；需要正文时走 fetch_document。
    """
    kind = _normalize_doc_type(doc_type)
    if kind == "docx":
        return f"https://feishu.cn/docx/{token}"
    if kind == "sheet":
        return f"https://feishu.cn/sheets/{token}"
    if kind == "bitable":
        return f"https://feishu.cn/base/{token}"
    return f"https://feishu.cn/file/{token}"


async def list_documents(limit: int = 20) -> dict[str, Any]:
    """列出当前账号可访问的云文档。

    「登录成功后能看到自己的云文档」需要的是一个列表，而不是搜索框：
    刚扫码的人多半想不起关键词，只想验收「我确实连上了、能看到自己的文件」。
    这里用云空间文件列表做入口，失败时退回到「无内容但连接正常」，
    不让一个可选接口把整条登录链路判成失败。
    """
    count = max(1, min(int(limit or 20), 50))
    access = await _access_token()
    try:
        data = await _json_request(
            "GET", RECENT_URL, token=access,
            params={"page_size": count, "order_by": "EditedTime", "direction": "DESC"},
            timeout=30.0,
        )
    except FeishuError as exc:
        # 云空间列表需要 drive 权限，很多租户只开了文档读取。
        # 这不是错误，只是「拿不到清单」——如实说明，界面照样可用。
        return {
            "ok": False,
            "documents": [],
            "count": 0,
            "reason": "当前授权没有云空间列表权限，可直接用搜索查找文档",
            "detail": str(exc),
        }
    documents: list[dict[str, str]] = []
    for raw in data.get("files") or []:
        if not isinstance(raw, dict):
            continue
        token = str(raw.get("token") or raw.get("file_token") or "").strip()
        if not token:
            continue
        kind = _normalize_doc_type(raw.get("type"))
        owner = str(raw.get("owner_name") or raw.get("owner_id") or "")
        documents.append({
            "title": _clean_title(raw.get("name") or raw.get("title")) or "未命名文档",
            "token": token,
            "type": kind,
            "url": str(raw.get("url") or _file_url(token, kind)),
            "owner": owner,
            "edited_at": str(raw.get("modified_time") or raw.get("edit_time") or ""),
        })
    return {"ok": True, "documents": documents, "count": len(documents)}


def _valid_document_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    return parsed.scheme == "https" and any(host == suffix or host.endswith("." + suffix) for suffix in _ALLOWED_HOST_SUFFIXES)


def _clean_title(value: Any) -> str:
    title = re.sub(r"<[^>]+>", "", str(value or ""))
    return re.sub(r"\s+", " ", title).strip()[:200]


def _find_results(value: Any) -> list[Any]:
    if isinstance(value, dict):
        for key in ("results", "items", "docs", "files"):
            if isinstance(value.get(key), list):
                return value[key]
        for child in value.values():
            found = _find_results(child)
            if found:
                return found
    return []


async def search_documents(query: str) -> dict[str, Any]:
    term = str(query or "").strip()[:30]
    if not term:
        return {"ok": False, "results": [], "reason": "飞书文档检索词为空"}
    token = await _access_token()
    data = await _json_request(
        "POST", SEARCH_URL, token=token,
        json_body={"query": term, "page_size": 10}, timeout=45.0,
    )
    results: list[dict[str, str]] = []
    for raw in _find_results(data):
        if not isinstance(raw, dict):
            continue
        meta = raw.get("result_meta") if isinstance(raw.get("result_meta"), dict) else {}
        url = str(raw.get("url") or meta.get("url") or "").strip()
        if not _valid_document_url(url):
            continue
        results.append({
            "title": _clean_title(raw.get("title") or raw.get("title_highlighted") or meta.get("title")) or "未命名飞书文档",
            "url": url,
            "type": str(raw.get("entity_type") or raw.get("type") or meta.get("doc_type") or "document")[:40],
        })
    return {"ok": True, "query": term, "results": results, "count": len(results)}


def _document_ref(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    for index, part in enumerate(parts[:-1]):
        if part in {"docx", "docs", "wiki"}:
            token = re.sub(r"[^A-Za-z0-9_-]", "", parts[index + 1])
            if token:
                return part, token
    query = parse_qs(parsed.query)
    token = str((query.get("token") or [""])[0])
    kind = str((query.get("type") or [""])[0])
    if token and kind in {"docx", "docs", "wiki"}:
        return kind, re.sub(r"[^A-Za-z0-9_-]", "", token)
    raise FeishuError("无法从飞书链接识别文档编号")


async def _resolve_wiki(node_token: str, access: str) -> tuple[str, str]:
    node = await _json_request(
        "GET", f"{API_BASE}/wiki/v2/spaces/get_node", token=access,
        params={"token": node_token},
    )
    value = node.get("node") if isinstance(node.get("node"), dict) else node
    obj_token = str(value.get("obj_token") or "")
    obj_type = str(value.get("obj_type") or "")
    if not obj_token:
        raise FeishuError("飞书知识库节点没有对应的文档")
    return obj_type, obj_token


async def fetch_document(url: str) -> dict[str, Any]:
    target = str(url or "").strip()
    if not _valid_document_url(target):
        return {"ok": False, "url": target, "reason": "只允许读取飞书或 Lark 云文档链接", "error_type": "invalid_url"}
    kind, document_token = _document_ref(target)
    access = await _access_token()
    if kind == "wiki":
        kind, document_token = await _resolve_wiki(document_token, access)
    if kind == "docx":
        data = await _json_request(
            "GET", f"{API_BASE}/docx/v1/documents/{quote(document_token)}/raw_content",
            token=access, timeout=60.0,
        )
    elif kind in {"docs", "doc"}:
        data = await _json_request(
            "GET", f"{API_BASE}/doc/v2/{quote(document_token)}/raw_content",
            token=access, timeout=60.0,
        )
    else:
        return {"ok": False, "url": target, "reason": f"暂不支持读取此飞书资源类型：{kind}", "error_type": "unsupported_document_type"}
    body = str(data.get("content") or data.get("raw_content") or data.get("text") or "").strip()
    body = "".join(ch for ch in body if ch in "\n\t" or ord(ch) >= 32)[:MAX_DOCUMENT_CHARS]
    if not body:
        return {"ok": False, "url": target, "reason": "飞书返回了空文档，或当前账号没有正文读取权限", "error_type": "empty_document"}
    return {"ok": True, "url": target, "title": "", "text": body, "chars": len(body), "material_ready": True}
