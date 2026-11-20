"""知识库的 HTTP 接口。

界面上的那些操作——看列表、看条目、入库、放弃、手动检索、填检索 key——都在这里。
没有"就地编辑知识"的入口：条目要么是检索生成的，要么是人工录入并确认过的，改一处
正文而不留痕，等于让这条知识失去了可追溯性。要改就删掉重建。

一个约定：`approved` 才代表"这是我们的知识"，`pending` 是"刚查到的草稿"。列表
默认两者都返回，因为用户需要看到待审核的队列；但**检索只用 approved 优先、pending
可用**——理由见 `kb/store.search` 的注释。
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .. import db
from ..kb import ResearchError, document, import_book, research, store
from ..kb import vector
from ..kb import websearch
from ..kb.authority import allowed_sites, normalize_sites

router = APIRouter()


class ResearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=120, description="主题")
    allow_web: bool = Field(default=True, description="库里没有时是否联网检索")
    entry_id: str = Field(
        default="",
        max_length=40,
        description="可选：要刷新的条目 id。给了就更新这一条，而不是新建",
    )


class AllowedSiteRequest(BaseModel):
    domain: str = Field(min_length=1, max_length=500)
    label: str = Field(default="", max_length=80)
    tier: int = Field(default=2, ge=1, le=2)


class SettingsRequest(BaseModel):
    provider: str = Field(default="bocha", description="bocha | tavily | serper")
    api_key: str = Field(default="", description="搜索服务 Key；留空表示清除")
    endpoint: str = Field(default="", description="可选：自定义接口地址")
    allowed_sites: list[AllowedSiteRequest] | None = Field(
        default=None,
        max_length=100,
        description="完整的允许来源名单；不传表示保留现有设置",
    )


class ManualSource(BaseModel):
    kind: str = Field(default="参考资料", max_length=20)
    org: str = Field(default="", max_length=80)
    title: str = Field(default="", max_length=160)
    url: str = Field(default="", max_length=500)


class ManualEntryRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)
    category: str = Field(default="历史知识", max_length=40)
    period: str = Field(default="", max_length=40)
    body: str = Field(min_length=20, max_length=30000)
    key_points: list[str] = Field(default_factory=list, max_length=20)
    sources: list[ManualSource] = Field(default_factory=list, max_length=8)


@router.get("/api/knowledge")
async def list_entries(status: str = "", q: str = "", kind: str = "", limit: int = 100) -> dict:
    """条目列表 + 各状态数量，一次返回，避免界面刚打开时闪两下。"""
    entries = store.list_entries(status=status.strip(), q=q, kind=kind.strip(), limit=limit)
    # 整本资料的章节标题往往就是「1921 年」这种层级名，面板光画标题等于什么都没说。
    # 把章内事件标题一起带上，用户才知道这本书里到底有哪几件事。
    from ..kb import vector

    for entry in entries:
        if entry.get("kind") == store.BOOK:
            entry["chapters"] = vector.outline(
                entry["id"], order=list(entry.get("key_points") or [])
            )
    return {
        "entries": entries,
        "stats": store.stats(),
        "statuses": [
            {"value": "", "label": "全部"},
            {"value": store.APPROVED, "label": "已入库"},
            {"value": store.PENDING, "label": "待审核"},
            {"value": store.REJECTED, "label": "已放弃"},
        ],
        "kinds": [
            {"value": "", "label": "全部类型"},
            {"value": store.CARD, "label": store.KIND_LABEL[store.CARD]},
            {"value": store.BOOK, "label": store.KIND_LABEL[store.BOOK]},
        ],
    }


@router.post("/api/knowledge", status_code=201)
async def create_manual_entry(payload: ManualEntryRequest) -> dict:
    """Create a human-supplied, immediately approved knowledge entry.

    Manual text is not presented as web-verified: ``origin=manual`` and the note
    remain visible in the detail view.  It is approved because the human supplied
    it deliberately; forcing a second "approve" click would add ceremony without
    adding another reviewer.
    """
    body = payload.body.strip()
    if len(body) < 20:
        raise HTTPException(422, "知识正文至少需要 20 个字")
    points = [p.strip()[:200] for p in payload.key_points if p.strip()][:20]
    sources = [s.model_dump() for s in payload.sources]
    entry = store.save(
        {
            "title": payload.title.strip(),
            "category": payload.category.strip() or "历史知识",
            "period": payload.period.strip(),
            "body": body,
            "key_points": points,
            "sources": sources,
            "status": store.APPROVED,
            "origin": "manual",
            "query": payload.title.strip(),
            "note": "人工录入并确认；来源由录入者提供",
        }
    )
    index = await vector.index_entry(entry)
    return {"entry": entry, "index": index}


@router.get("/api/knowledge/settings")
async def get_settings() -> dict:
    """检索设置。**key 只回是否已配置和尾部四位**，不把明文送回浏览器。"""
    cfg = websearch.load_config()
    key = cfg["api_key"]
    return {
        "provider": cfg["provider"],
        "has_key": bool(key),
        "key_hint": f"****{key[-4:]}" if len(key) >= 4 else "",
        "endpoint": cfg["endpoint"],
        "providers": [
            {"value": "bocha", "label": "博查 Bocha（中文推荐）"},
            {"value": "tavily", "label": "Tavily"},
            {"value": "serper", "label": "Serper（Google）"},
        ],
        "allowed_sites": [
            {"domain": suffix, "label": label, "tier": tier}
            for suffix, label, tier in allowed_sites()
        ],
    }


@router.post("/api/knowledge/settings")
async def save_settings(payload: SettingsRequest) -> dict:
    known = {"bocha", "tavily", "serper"}
    if payload.provider.strip() not in known:
        raise HTTPException(400, f"未知检索服务：{payload.provider}")
    current = websearch.load_config()
    # 留空 = 不改，而不是清空：用户只想换个服务商时不该被迫重填 key。
    key = payload.api_key.strip() or current["api_key"]
    sites = None
    if payload.allowed_sites is not None:
        try:
            sites = normalize_sites([site.model_dump() for site in payload.allowed_sites])
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    endpoint = payload.endpoint.strip() or current["endpoint"]
    websearch.save_config(
        payload.provider,
        key,
        endpoint,
        allowed_sites=sites,
    )
    return await get_settings()


@router.post("/api/knowledge/research")
async def run_research(payload: ResearchRequest) -> dict:
    """手动触发一次检索，方便在界面里先囤材料，不必真的跑一条片。

    带 `entry_id` 时是「刷新这一条」：后端会把该条目已抓到的正文带进新一轮
    整理，抓到更完整的材料就更新它本身，而不是在库里再添一条。
    """
    entry_id = payload.entry_id.strip()
    if entry_id and store.get(entry_id) is None:
        raise HTTPException(404, "要刷新的条目不存在")
    try:
        result = await research(
            payload.query,
            allow_web=payload.allow_web,
            entry_id=entry_id,
        )
    except ResearchError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "entry": result["entry"],
        "source": result["source"],
        # 是主题命中还是语义召回。界面上这两者该有不同的说法：语义召回是
        # 「按意思找到的」，用户据此判断要不要采信，比只看到「命中」有用。
        "retrieval": result.get("retrieval") or "topic",
        "trace": result["trace"],
        # 命中书时，这一段的出处是「1927 年 · 南昌起义」。卡片为空。界面用它
        # 说明「答案在哪」，否则用户只看到书名，还得自己翻。
        "kind": result.get("kind") or "card",
        "locator": result.get("locator") or "",
        "matched_chunk": result.get("matched_chunk") or "",
    }


# ============================================================================
# 语义索引的运维接口
#
# 这一层最危险的失败是**静默**的：没配模型时检索照样返回结果（只是走了主题
# 检索），索引过期时召回直接变空却不报错。所以健康状态、探活、重建单独开出来，
# 让「语义检索到底生效了没有」能一眼看到，而不是靠猜。
# ============================================================================


@router.get("/api/knowledge-index/health")
async def vector_health() -> dict:
    return vector.health()


@router.post("/api/knowledge-index/probe")
async def vector_probe() -> dict:
    """真调一次嵌入接口，证明端点通不通。状态页的「测试连接」用它。

    和 health 分开是有意的：health 只读配置（高频调用要便宜），probe 付一次
    网络往返。把两者合成一个接口，要么状态页变慢，要么状态是假的。
    """
    return await vector.probe()


@router.post("/api/knowledge-index/reindex")
async def vector_reindex(force: bool = False) -> dict:
    """重建语义索引。

    默认只重嵌内容变过的条目——全量重建的代价是每条目一次嵌入调用，几百条时
    明显变慢，也更容易撞上提供方限流。
    """
    rows = store.list_entries(limit=500)
    alive = [store.get(str(row["id"])) for row in rows]
    entries = [entry for entry in alive if entry]
    if not force:
        stale = set(vector.stale_entry_ids(entries))
        entries = [entry for entry in entries if str(entry.get("id")) in stale]
    result = await vector.reindex_all(entries, force=force)
    return {**result, "candidates": len(entries), "health": vector.health()}


#: 上传接口的体量上限。六万字的中文书约 180 KB；20 MB 够放一本带排版的大部头，
#: 又不至于让一次误传整个目录把内存吃光。
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

#: 第一期只解析文本。pdf / docx 需要各自的解析库，而那会引入新的安装依赖；
#: 与其装一半再报错，不如明确告诉用户该先转成 txt 或 md。
TEXT_SUFFIXES = (".txt", ".md", ".markdown", ".text")


@router.post("/api/knowledge/documents", status_code=201)
async def upload_document(
    request: Request,
    filename: str = "",
    title: str = "",
    category: str = "整本资料",
) -> dict:
    """上传一整本文本资料，切分、入库、建索引。

    接受两种 body，共用同一段处理逻辑：

    · **原始字节**（`text/plain`、`application/octet-stream`，或直接用 `File`
      作为 fetch 的 body）—— 这是前端走的路。收原始字节而不是 JSON 里的字符串，
      是因为中文 txt 有一半是 GBK，前端按 UTF-8 读出来就是乱码，而**乱码入库
      不会报错**，只会在之后表现为「这本书什么都搜不到」。编码只能在后端按字节判断。

    · **JSON**（`{"content": "..."}`）—— 给脚本和测试用，内容已经是解码后的文本。
    """
    content_type = (request.headers.get("content-type") or "").lower()
    raw = await request.body()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"文件超过 {MAX_UPLOAD_BYTES // 1024 // 1024} MB，请拆分后再传")
    if not raw:
        raise HTTPException(400, "文件是空的")

    suffix = ""
    name = (filename or "").strip()
    if "json" in content_type:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(400, f"请求体不是合法 JSON：{exc}") from exc
        text = str(payload.get("content") or "")
        name = name or str(payload.get("filename") or "")
        title = title or str(payload.get("title") or "")
        category = category or str(payload.get("category") or "整本资料")
        encoding = "utf-8(json)"
    else:
        text, encoding = document.decode(raw, filename=name)
        suffix = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""

    if suffix and suffix not in TEXT_SUFFIXES:
        raise HTTPException(
            415,
            f"暂不支持 {suffix} 格式，请先转成 txt 或 md。"
            "（pdf / docx 的解析要另外装库，排在下一批）",
        )
    if not text.strip():
        raise HTTPException(400, "文件里没有可读文本")

    try:
        result = await import_book(
            text=text, title=title, filename=name, category=category
        )
    except ResearchError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "entry": result["entry"],
        "structure": result["structure"],
        "index": result["index"],
        "replaced": result["replaced"],
        "encoding": encoding,
        "bytes": len(raw),
    }


@router.get("/api/knowledge/{entry_id}")
async def get_entry(entry_id: str) -> dict:
    entry = store.get(entry_id)
    if entry is None:
        raise HTTPException(404, "条目不存在")
    return {"entry": entry}


@router.post("/api/knowledge/{entry_id}/approve")
async def approve(entry_id: str) -> dict:
    entry = store.decide(entry_id, store.APPROVED)
    if entry is None:
        raise HTTPException(404, "条目不存在")
    # 转正后立刻进索引：下一次运行就该能语义召回它。
    index = await vector.index_entry(entry)
    return {"entry": entry, "index": index}


@router.post("/api/knowledge/{entry_id}/reject")
async def reject(entry_id: str) -> dict:
    entry = store.decide(entry_id, store.REJECTED)
    if entry is None:
        raise HTTPException(404, "条目不存在")
    # 弃用的内容必须从索引里撤掉。留着它还会被语义召回，等于悄悄推翻了审核
    # 决定——而「已放弃」正是这个库最不该被绕过的那个状态。
    removed = vector.remove_entry(entry_id)
    return {"entry": entry, "removed_chunks": removed}


@router.delete("/api/knowledge/{entry_id}")
async def delete(entry_id: str) -> dict:
    if store.get(entry_id) is None:
        raise HTTPException(404, "条目不存在")
    store.delete(entry_id)
    # 外键级联也能删，但那条路径依赖 PRAGMA foreign_keys，显式删一次更稳。
    vector.remove_entry(entry_id)
    return {"deleted": entry_id, "stats": store.stats()}


@router.post("/api/knowledge/manual")
async def create_manual(payload: dict[str, Any]) -> dict:
    """手写一条。用于外部检索查不到、但用户自己手里有材料的情况。"""
    title = str(payload.get("title") or "").strip()
    body = str(payload.get("body") or "").strip()
    if len(title) < 2 or len(body) < 20:
        raise HTTPException(400, "标题至少 2 个字，正文至少 20 个字")
    entry = store.save(
        {
            "title": title[:40],
            "category": str(payload.get("category") or "历史知识")[:16],
            "period": str(payload.get("period") or "")[:24],
            "body": body,
            "key_points": [
                str(p)[:160] for p in (payload.get("key_points") or []) if str(p).strip()
            ][:12],
            "sources": store.to_source_list(payload.get("sources") or []),
            "status": store.APPROVED,      # 人工录入即人已确认
            "origin": "manual",
            "query": "",
            "note": "手动录入",
        }
    )
    index = await vector.index_entry(entry)
    return {"entry": entry, "index": index}


@router.get("/api/knowledge/mcp/tools")
async def mcp_tools() -> dict:
    """列出知识库暴露给 MCP 的工具，方便排查"外部客户端为什么调不到"。

    返回的是工具名与描述，不含实现。放在这里而不是 /api/mcp，是因为它只和知识库
    相关；将来工具多了再拆出去。
    """
    from ..platform.mcp_tools import outbound_server

    server = outbound_server()
    names = [n for n in server.tool_names() if "knowledge" in n]
    return {
        "tool_names": names,
        "note": "启动 `python -m app.mcp_main` 即可供外部客户端调用",
        "entries": store.stats()["total"],
    }
