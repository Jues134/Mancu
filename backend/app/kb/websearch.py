"""外部检索：搜索服务适配 + 正文抓取 + 检索设置。

配了 key 就走搜索 API（博查 / Tavily / Serper），存在 `config/knowledge.json`：

    博查 bocha   国内可用，中文结果最好，推荐
    Tavily      英文为主，但支持中文查询
    Serper      走 Google，需要能出去

**没配 key 也能联网**：用模型自带的联网检索（`tools: [{type: web_search}]`）。这条路
实测可靠——同一个查询它返回的是共产党员网、党史和文献研究院、求是网的原文链接，
而"直连必应结果页"返回的是完全不相关的站点（查「党的诞生」给出降噪耳机），所以
爬搜索引擎这条路已经删掉了：宁可查不到，也不能把不相干的内容当史料。

模型的联网结果同样过白名单，宁可少也不能拿百科凑数。

抓正文这一步不可避免：搜索接口只给摘要，而摘要常常是导语，用来写脚本不够。抓不下
来时要如实返回失败，不能拿摘要凑一条知识进库。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

#: 20 秒：政府网站偶尔很慢，但超过这个时间说明站点不想被程序访问。
FETCH_TIMEOUT = 20.0
#: 单页正文上限。一条知识条目用不到一万字，多抓的只会挤占摘要模型的上下文。
MAX_PAGE_CHARS = 12000


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    site: str = ""


class SearchError(RuntimeError):
    """The external search service failed, as opposed to returning zero hits."""

    def __init__(self, message: str, *, kind: str = "search_error") -> None:
        super().__init__(message)
        self.kind = kind


class FetchError(RuntimeError):
    """A classified page-download failure suitable for MCP and UI reporting."""

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        url: str = "",
        status_code: int = 0,
        retryable: bool = False,
        action: str = "",
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.url = url
        self.status_code = status_code
        self.retryable = retryable
        self.action = action

    def as_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.kind,
            "status_code": self.status_code or None,
            "retryable": self.retryable,
            "action": self.action,
            "reason": str(self),
        }


def config_path() -> Path:
    from ..config import CONFIG_DIR

    return Path(CONFIG_DIR) / "knowledge.json"


def load_config() -> dict[str, Any]:
    path = config_path()
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
    provider = str(data.get("provider") or os.environ.get("STUDIO_SEARCH_PROVIDER") or "bocha")
    api_key = str(data.get("api_key") or os.environ.get("STUDIO_SEARCH_API_KEY") or "")
    direct = data.get("direct", True)
    if isinstance(direct, str):
        direct = direct.strip().lower() not in ("0", "false", "no", "off")
    return {
        "provider": provider,
        "api_key": api_key,
        "endpoint": str(data.get("endpoint") or ""),
        "max_results": int(data.get("max_results") or 8),
        # None means an installation created before editable allowlists and uses
        # the reviewed defaults. An empty list is intentional: allow no website.
        "allowed_sites": data.get("allowed_sites") if isinstance(data.get("allowed_sites"), list) else None,
        #: 没配 key 时是否直连必应结果页。默认开——装完就能查，才叫"可以联网"。
        "direct": bool(direct),
    }


def save_config(
    provider: str,
    api_key: str,
    endpoint: str = "",
    *,
    allowed_sites: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    current = load_config()
    data = {
        "provider": (provider or "bocha").strip(),
        "api_key": (api_key or "").strip(),
        "endpoint": (endpoint or "").strip(),
        "max_results": current.get("max_results", 8),
        "direct": current.get("direct", True),
    }
    previous_sites = current.get("allowed_sites")
    if allowed_sites is not None:
        data["allowed_sites"] = allowed_sites
    elif previous_sites is not None:
        data["allowed_sites"] = previous_sites
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return load_config()


def search_available() -> bool:
    """Can we search at all? A key is preferred, the direct fallback also counts."""
    cfg = load_config()
    if cfg["api_key"]:
        return True
    return bool(cfg.get("direct", True))


DEFAULT_ENDPOINTS = {
    "bocha": "https://api.bochaai.com/v1/web-search",
    "tavily": "https://api.tavily.com/search",
    "serper": "https://google.serper.dev/search",
}

#: 模型联网返回的是几行「标题 | 链接」，解析这一层比解析网页结构稳。
_LINE_URL_RE = re.compile(r"https?://[^\s)\]>\"'，。；）]+")
_MARKUP_RE = re.compile(r"[*#>`\[\]（）()「」【】：:—–\-]+")


async def _model_search(query: str, limit: int) -> list[SearchResult]:
    """Search through the model's own web tool.

    Used when no search API key is configured. The instruction is deliberately
    narrow — one line per hit, `标题 | 链接` — because anything looser comes back as
    prose with links buried inside it, and parsing prose for URLs picks up
    unrelated ones (a "参考链接" line at the bottom, the site's homepage).
    """
    import httpx as _httpx

    from .. import models_config

    try:
        profile = models_config.load().provider_for("director")
        config = models_config.provider_to_llm_config(profile) if profile else None
    except Exception:
        config = None
    if config is None or not config.configured:
        return []

    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是检索助手。用联网检索找权威来源。"
                    "只输出结果本身，每行一条，格式严格为：标题 | 链接。"
                    "不要编号，不要解释，不要加粗，不要写来源说明。最多 6 行。"
                ),
            },
            {"role": "user", "content": f"检索：{query}"},
        ],
        "tools": [{"type": "web_search", "web_search": {"enable": True, "search_result": True}}],
        "max_tokens": 1200,
        # 搜索不需要推理；开着思考这条请求会被思考吃光预算，返回一个"我"字。
        "thinking": {"type": "disabled"},
    }
    try:
        async with _httpx.AsyncClient(timeout=config.timeout_seconds) as client:
            resp = await client.post(
                config.base_url.rstrip("/") + "/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except _httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        kind = "model_web_unsupported" if status in (400, 404, 422) else "model_search_http"
        hint = (
            "当前模型接口可能不支持 web_search 联网工具"
            if kind == "model_web_unsupported"
            else "模型联网检索接口返回错误"
        )
        raise SearchError(f"{hint}（HTTP {status}）", kind=kind) from exc
    except (_httpx.ConnectError, _httpx.ProxyError, _httpx.TimeoutException) as exc:
        raise SearchError(
            f"无法连接模型联网检索接口（{type(exc).__name__}）",
            kind="model_search_connection",
        ) from exc
    content = str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")

    out: list[SearchResult] = []
    seen: set[str] = set()
    for line in content.splitlines():
        urls = _LINE_URL_RE.findall(line)
        if not urls:
            continue
        url = urls[0].rstrip(".,;")
        if url in seen:
            continue
        seen.add(url)
        title = _LINE_URL_RE.sub(" ", line)
        title = _MARKUP_RE.sub(" ", title).strip()
        out.append(SearchResult(title=title[:120] or url, url=url))
        if len(out) >= limit:
            break
    if not out:
        raise SearchError(
            "模型联网调用成功，但没有返回可解析的网页链接；"
            "可能是该模型不支持 web_search，或本次检索确实没有结果",
            kind="model_search_empty",
        )
    return out


async def search(query: str, *, limit: int = 8) -> list[SearchResult]:
    """Query the configured service, or the model's web tool. Never raises on no results."""
    cfg = load_config()
    key = cfg["api_key"]
    provider = cfg["provider"]
    count = max(1, min(20, limit or cfg["max_results"]))

    if not key:
        if not cfg.get("direct", True):
            return []
        return await _model_search(query, count)

    endpoint = cfg["endpoint"] or DEFAULT_ENDPOINTS.get(provider, "")
    if not endpoint:
        return []

    # A local proxy in the environment silently breaks some of these calls
    # (`HTTP_PROXY` pointing at 127.0.0.1:7890 answered 502 in this project once).
    # The search call is to a public endpoint, so honouring the proxy is correct;
    # what is not correct is failing without saying so.
    try:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT) as client:
            if provider == "bocha":
                resp = await client.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"query": query, "count": count, "summary": True, "freshness": "noLimit"},
                )
                resp.raise_for_status()
                return _parse_bocha(resp.json())
            if provider == "tavily":
                resp = await client.post(
                    endpoint,
                    headers={"Content-Type": "application/json"},
                    json={
                        "api_key": key, "query": query, "max_results": count,
                        "search_depth": "basic", "include_answer": False,
                    },
                )
                resp.raise_for_status()
                return _parse_tavily(resp.json())
            if provider == "serper":
                resp = await client.post(
                    endpoint,
                    headers={"X-API-KEY": key, "Content-Type": "application/json"},
                    json={"q": query, "gl": "cn", "hl": "zh-cn", "num": count},
                )
                resp.raise_for_status()
                return _parse_serper(resp.json())
    except httpx.HTTPStatusError as exc:
        raise SearchError(
            f"外部检索服务返回 HTTP {exc.response.status_code}",
            kind="search_service_http",
        ) from exc
    except (httpx.ConnectError, httpx.ProxyError, httpx.TimeoutException) as exc:
        raise SearchError(
            f"无法连接外部检索服务（{type(exc).__name__}）",
            kind="search_service_connection",
        ) from exc
    return []


def _parse_bocha(data: dict[str, Any]) -> list[SearchResult]:
    """Bocha nests results two levels deep: data.webPages.value[]."""
    pages = ((data.get("data") or {}).get("webPages") or {}).get("value") or []
    out: list[SearchResult] = []
    for item in pages:
        url = str(item.get("url") or "")
        if not url:
            continue
        out.append(
            SearchResult(
                title=str(item.get("name") or item.get("title") or url),
                url=url,
                snippet=str(item.get("summary") or item.get("snippet") or ""),
                site=str(item.get("siteName") or ""),
            )
        )
    return out


def _parse_tavily(data: dict[str, Any]) -> list[SearchResult]:
    return [
        SearchResult(
            title=str(r.get("title") or r.get("url") or ""),
            url=str(r.get("url") or ""),
            snippet=str(r.get("content") or ""),
        )
        for r in (data.get("results") or [])
        if r.get("url")
    ]


def _parse_serper(data: dict[str, Any]) -> list[SearchResult]:
    return [
        SearchResult(
            title=str(r.get("title") or ""),
            url=str(r.get("link") or ""),
            snippet=str(r.get("snippet") or ""),
        )
        for r in (data.get("organic") or [])
        if r.get("link")
    ]


_SCRIPT_RE = re.compile(r"<(script|style|noscript|svg)[^>]*>.*?</\1>", re.S | re.I)
_BLOCK_RE = re.compile(r"</?(p|div|br|li|tr|h[1-6]|section|article|p)[^>]*>", re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"[ \t\u3000]+")
_BLANK_RE = re.compile(r"\n{3,}")

#: `<meta charset=...>` and `<meta http-equiv=content-type content="...charset=...">`.
#: Official gazette sites are predominantly GB2312/GBK and often send **no**
#: charset in the response header, so httpx falls back to UTF-8. That is not a
#: cosmetic problem: a wrongly decoded page still yields 12000 characters of
#: mojibake, and the summariser then has ~100 usable characters to work with --
#: which is exactly how a detailed 12000-character article became a 121-character
#: entry. The bytes are the only reliable source of the encoding here.
_META_CHARSET_RE = re.compile(
    rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_\-]+)""", re.I
)

#: Encodings that cannot be decoded from bytes alone: they carry no BOM, are
#: mislabelled as a legacy GB name, or are a superset we must prefer. Anything
#: GB2312/GBK bytes actually contain decodes correctly as gb18030; decoding as
#: gb2312 first raises on the slightly newer characters those pages do use.
_CHARSET_ALIASES = {
    "gb2312": "gb18030",
    "gbk": "gb18030",
    "gb_2312-80": "gb18030",
    "x-gbk": "gb18030",
    "cp936": "gb18030",
    "big5": "big5hkscs",
    "utf8": "utf-8",
}


def _declared_charset(content: bytes) -> str:
    """The charset the document declares about itself, if it declares one."""
    match = _META_CHARSET_RE.search(content[:4096])
    if not match:
        return ""
    name = match.group(1).decode("ascii", "ignore").strip().lower()
    return _CHARSET_ALIASES.get(name, name)


def decode_html(content: bytes, *, header_charset: str = "") -> str:
    """Decode page bytes using the charset that actually applies.

    Order is deliberate: an explicit ``charset`` in the response header wins (the
    server is stating it authoritatively), then the in-document ``<meta>``, then
    UTF-8, then GB18030. UTF-8 is tried before GB18030 rather than trusting the
    header alone, because a wrong-but-plausible header must not turn a perfectly
    readable UTF-8 page into mojibake either — a strict decode either succeeds or
    is discarded, so the guess is never silently half-applied.
    """
    candidates: list[str] = []
    for name in (header_charset, _declared_charset(content)):
        if not name:
            continue
        resolved = _CHARSET_ALIASES.get(name.strip().lower(), name.strip().lower())
        if resolved and resolved not in candidates:
            candidates.append(resolved)
    for name in ("utf-8", "gb18030"):
        if name not in candidates:
            candidates.append(name)
    for name in candidates:
        try:
            return content.decode(name)
        except (UnicodeDecodeError, LookupError):
            continue
    # Last resort: never lose the page, but make the damage impossible to miss.
    return content.decode("utf-8", "replace")


_A_BODY_RE = re.compile(r"<a\b[^>]*>(.*?)</a>", re.S | re.I)
_TITLE_TEXT_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)

#: 站点把失效文章统一跳到这类地址上。它不是越权，是链接没了。
_ERROR_PAGE_RE = re.compile(
    r"/(?:404|410|error|errors|not[-_]?found|err)(?:[./_-]|$)", re.I
)


def _looks_like_error_page(url: str) -> bool:
    try:
        path = urlsplit(url).path or ""
    except ValueError:
        return False
    return bool(_ERROR_PAGE_RE.search(path))


def link_share(html: str) -> float:
    """Fraction of the page's visible characters that sit inside <a> elements.

    This is what separates an article from an index. A story page links a few
    related items and measures under 0.1; a section landing page, a site
    homepage or a 「投稿指南」 is almost entirely a list of links and measures
    over 0.5. The distinction matters because both are on whitelisted domains:
    authority says *where* the text comes from, not whether it is a document at
    all. Measured on the pages that produced the bad entry — article 0.003,
    homepage 0.80, 历次党代会 index 0.61.
    """
    source = _SCRIPT_RE.sub(" ", html or "")
    plain = len(_SPACE_RE.sub("", _TAG_RE.sub("", source)))
    if not plain:
        return 1.0
    link_text = sum(
        len(_SPACE_RE.sub("", _TAG_RE.sub("", match.group(1))))
        for match in _A_BODY_RE.finditer(source)
    )
    return min(1.0, link_text / plain)


def html_to_text(html: str) -> str:
    """Strip a page down to readable text.

    Not a readability algorithm: government pages keep their body in predictable
    blocks, and a heavyweight extractor would add a dependency for marginal gain.
    What matters here is that navigation and script noise is gone, because the
    summariser charges for every character it reads.
    """
    text = _SCRIPT_RE.sub(" ", html or "")
    text = _BLOCK_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = unescape(text)
    text = _SPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _BLANK_RE.sub("\n\n", text)
    return text.strip()[:MAX_PAGE_CHARS]


def fetch_candidates(url: str) -> list[str]:
    """Safe URL variants to try, in preference order.

    Old search indexes still return ``http://`` links for sites that have since
    moved to HTTPS.  Browsers silently upgrade many of them; ``httpx`` does not,
    so a perfectly usable page looked unreachable to the studio.  The host and
    path never change here — authority checking remains the caller's job and an
    HTTP result can never redirect this helper to an unrelated search result.
    """
    value = (url or "").strip()
    try:
        parts = urlsplit(value)
    except ValueError:
        return [value] if value else []
    out: list[str] = []
    if parts.scheme.lower() == "http":
        out.append(urlunsplit(("https", parts.netloc, parts.path, parts.query, parts.fragment)))
    if value and value not in out:
        out.append(value)
    return out


def is_navigation_url(url: str) -> bool:
    """True for a site root / section landing page, judged by the URL alone.

    `link_share` catches an index by its content, but a homepage can look sparse
    to a naive extractor and slip through with a few hundred characters of
    navigation. The URL is the cheaper and more honest signal: a root path or a
    bare `index.html` is a directory listing, never a single document.
    """
    try:
        parts = urlsplit((url or "").strip())
    except ValueError:
        return False
    path = (parts.path or "/").rstrip("/")
    if not path:
        return True
    last = path.rsplit("/", 1)[-1].lower()
    return last in ("index.html", "index.htm", "index.shtml", "default.html", "default.htm")


_NON_ARTICLE_TITLE_RE = re.compile(
    r"(投稿指南|投稿须知|征稿启事|版权声明|网站地图|联系我们|关于我们|"
    r"办公人导航|广告服务|人才招聘|订阅指南)"
)


def looks_like_index(title: str, html: str) -> bool:
    """True when a fetched page is a directory, not a source document.

    Three independent signals, any one of which is enough. They were taken from
    the pages that actually polluted one entry: the site homepage, a
    「党的文献杂志投稿指南」, and an 「办公人导航」 aggregator page — all of them
    on a whitelisted institution's domain, so the domain check had nothing to say
    about them.
    """
    if _NON_ARTICLE_TITLE_RE.search(title or ""):
        return True
    return link_share(html) >= 0.45


async def fetch_text(url: str) -> str:
    """Fetch one page and return its text, with bounded connection fallbacks.

    Search and page download take different network paths: model-hosted search can
    work while the local machine's proxy (or a stale HTTP URL) cannot.  Try HTTPS
    first for old HTTP links, then retry connection-level failures without proxy
    environment variables.  HTTP status errors and anti-bot pages are not retried
    through another route because doing so would only delay the same answer.
    """
    candidates = fetch_candidates(url)
    if not candidates:
        raise FetchError("不是有效链接", kind="invalid_url", url=url)
    if is_navigation_url(url):
        # Refused before the request: a root URL is a listing by construction, and
        # fetching it only buys a page of link text that the summariser will try
        # to treat as source material.
        raise FetchError(
            "该链接是站点首页或栏目列表，不是正文页",
            kind="not_article",
            url=url,
            action="请选择具体文章页的链接",
        )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "zh-CN,zh;q=0.9",
    }
    failures: list[FetchError] = []
    retry_without_env = False

    for trust_env in (True, False):
        if not trust_env and not retry_without_env:
            break
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(FETCH_TIMEOUT, connect=min(8.0, FETCH_TIMEOUT)),
            follow_redirects=True,
            headers=headers,
            trust_env=trust_env,
        ) as client:
            for candidate in candidates:
                try:
                    resp = await client.get(candidate)
                    resp.raise_for_status()
                    # The initial URL is checked by the MCP tool, but redirects
                    # can leave that host.  Keep the authority boundary after the
                    # redirect as well; a whitelisted short-link must not become
                    # a route to an arbitrary page.
                    from .authority import is_authoritative

                    # Checked first, and independently of the allowlist: a dead
                    # article is very often "302 to a site-wide error page"
                    # rather than a clean 404. Reporting that as "left the
                    # allowlist" read like an access violation and sent the user
                    # looking at their network settings; reporting it as an
                    # anti-bot page sends them looking for a proxy. It is simply
                    # a dead link.
                    if _looks_like_error_page(str(resp.url)):
                        raise FetchError(
                            f"链接已失效：被跳转到站点错误页（{resp.url.path}）",
                            kind="not_found",
                            url=candidate,
                            status_code=404,
                            action="自动更换下一条权威来源",
                        )
                    if not is_authoritative(str(resp.url)):
                        raise FetchError(
                            f"网页跳转到了非权威站点（{resp.url.host or '未知域名'}）",
                            kind="redirect_outside_allowlist",
                            url=candidate,
                            action="该页面不能作为权威来源，请更换链接",
                        )
                    ctype = resp.headers.get("content-type", "")
                    if "html" not in ctype and "text" not in ctype:
                        raise FetchError(
                            f"返回内容不是网页正文（{ctype or '未知类型'}）",
                            kind="not_html",
                            url=candidate,
                            action="更换其他权威网页",
                        )
                    # `resp.text` trusts the header's charset and falls back to
                    # UTF-8. Most official gazette sites are GB2312 and send no
                    # charset at all, so decoding must consult the <meta> tag too;
                    # otherwise the page "arrives" as 12000 characters of
                    # mojibake and the summariser can only salvage a line or two.
                    html = decode_html(
                        resp.content,
                        header_charset=resp.charset_encoding or "",
                    )
                    text = html_to_text(html)
                    if len(text) < 120:
                        raise FetchError(
                            "页面正文为空，可能需要 JavaScript 渲染或触发了反爬",
                            kind="empty_or_antibot",
                            url=candidate,
                            action="更换其他权威网页，或使用支持浏览器渲染的抓取服务",
                        )
                    title_match = _TITLE_TEXT_RE.search(html)
                    title = (
                        _SPACE_RE.sub(" ", unescape(title_match.group(1))).strip()
                        if title_match else ""
                    )
                    if looks_like_index(title, html):
                        raise FetchError(
                            "抓到的是栏目列表或索引页，不是正文"
                            + (f"（{title[:24]}）" if title else ""),
                            kind="not_article",
                            url=candidate,
                            action="换一条具体文章页的链接",
                        )
                    return text
                except FetchError as exc:
                    failures.append(exc)
                except httpx.HTTPStatusError as exc:
                    status = int(exc.response.status_code)
                    if status in (404, 410):
                        failures.append(
                            FetchError(
                                f"链接已失效（HTTP {status}）",
                                kind="not_found",
                                url=candidate,
                                status_code=status,
                                action="自动更换下一条权威来源",
                            )
                        )
                    elif status in (401, 403):
                        failures.append(
                            FetchError(
                                f"网站拒绝自动抓取（HTTP {status}）",
                                kind="access_denied",
                                url=candidate,
                                status_code=status,
                                action="更换其他权威网站或使用浏览器抓取服务",
                            )
                        )
                    elif status == 429:
                        failures.append(
                            FetchError(
                                "网站请求频率受限（HTTP 429）",
                                kind="rate_limited",
                                url=candidate,
                                status_code=status,
                                retryable=True,
                                action="稍后重试，当前先更换其他权威来源",
                            )
                        )
                    else:
                        failures.append(
                            FetchError(
                                f"网站返回 HTTP {status}",
                                kind="server_error" if status >= 500 else "http_error",
                                url=candidate,
                                status_code=status,
                                retryable=status >= 500,
                                action="更换其他权威来源" if status < 500 else "稍后重试或更换来源",
                            )
                        )
                except httpx.ProxyError as exc:
                    retry_without_env = True
                    failures.append(
                        FetchError(
                            f"代理连接失败（{type(exc).__name__}）",
                            kind="proxy_error",
                            url=candidate,
                            retryable=True,
                            action="检查 HTTP_PROXY/HTTPS_PROXY，程序已尝试绕过代理重试",
                        )
                    )
                except httpx.ConnectError as exc:
                    retry_without_env = True
                    raw = str(exc).lower()
                    dns = any(x in raw for x in ("getaddrinfo", "name or service", "nodename", "11001"))
                    failures.append(
                        FetchError(
                            "域名解析失败（DNS）" if dns else f"本机无法连接目标网站（{type(exc).__name__}）",
                            kind="dns_error" if dns else "connection_error",
                            url=candidate,
                            retryable=True,
                            action=(
                                "检查 DNS 设置和网络连接"
                                if dns
                                else "检查网络出口、防火墙或代理设置；程序已尝试直连重试"
                            ),
                        )
                    )
                except httpx.TimeoutException as exc:
                    retry_without_env = True
                    failures.append(
                        FetchError(
                            f"连接或读取超时（{type(exc).__name__}）",
                            kind="timeout",
                            url=candidate,
                            retryable=True,
                            action="当前先更换其他权威来源，并检查网络质量",
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - report every attempted route
                    failures.append(
                        FetchError(
                            f"抓取器内部错误：{type(exc).__name__}: {str(exc)[:100]}",
                            kind="fetcher_error",
                            url=candidate,
                            action="检查后端抓取器日志",
                        )
                    )

    if not failures:
        raise FetchError("未知抓取错误", kind="fetcher_error", url=url)

    kinds = {f.kind for f in failures}
    if kinds == {"not_article"}:
        raise FetchError(
            "抓到的是首页、栏目列表或索引页，不是正文页",
            kind="not_article",
            url=url,
            action="换一条具体文章页的链接",
        )
    # Both HTTP and HTTPS variants commonly return the same dead-link response;
    # report the cause once, not as two unrelated failures.
    if kinds == {"not_found"}:
        status = next((f.status_code for f in failures if f.status_code), 404)
        raise FetchError(
            f"链接已失效（HTTP {status}）；已尝试 HTTP/HTTPS，需更换其他权威来源",
            kind="not_found",
            url=url,
            status_code=status,
            action="自动更换下一条权威来源",
        )

    priority = (
        "dns_error", "proxy_error", "connection_error", "timeout",
        "access_denied", "rate_limited", "empty_or_antibot", "not_article",
        "redirect_outside_allowlist", "server_error", "http_error", "not_html",
        "fetcher_error", "not_found",
    )
    primary = next((kind for kind in priority if kind in kinds), failures[-1].kind)
    chosen = next(f for f in reversed(failures) if f.kind == primary)
    counts = {kind: sum(1 for f in failures if f.kind == kind) for kind in kinds}
    detail = "、".join(f"{kind}×{count}" for kind, count in sorted(counts.items()))
    raise FetchError(
        f"{chosen}（尝试结果：{detail}）",
        kind=primary if len(kinds) == 1 else "multiple_failures",
        url=url,
        status_code=chosen.status_code,
        retryable=any(f.retryable for f in failures),
        action=chosen.action,
    )
