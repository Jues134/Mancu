"""哪些网站算「权威」——这份白名单就是知识库的可信度边界。

这条线画在哪，决定这个库能拿来做什么。收进来的是官方与学术机构：政府网、
党媒、国家图书馆、社科文献中心。**不收**百科、聚合站、自媒体、境外站点。
理由不是学术洁癖，而是这些内容要进视频脚本，片子里出现一句错的年份或人名，
观众没有能力当场判断，事后也难纠正。

同样重要的是「查不到就承认查不到」：宁可让这一条检索失败并告诉用户手动补，
也不要为了让流程走下去拿一个百科条目充数。`is_authoritative` 返回 False 时，
调用方必须丢弃，不能降级使用。
"""

from __future__ import annotations

import re
from typing import Any, Iterable
from urllib.parse import urlparse

#: (域名后缀, 站点名, 级别). 1 = 官方/党媒, 2 = 学术与文献机构.
#: The suffix match is on a dot boundary, so `evil-gov.cn` does not pass.
DEFAULT_ALLOWED_SITES: tuple[tuple[str, str, int], ...] = (
    ("gov.cn", "中国政府网", 1),
    ("people.com.cn", "人民网", 1),
    # 人民网的正式主域名。旧文章页现在会 302 到 www.people.cn（失效文章跳到
    # /404/error.html），而白名单只有 people.com.cn——于是每一次「人民网正文
    # 抓取」都以 redirect_outside_allowlist 失败，错误信息还写着"跳转到非权威
    # 站点"，读起来像越权，实际只是域名写法不同。两个域指向同一家机构。
    ("people.cn", "人民网", 1),
    ("news.cn", "新华网", 1),
    ("xinhuanet.com", "新华网", 1),
    ("qstheory.cn", "求是网", 1),
    ("12371.cn", "共产党员网", 1),
    ("cctv.com", "央视网", 1),
    ("mod.gov.cn", "国防部网", 1),
    # 中共中央党史和文献研究院：官修党史的源头。上一次实测它就排在结果第一位，
    # 却因为不在表里被丢掉——白名单漏掉最该有的那一家，比白名单太严更常见。
    ("dswxyjy.org.cn", "中共中央党史和文献研究院", 1),
    # 官方纪念馆与地方党史网。只收省级以上党史部门与国家级纪念馆，它们讲的是
    # 本馆/本地档案里的一手内容。
    ("zjds.cn", "浙江党史网", 2),
    ("zgyd1921.com", "中共一大纪念馆", 2),
    ("nanhu1921.com", "南湖革命纪念馆", 2),
    ("cssn.cn", "中国社会科学网", 2),
    ("nlc.cn", "中国国家图书馆", 2),
    ("nssd.cn", "国家哲学社会科学文献中心", 2),
    ("cnki.net", "中国知网", 2),
    ("sinoss.net", "中国高校人文社会科学信息网", 2),
)

# Backwards-compatible name for code/tests that import the built-in catalogue.
# Runtime checks deliberately use ``allowed_sites()`` so a user's saved changes
# take effect without restarting the backend.
ALLOWED_SITES = DEFAULT_ALLOWED_SITES

_DOMAIN_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$",
    re.IGNORECASE,
)


def normalize_domain(value: str) -> str:
    """Turn a domain or pasted URL into the hostname stored in the allowlist."""
    raw = (value or "").strip().lower()
    if not raw:
        raise ValueError("来源域名不能为空")
    if "://" not in raw:
        raw = "https://" + raw.lstrip("*.")
    try:
        host = (urlparse(raw).hostname or "").rstrip(".").lower()
    except ValueError as exc:
        raise ValueError("来源域名格式不正确") from exc
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("来源域名格式不正确") from exc
    if not _DOMAIN_RE.fullmatch(host):
        raise ValueError("来源域名格式不正确，请填写 example.gov.cn 或完整网址")
    return host


def normalize_sites(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate, normalize and de-duplicate a user supplied allowlist."""
    sites: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        domain = normalize_domain(str(item.get("domain") or ""))
        if domain in seen:
            continue
        seen.add(domain)
        label = str(item.get("label") or domain).strip()[:80] or domain
        try:
            tier = int(item.get("tier") or 2)
        except (TypeError, ValueError):
            tier = 2
        sites.append({"domain": domain, "label": label, "tier": 1 if tier == 1 else 2})
    return sites


def allowed_sites() -> tuple[tuple[str, str, int], ...]:
    """Return the effective allowlist, including persisted user changes."""
    try:
        # Lazy import avoids a module cycle: page fetching imports this module too.
        from . import websearch

        configured = websearch.load_config().get("allowed_sites")
    except (OSError, ValueError, TypeError):
        configured = None
    if configured is None:
        return DEFAULT_ALLOWED_SITES
    if not isinstance(configured, list):
        return DEFAULT_ALLOWED_SITES
    try:
        normalized = normalize_sites(configured)
    except ValueError:
        # A partially-written/manual config must not silently broaden the trust
        # boundary. Fall back to the reviewed built-ins instead.
        return DEFAULT_ALLOWED_SITES
    return tuple((site["domain"], site["label"], site["tier"]) for site in normalized)


def host_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def is_authoritative(url: str) -> bool:
    """True when the URL belongs to a whitelisted institution.

    Dot-boundary check: `cpc.people.com.cn` matches `people.com.cn`, while a
    lookalike such as `people.com.cn.evil.net` does not — the suffix must be the
    host's end, not a middle fragment.
    """
    host = host_of(url)
    if not host:
        return False
    for suffix, _label, _tier in allowed_sites():
        if host == suffix or host.endswith("." + suffix):
            return True
    return False


def site_label(url: str) -> str:
    """The institution's name for the citation line, e.g. 人民网."""
    host = host_of(url)
    if not host:
        return "未知来源"
    matched = ""
    label = host
    for suffix, name, _tier in allowed_sites():
        # Longest suffix wins: `cpc.people.com.cn` is 中国共产党新闻网 only if that
        # were listed; otherwise it falls back to 人民网 rather than to something
        # arbitrary.
        if (host == suffix or host.endswith("." + suffix)) and len(suffix) > len(matched):
            matched, label = suffix, name
    return label


def describe_allowed() -> str:
    labels: list[str] = []
    sites = allowed_sites()
    for _suffix, label, _tier in sites:
        if label not in labels:
            labels.append(label)
    if not labels:
        return "当前未配置任何允许来源"
    return "、".join(labels[:6]) + f" 等 {len(labels)} 个机构"
