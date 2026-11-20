"""把抓到的网页正文整理成一条结构化的知识条目。

输出的格式是固定的，因为界面上要按「一、知识正文 / 二、关键知识点 / 三、来源与
参考资料」三段渲染，也因为这四块正好对应四种用途：

    正文      → 改编写脚本时的材料
    关键知识点 → 分镜时的锚点，也是审查时逐条核对的对象
    分类·年份  → 检索与去重的依据
    来源      → 出了错能追到是哪份文件

一条硬规则贯穿全文：**只用材料里出现过的史实**。摘要模型最容易犯的错是"补常识"——
材料写"五四运动之后马克思主义传播扩大"，它顺手补上"1919年5月4日北京学生游行"，
听起来对，但材料里没有，于是这条知识有了一个无法追溯来源的句子。宁可正文短，
也不许出现材料外的年份、人名、数字。这条要求写进了提示词，也在 `_audit` 里做
一次机器核对：正文里出现但材料里没有的四位年份会被剔掉。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..tools import llm as llm_tools

SYSTEM = """你是历史题材纪录片的知识库编辑。把给定网页正文整理成一条结构化知识条目，
只输出 JSON，不要解释。

格式：
{
  "title": "条目标题，不超过 20 字",
  "category": "分类，如：历史知识 / 党史 / 近代史",
  "period": "时间范围，如：1919—1921（材料没有就留空）",
  "body": "知识正文，段落之间用空行分隔",
  "key_points": ["可核对的断言，一条一句"],
  "sources": [{"kind": "历史文献 或 学术研究", "org": "来源机构", "title": "原始标题", "url": "链接"}]
}

body 的篇幅不设上下限，完全由材料决定：
1. 材料详实（有背景、过程、人物、时间线、细节、结果）→ **就写详细**。可以分 6~12 段，
   按「背景 → 经过（按时间推进）→ 关键细节 → 结果与影响」铺开。具体的时间、地点、
   人名、数字、场景细节都要留下来，不要压成一句概括——这份正文是后面写脚本、定分镜
   的原料，概括过头等于把肉削掉了。
2. 材料只有概要（一篇短讯、一段简介）→ **就写短**。一段、两三句话也正常。
   有多少写多少，不要为了显得完整去补内容，也不要因为短而加一句"详细内容待补充"。
3. 中间状态最常见：写到材料支撑不住为止。判断标准是「这句话材料里有依据吗」，
   不是「够不够三百字」。
4. 长短都不能注水：同一件事不换三种说法重复；不用形容词和评价凑长度；
   不写「具有重要意义」「影响深远」这类没有信息量的话。

硬规则：
1. 只写材料里出现过的史实。材料没写的年份、人名、数字、评价，一律不许补。
2. 不要写"本文""据记载""众所周知"这类空话，也不要写导语式的开场。
3. key_points 条数随内容：材料多就多列（最多 12 条），材料少 3 条即可。
   每条必须是可核对的断言（时间、人物、事件、因果），不是感想。
4. sources 用给定材料里的标题和链接；没有链接就不写 url 字段。"""


@dataclass
class EntryDraft:
    title: str
    category: str = ""
    period: str = ""
    body: str = ""
    key_points: list[str] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    dropped_years: list[str] = field(default_factory=list)


def _llm_config() -> Any:
    """Which model writes the summary.

    Bound to 内容改编 rather than a new agent id: it is the same job (turn source
    material into a shootable script), and a seventh id would have to be added to
    the settings table for no benefit.
    """
    from .. import models_config

    cfg = models_config.load()
    for agent_id in ("script_adaptor", "director", "storyboard"):
        profile = cfg.provider_for(agent_id)
        if profile is not None:
            return models_config.provider_to_llm_config(profile)
    raise RuntimeError("没有可用于摘要的文本模型，请先在设置里配置模型供应商")


_YEAR_RE = re.compile(r"(1[89]\d{2}|20\d{2})\s*年")

#: 材料里常写「一九二一年」，正文若转成阿拉伯数字，旧核对会**把正确的句子删掉**
#: ——它查的是"材料里没出现过 1921"，而材料里是汉字。正文写详细之后年份出现得更
#: 多，这个误判的代价也就更大，所以两种写法都算出现过。
_CN_DIGITS = {
    "〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_YEAR_RE = re.compile(r"(?<![一二三四五六七八九〇零])([一二三四五六七八九〇零]{4})\s*年")


def _years_in(text: str) -> set[str]:
    """Every four-digit year in `text`, in either writing system."""
    found = set(_YEAR_RE.findall(text))
    for group in _CN_YEAR_RE.findall(text):
        value = "".join(str(_CN_DIGITS[ch]) for ch in group)
        if value[:2] in ("18", "19", "20"):
            found.add(value)
    return found


def _audit(draft: EntryDraft, material: str) -> EntryDraft:
    """Drop sentences carrying a year the material never mentions.

    A summary that adds "1919年5月4日" is not more useful than one that says
    "五四运动之后" — it is less trustworthy, because the reader cannot tell which
    facts came from the source. Removing the sentence is crude but honest, and the
    dropped years are reported so the reviewer sees what was removed.
    """
    allowed = _years_in(material)
    dropped: list[str] = []

    def keep(sentence: str) -> bool:
        missing = [y for y in _years_in(sentence) if y not in allowed]
        if missing:
            dropped.extend(missing)
            return False
        return True

    paragraphs = []
    for para in re.split(r"\n+", draft.body or ""):
        sentences = re.split(r"(?<=[。！？])", para)
        kept = [s for s in sentences if s.strip() and keep(s)]
        if kept:
            paragraphs.append("".join(kept).strip())
    draft.body = "\n\n".join(paragraphs).strip()
    draft.key_points = [p for p in draft.key_points if keep(str(p))]
    draft.dropped_years = sorted(set(dropped))
    return draft


async def summarize(query: str, pages: list[str]) -> EntryDraft:
    """Turn fetched pages into one entry draft."""
    material = "\n\n".join(pages)[:20000]
    user = (
        f"用户想做的主题：{query}\n\n"
        f"以下是权威网站正文（{len(pages)} 篇，可能相互重叠，请合并去重）：\n\n"
        f"{material}"
    )
    raw = await llm_tools.chat_json(
        _llm_config(),
        system=SYSTEM,
        user=user,
        # A 2000-token cap is what actually cut long entries short: a detailed
        # body plus 8 points plus sources runs past it, and a truncated JSON is
        # either repaired into something shorter or thrown away. The ceiling has
        # to clear the *longest* legitimate answer, not the typical one.
        max_tokens=6000,
    )
    data = raw if isinstance(raw, dict) else {}
    draft = EntryDraft(
        title=str(data.get("title") or query)[:40],
        category=str(data.get("category") or "历史知识")[:16],
        period=str(data.get("period") or "")[:24],
        body=str(data.get("body") or "").strip(),
        key_points=[str(p)[:160] for p in (data.get("key_points") or []) if str(p).strip()][:12],
        sources=[
            {
                "kind": str(s.get("kind") or "参考资料")[:8],
                "org": str(s.get("org") or "")[:40],
                "title": str(s.get("title") or "")[:80],
                "url": str(s.get("url") or "")[:300],
            }
            for s in (data.get("sources") or [])
            if isinstance(s, dict)
        ][:4],
    )
    if not draft.body:
        raise ValueError("摘要模型没有产出正文")
    return _audit(draft, material)
