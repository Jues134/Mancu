"""判断用户这句话是「一段材料」还是「一个题目」。

两种输入走的路不一样，而用错路的代价不对称：

- 粘贴长文 → 当作正文直接改编。如果误判成题目去检索，会拿一段别人的概述覆盖掉
  用户自己的材料，而且用户看不出被换了。
- 说一句「帮我做个关于党的诞生的视频，1 分钟」→ 必须先去知识库找材料。如果误判
  成正文，分镜会拿这一句话去拆镜头，产出六个内容重复的空镜。

所以判定用**长度**做第一道闸，且倾向于「不是题目」：只有明确的短请求才走检索。
`长度 ≥ 160 字` 或含多段换行，直接当正文——这条边界比任何关键词都稳，因为题目式
请求天然很短。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Longer than this and it is material, not a request. A one-line request that
#: long would have to be a paragraph about what the video should contain — at
#: which point it *is* material, and researching over it would discard the user's
#: own words.
MAX_REQUEST_CHARS = 160

#: Words that mark a request rather than content. At least one must appear.
REQUEST_MARKERS = (
    "视频", "短片", "纪录片", "片子", "成片", "宣传片", "微纪录",
    "做一个", "做一期", "做条", "生成", "帮我做", "剪一个", "拍一个", "出一条",
    # 「介绍一下遵义会议」是一句请求，不是一段材料。缺了这两个词，它既进不了
    # 主题路径（没有请求标记），又因为以动词开头被排除在裸主题之外，最后只能
    # 判成「信息不足」——而它明明有一个能去检索的名字。
    "介绍一下", "介绍", "讲讲", "聊聊", "说说",
)

#: Phrases stripped before the topic is taken from the request.
BOILERPLATE = (
    "帮我", "帮忙", "请", "麻烦", "做一个", "做一期", "做条", "生成一个", "生成",
    "剪一个", "拍一个", "出一条", "关于", "有关", "视频", "短片", "纪录片", "片子",
    "宣传片", "微纪录", "时长", "大概", "大约", "左右", "要求", "需要", "想要",
    "一下", "一个", "一期", "的",
    # 「做一个习近平是谁的人物介绍视频」这类句式。不加这几个词，剥完剩下的是
    # 「习近平是谁 人物介绍」，拿它去检索会多出两段没有信息量的字。
    # 长的写在前面：「人物介绍」必须先于「介绍」被剥掉，否则只剩一个「人物」。
    "人物介绍", "人物传记", "是谁", "简介", "生平", "介绍",
)

_CN_DIGITS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

_SECONDS_RE = re.compile(r"(\d+(?:\.\d+)?|十?[一二三四五六七八九十两])\s*(秒|分钟|分)")

#: 「关于X的（视频|短片|…）」是最常见的说法，先按它取，避免把「的」当成边界。
#: 旧写法 `关于(.+?)的` 在「关于党的诞生的视频」上只能取到「党」——惰性匹配遇到
#: 第一个「的」就停了，于是主题被截成一个字，判定为无效请求，整条路走不到知识库。
_TOPIC_RE = re.compile(
    r"(?:关于|有关|讲讲|介绍一下|介绍)"
    r"(.+?)"
    r"(?:的)?"
    r"(?:视频|短片|纪录片|片子|成片|宣传片|微纪录|，|,|。|$)"
)

#: 只有这些字的"主题"其实还是请求本身（「做个视频」），不是主题。
_REQUEST_ONLY = set("做个来生成剪拍一条一期好看的的了短片视频记录宣传微")


@dataclass
class TopicRequest:
    topic: str
    seconds: int = 0        # 0 = 用户没提时长，由管线按材料体量决定
    raw: str = ""


def _cn_number(token: str) -> float | None:
    if token.isdigit() or re.fullmatch(r"\d+(?:\.\d+)?", token):
        return float(token)
    if token in _CN_DIGITS:
        return float(_CN_DIGITS[token])
    return None


def parse_duration(text: str) -> int:
    """Seconds from「1 分钟」「90 秒」「两分钟」; 0 when absent or implausible."""
    match = _SECONDS_RE.search(text or "")
    if not match:
        return 0
    value = _cn_number(match.group(1))
    if value is None:
        return 0
    seconds = value * 60 if match.group(2) in ("分钟", "分") else value
    # The API accepts 15–300s. A request outside that range is more likely a
    # misread number than an intent, so it is ignored rather than clamped.
    if 15 <= seconds <= 300:
        return int(round(seconds))
    return 0


def _strip_topic(text: str) -> str:
    """剥掉请求用语，剩下的就是主题。"""
    candidate = text or ""
    for phrase in BOILERPLATE:
        candidate = candidate.replace(phrase, " ")
    candidate = _SECONDS_RE.sub(" ", candidate)
    candidate = re.sub(r"[，。、；：！？,.;:!?\s]+", " ", candidate).strip()
    return candidate[:40]


def _extract_topic(text: str) -> str:
    """The subject of the request, e.g. 「党的诞生」.

    Two paths, and which one ran matters: when「关于X的」matched, the captured
    subject is used **verbatim** — running the boilerplate stripper over it turned
    「党的诞生」 into 「党 诞生」, because 「的」 is itself a filler word. That
    damaged the query the knowledge base searches with, which is the one place a
    single character changes whether an entry is found.

    The verbatim path yields to the stripping one when it captures nothing:
    「人物介绍视频」 contains 「介绍」, so the regex happily matches it and captures
    an empty subject between 「介绍」 and 「视频」. Taking that at face value made
    the whole request read as "no topic" — 「做一个习近平是谁的人物介绍视频」 was
    classified as having nothing to search for, and the run stopped instead of
    looking the name up.
    """
    match = _TOPIC_RE.search(text or "")
    if match:
        candidate = match.group(1).strip()
        if candidate.endswith("的"):
            candidate = candidate[:-1]
        # 「介绍」后面的 `(.+?)` 可能只捞到一个结尾名词（「人物介绍视频」里那一版
        # 取到的是「视频」）。捕到的东西如果全是请求用语，等于什么都没捕到。
        if len(candidate) >= 2 and set(candidate) - _REQUEST_ONLY:
            return candidate[:40]
    return _strip_topic(text)


def parse_request(text: str) -> TopicRequest | None:
    """Return the topic when `text` is a request, else None (it is material)."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    if len(cleaned) >= MAX_REQUEST_CHARS:
        return None
    # Multiple paragraphs means somebody pasted source material with a request
    # wrapped around it; the material is the part that matters.
    if cleaned.count("\n") >= 2:
        return None
    if not any(marker in cleaned for marker in REQUEST_MARKERS):
        return None
    topic = _extract_topic(cleaned)
    if len(topic) < 2:
        return None
    # 「做个视频」「生成一个」这类没有主题的请求，剥掉请求用语后只剩动词。
    if not set(topic) - _REQUEST_ONLY:
        return None
    # A request names a subject; "做一个视频" alone does not, and searching the
    # knowledge base for it would return whatever happens to be stored.
    return TopicRequest(topic=topic, seconds=parse_duration(cleaned), raw=cleaned)
