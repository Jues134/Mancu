"""知识库的语义索引：把「一个条目」变成「若干可召回的段落」。

为什么不复用 `platform/rag` 的 `studio_chunks`：

  · 那张表挂 `run_id` 且 `ON DELETE CASCADE`，生命周期是「一次运行」。知识库
    是跨运行持久的，一条 entry 会被几十次运行复用。硬塞进去要么造假 run（删它
    会级联清掉全部知识向量），要么改列约束。所以独立一张 `studio_kb_chunks`。
  · 两者的内容形状也不同。rag 的 chunk 是「原文的一段」；知识库条目除了正文，
    还有 `key_points`——那些是摘要，信息密度比正文高，必须单独成块，否则一个
    三百字段落会把一条结论稀释掉。

索引范围是 `approved` + `pending`，排除 `rejected`：

  这是刻意和 `store.search()` 保持一致。`pending` 是「本次运行查到、还没经人
  确认」的材料，运行内部本来就能用；把它排除在索引外，会让语义检索在首次运行
  时完全失效。安全边界不靠「不索引」来保证，而靠**召回之后再走一遍同样的闸门**
  （权威来源 + 草稿质量 + 状态），也就是 `kb/__init__.py` 里那三道判定。把审核
  态在索引层过滤掉，等于用一层更粗的规则替代了原来的细规则。

关于嵌入时拼标题：

  正文段落经常不提主题名（「他在这里作出了决定」），单看这一段无法判断它属于
  哪件事。所以送进嵌入接口的文本是 `<标题><段落>`，而库里存的 `text` 保持原样
  ——引用时不该重复标题。这是检索质量与可读性之间唯一需要分开的地方。
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import time
from typing import Any, Iterable

from .. import db, logs
from ..platform.rag.embedder import Embedder, EmbeddingError, SplitConfig, split_block
from . import document

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy ships with the project deps
    np = None  # type: ignore[assignment]


#: 与 rag 的切块参数一致：320 字约含一条可核验的断言，520 是硬上限。
TARGET_CHARS = 320
MAX_CHARS = 520
_SPLIT = SplitConfig(target=TARGET_CHARS, max_chars=MAX_CHARS)

#: 低于这个余弦分数不当作召回（主题卡片）。
#:
#: 0.55 是量出来的，不是拍的。用 embedding-3（2048 维）在本项目的知识库上测：
#:   正例（库里确实有的主题、含换说法）0.61 ~ 0.80
#:   负例（唐朝诗歌、量子纠缠、红烧肉这类库里没有的）0.31 ~ 0.42
#: 两者之间的间隔很小，所以阈值必须落在中间偏高的一端：
#:   · 0.30（rag 用的 0.25 同量级）会把「如何做红烧肉」0.34 全放进来
#:   · 0.55 挡掉全部负例，代价是丢掉个别弱正例（「一国两制」0.445 被拒，
#:     转去联网——它本来就是《邓小平生平》里的一个从属概念，不是独立主题）
#:
#: 顺带解决了主题检索那套加权要防的问题：查「毛主席」时《邓小平生平》得 0.53，
#: 单看分数像命中，其实是「提到过」而不是「讲这件事」。0.55 正好把它挡住。
#:
#: **换嵌入模型必须重测这个值**，不同模型的分数分布不通用。
MIN_SCORE = 0.55

#: 整本资料用更低的阈值，因为它的分数分布本来就低一档。
#:
#: 这不是「书更重要所以放宽」这种含混的理由，是两件具体的事：
#:
#: 1. **匹配粒度不同**。一条卡片代表整个主题，命中它等于命中一个结论，误召回
#:    的代价是「拿错材料写了一整段」。书里的一块只是全书 1/390 的一个段落，
#:    误召回的代价是「多给了一段不相关的上下文」，下游还能靠章节定位澄清。
#:
#: 2. **文本长度不同**。卡片正文是精炼摘要（三百来字），和查询的词汇重叠高；
#:    书正文是叙事段落（两百字里塞进了时间、地点、人物、前因后果），向量被平均
#:    过之后与短查询的相似度天然更低。
#:
#: 在一本 100 章、390 块的书上实测（embedding-3）：
#:   正例（南昌起义 / 遵义会议 / 古田会议 / 皖南事变 / 渡江战役 …共 14 个）
#:       0.528 ~ 0.730，中位 0.625
#:   负例（量子纠缠 / 红烧肉 / 唐朝诗歌 / Python 异步 …共 10 个）
#:       0.298 ~ 0.405
#:   间隔 0.123。取 0.48 时正例 14/14 全过、负例 10/10 全挡。
#:
#: 顺带记一个容易忽略的现象：**库越大，负例的最高分越高**。同样 10 个负例，
#: 库里只有 33 块时最高 0.42，加到 423 块后升到 0.405 但正例最低分也降到了
#: 0.528——所以阈值必须随库规模重标，一次测完用一辈子是错的。
MIN_SCORE_BOOK = 0.48

#: 正文超过这个长度还按卡片结构切，一定是错的。
#:
#: 卡片正文是八百到两千字；超过两万字的正文只可能是整本上传的资料。用它当
#: 判据而不是看 `kind`，是因为 `kind` 恰好可能错——见 `index_entry` 里的说明。
_LONG_DOC_CHARS = 20000

#: 同一本书里最多带几块给下游。
#:
#: 问「1927 年有哪些大事」命中那一章的目录，问「南昌起义怎么打起来的」命中
#: 正文——两者在同一个 entry 里但不是同一块。只返回一块，下游会以为「这本书
#: 就这一点内容」；返回全部 390 块，等于把检索的活儿推给了下游。
BOOK_CHUNKS = 6

#: 目录类块在排序时的折扣。**这是索引，不是内容。**
#:
#: 章节目录块是一串事件名（「1927 年：四一二反革命政变、中共五大召开、……、
#: 南昌起义、……」）。它对**任何**与该章相关的问题都有 0.52~0.63 的相似度：
#: 因为问题里那个事件名就明明白白写在里面。但真问「南昌起义怎么打起来的」时，
#: 它一个字都答不了。
#:
#: 实测这一现象：查「南昌起义」，目录块 0.6252 排第一，真正描述南昌起义的正文
#: 段落只有 0.6210，被挤到后面。查「抗日战争什么时候开始的」，目录块给的是
#: 1940 年百团大战（0.5172），而正确答案在 1937 年七七事变那段的正文里（0.5005）。
#:
#: 这和 `store.py` 里 `BODY_WEIGHT = 0.7` 是同一件事的两个方向：那边压低「正文
#: 里提到过」，这边压低「目录里列举过」。两者都在对付同一个错误——把「出现过
#: 这个词」当成「讲了这件事」。
#:
#: 0.85 / 0.80 是看着上面那组 case 定的，不是标定出来的。它能修好「目录挤掉
#: 正文」，代价是「1927 年有哪些大事」这类**确实该由目录回答**的问题会变得不
#: 确定（目录 0.6045×0.85=0.514，与该章正文最高分 0.5095 几乎持平）。这类问题
#: 要靠第 5 步的查询重写和 RRF 才能稳定解决，现在只能保证「两类块都带回去」。
_PART_WEIGHT = {"chapter": 0.85, "title": 0.80}

#: RRF 的平滑常数，取文献里的常用值。只用排名，因此不需要标定分数尺度。
RRF_K = 60


# ============================================================================
# 摘要与切块
# ============================================================================


def entry_hash(entry: dict[str, Any]) -> str:
    """内容指纹。变了才重新嵌入，没变就跳过——省掉重复的向量化调用。"""
    payload = "\u0000".join(
        [
            str(entry.get("title") or ""),
            str(entry.get("category") or ""),
            str(entry.get("period") or ""),
            str(entry.get("body") or ""),
            "\u0001".join(str(p) for p in (entry.get("key_points") or [])),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _with_title(title: str, text: str) -> str:
    """把标题拼到待嵌入文本前面，但正文里已经以标题开头时不重复。"""
    if not title:
        return text
    if text[: len(title) + 4].find(title) >= 0:
        return text
    return f"《{title}》{text}"


def split_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """把一个条目切成可召回的块。

    两类条目走两条切法，因为它们的形状不同：

    · **card**（AI 摘要出的主题卡片）：标题块负责「这个主题是不是库里这条」，
      摘要块负责「结论是什么」，正文块负责「依据是什么」。三类分开，因为混在
      一起时正文的长度会压过标题和摘要的信号。

    · **book**（整本上传的资料）：先还原「章 > 节 > 段」结构，再在结构内部
      切块。层级本身是检索信号——问「1927 年发生了什么」时该命中那一章，而不是
      某个孤立段落。细节见 `document.py` 的模块注释。

    两者共用同一套下游（向量存储、召回、闸门），所以返回的字段必须一致。
    """
    if str(entry.get("kind") or "card") == "book":
        return _split_book(entry)

    title = str(entry.get("title") or "").strip()
    section = title
    parts: list[dict[str, Any]] = []

    head = " ".join(
        str(x).strip()
        for x in (title, entry.get("category"), entry.get("period"))
        if str(x or "").strip()
    )
    if head:
        parts.append(
            {
                "part": "title",
                "ordinal": 0,
                "text": head,
                "section": section,
                "chapter": "",
                "heading": title,
                "embed_text": head,
            }
        )

    for index, point in enumerate(entry.get("key_points") or []):
        text = str(point or "").strip()
        if len(text) < 4:
            continue
        parts.append(
            {
                "part": "point",
                "ordinal": index,
                "text": text,
                "section": section,
                "chapter": "",
                "heading": "",
                "embed_text": _with_title(title, text),
            }
        )

    body = str(entry.get("body") or "").strip()
    for index, piece in enumerate(split_block(body, _SPLIT)):
        parts.append(
            {
                "part": "body",
                "ordinal": index,
                "text": piece,
                "section": section,
                "chapter": "",
                "heading": "",
                "embed_text": _with_title(title, piece),
            }
        )

    return parts


def _split_book(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """整本书的切块。原文存在 `body` 里，每次重新解析。

    重新解析而不是把切好的块存下来，是刻意的：切块规则会改（目标长度、章节
    正则），改了之后必须能从原文重新生成，否则旧书永远停在旧规则上，而用户
    只会觉得「有的书搜得到、有的搜不到」。解析六万字是毫秒级，这个代价换来的
    是可回溯性，值得。
    """
    raw = str(entry.get("body") or "")
    if not raw.strip():
        return []
    title = str(entry.get("title") or "").strip()
    doc = document.parse(raw, filename=title)
    if title:
        # 库里的标题是权威的：用户可能改过书名，改完索引应当跟着变，
        # 而不是沿用原文里那个。
        doc.title = title
    return document.split_document(doc)


def _looks_degenerate(vectors: list[Any]) -> str:
    """发现退化向量就返回原因，正常则返回空串。

    退化指「全零」或「所有分量相同」。归一化之后这类向量没有方向信息：它与
    **任何**查询的余弦都是同一个固定值，于是排序退化成随机顺序。它不会抛错，
    不会触发降级，健康检查也显示 ready——用户能看到的只有「这本书搜得不准」。

    这个检查是补上一个真实事故的：压测块级增量时用了一个返回常量向量的假嵌入
    器，它把正确的向量覆盖了，而 `chunk_hash` 是匹配的（文本没变），所以增量
    索引从此再也不会修正它。检不出「向量本身是坏的」，是所有增量策略共同的
    盲区——补一维度量比补一条注释有用。
    """
    if not vectors:
        return ""
    for index, vector in enumerate(vectors[:8]):  # 抽查前几条，全查没必要
        arr = _decode(_encode(vector)) if not isinstance(vector, (list, tuple)) else vector
        if np is not None:
            data = np.asarray(arr, dtype=np.float32)
            if data.size == 0:
                return f"第 {index} 条向量为空"
            if not np.any(data):
                return f"第 {index} 条向量全为零"
            if float(np.ptp(data)) == 0.0:
                return f"第 {index} 条向量所有分量相同（常量向量）"
        else:
            seq = list(arr)
            if not seq:
                return f"第 {index} 条向量为空"
            if all(v == 0 for v in seq):
                return f"第 {index} 条向量全为零"
            if min(seq) == max(seq):
                return f"第 {index} 条向量所有分量相同（常量向量）"
    return ""


def _chunk_hash(part: dict[str, Any]) -> str:
    """单个片段的指纹，用于增量索引。

    把参与嵌入的文本和展示用的文本都算进来：只改 `heading` 时 `embed_text`
    也会变，但反过来改 `text` 的标点而 `embed_text` 恰好没变的情况也存在。
    两者都算，才不会漏掉任何一次真实改动。
    """
    payload = "\u0000".join(
        [
            str(part.get("embed_text") or ""),
            str(part.get("text") or ""),
            str(part.get("section") or ""),
            str(part.get("chapter") or ""),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _chunk_id(entry_id: str, part: str, ordinal: int, chapter: str = "") -> str:
    """块 ID。书的块要带上章节，否则每章都从 0 开始编号，ID 会撞。

    用章节标题的短哈希而不是标题原文：中文标题进主键会让 ID 变得很长，而主键
    会被索引引用，短一些整张表都小。哈希够用——这里只需要「同一条目的不同章节
    得到不同 ID」，不需要可逆。
    """
    base = f"{entry_id}:{part}"
    if chapter:
        tag = hashlib.sha1(chapter.encode("utf-8")).hexdigest()[:8]
        base = f"{base}:{tag}"
    return f"{base}:{ordinal:03d}"


#: 向量在库里的表示：float32 的裸字节。
#:
#: JSON 是 27 KB/块（2048 维），BLOB 是 8 KB/块，而且读出来不用解析——1000 块的
#: 检索从 517 ms 降到 3 ms。差别在「每次查询」上重复发生，所以不是微优化。
def _encode(vector: Any) -> bytes:
    if np is not None:
        return np.asarray(vector, dtype=np.float32).tobytes()
    return struct.pack(f"<{len(vector)}f", *vector)


def _decode(value: Any) -> Any:
    """把库里的向量还原成可参与运算的数组。

    同时接受 BLOB 和 JSON 文本：前者是现在的写法，后者是改存储格式之前落下的数据。
    兼容是有意的——为了一个存储格式去把所有条目重新嵌入一遍，代价远高于在这里
    多写四行。

    有 numpy 时直接返回 ndarray 而不转成 list：1000 块 × 2048 维转成 Python 列表
    是 200 万个 float 对象，光是构造它们就要上百毫秒，而下游 `_rank` 本来就是用
    numpy 算的，转过去再转回来纯属浪费。
    """
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if not raw or len(raw) % 4:
            return None
        if np is not None:
            return np.frombuffer(raw, dtype=np.float32)
        return list(struct.unpack(f"<{len(raw) // 4}f", raw))
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "[]":
            return None
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(parsed, list):
            return None
        if np is not None:
            return np.asarray(parsed, dtype=np.float32)
        return [float(x) for x in parsed]
    return None


# ============================================================================
# 写入
# ============================================================================


def _degrade(reason: str, **fields: Any) -> dict[str, Any]:
    """降级返回，同时落一行日志。

    语义检索降级的表现是「搜不到」，而「搜不到」和「库里没有」在界面上长得
    一模一样。把原因同时写进日志，才能在事后回答「是没配模型、模型挂了，还是
    这本书确实没这段」。不抛异常：降级是设计好的行为，不是错误。
    """
    logs.event("vector.degraded", level="warn", reason=reason, **fields)
    return {"ok": False, "degraded": True, "reason": reason}


async def index_entry(
    entry: dict[str, Any],
    *,
    embedder: Embedder | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """把一个条目写入语义索引。幂等，且**按块增量**。

    早期版本是「内容变了就全删全建」。一条卡片只有两三块，无所谓；一本书有
    几百块，改一个标点就要重嵌几百次向量化调用，既慢又费钱，还很容易在限流时
    半途而废（前半本有索引、后半本没有）。

    所以改成逐块比对 `chunk_hash`：只有真正变过的块才调嵌入，没变的块保持
    原样，多余的块删掉。改一个段落 = 一次调用。

    返回 state 字典而不是抛异常，理由和 `RagIndex.index` 一样——语义检索是增强，
    不是前置条件。向量服务不可用时，调用方应当继续按主题检索工作，并且**知道**
    这一层没生效。
    """
    entry_id = str(entry.get("id") or "").strip()
    if not entry_id:
        return {"ok": False, "reason": "条目没有 id"}

    parts = split_entry(entry)
    if not parts:
        return {"ok": False, "reason": "条目没有可索引的正文"}

    # 超长正文被切成 `point` 块，说明它被当成卡片切了。
    #
    # 判断依据是**正文长度**，不是 `kind`——这一点很关键。真实故障里这本书的
    # `kind` 字段本身就是错的（`store.save` 的 UPDATE 分支不动 kind，于是它一直
    # 停在最初写入的 `card`），所以任何「看 kind 是不是 book」的校验都不会触发。
    # 长度不会骗人：卡片正文是八百到两千字，没有哪张卡片该有六万字。
    #
    # 后果值得单独说：card 分支会把 `key_points` 逐条变成独立向量，而书的
    # `key_points` 是一百个章节标题——「1921 年」「1922 年」。这些块几乎没有语义，
    # 却数量众多，会把真正的正文块挤出排序，而**整个过程不报错**。检索「国民党
    # 一大」时排第一的是「1921 年」，最后返回了讲中共一大的卡片，就是这么来的。
    body_chars = len(str(entry.get("body") or ""))
    if body_chars > _LONG_DOC_CHARS and any(part["part"] == "point" for part in parts):
        return {
            "ok": False,
            "reason": (
                f"正文 {body_chars} 字，却被按卡片结构切分（产生 point 块），索引未写入。"
                "整本资料应当按章节切成 chapter / body 块。"
            ),
        }

    for part in parts:
        part["chunk_id"] = _chunk_id(
            entry_id,
            str(part["part"]),
            int(part["ordinal"]),
            str(part.get("chapter") or ""),
        )
        part["chunk_hash"] = _chunk_hash(part)

    digest = entry_hash(entry)
    existing = _existing_chunks(entry_id)

    # 模型换过（维度会变，分数不可比）时必须全部重嵌，哪怕文本一个字没改。
    # 只比 hash 会让旧模型的向量留在库里，而检索时它们会被维度过滤掉——表现
    # 是「书明明在库里，就是搜不到」，且没有任何报错。
    current_model = embedder.model if embedder is not None else ""
    model_changed = bool(
        current_model and existing and any(v["model"] != current_model for v in existing.values())
    )

    # 有块但没向量：上一次建索引时嵌入这一层挂了，只落了文本。这类块必须当成
    # 「待补」而不是「已完成」——否则会永远停在「字面能搜、语义搜不到」的状态，
    # 而且重建索引也修不好它（指纹一致，会被跳过）。
    unembedded = {cid for cid, info in existing.items() if not int(info.get("dim") or 0)}

    if not force and not model_changed and not unembedded and stored_hash(entry_id) == digest:
        return {"ok": True, "skipped": True, "chunks": len(existing)}

    stale = [
        part
        for part in parts
        if force
        or model_changed
        or part["chunk_id"] in unembedded
        or existing.get(part["chunk_id"], {}).get("chunk_hash") != part["chunk_hash"]
    ]
    keep_ids = {part["chunk_id"] for part in parts}
    removed = [cid for cid in existing if cid not in keep_ids]

    if not stale and not removed:
        # 内容变过但切出来一模一样（比如只改了空白），把指纹更新掉即可，
        # 否则每次保存都会重新解析一遍全文。
        _update_entry_hash(entry_id, digest)
        return {"ok": True, "skipped": True, "chunks": len(existing)}

    emb = embedder or Embedder.from_settings()
    if model_changed and current_model and emb is not None and emb.model != current_model:
        stale = list(parts)

    # 嵌入失败不再「什么都不写」。
    #
    # 旧行为是嵌入一失败就 return，切块也一起不落库。而书的**字面检索**读的正是
    # `studio_kb_chunks` 的 text 列（`store._best_chunk`），所以一次嵌入故障会让
    # 整本书在检索体系里彻底消失：语义搜不到，字面也搜不到，界面上看起来就是
    # 「本地资料库没有这本书里的任何内容」。一次网络抖动换来一个空库，代价太大。
    #
    # 现在退化成「只有文本块、没有向量」：字面检索照常工作，语义层跳过 dim=0 的
    # 块（`_load_rows` 按维度过滤），等嵌入恢复了增量补齐（`unembedded`）。
    embed_note = ""
    vectors: list[Any] = []
    if emb is None:
        embed_note = "未配置向量化模型，语义检索不可用"
    elif stale:
        try:
            vectors = await emb.embed([part["embed_text"] for part in stale])
            if len(vectors) != len(stale):
                embed_note = "向量数量与块数量不一致"
                vectors = []
            else:
                # 退化向量（全零 / 常量）会让排序变成噪声，而这件事**不会报错**：
                # 健康检查显示 ready，检索正常返回结果，只是顺序是随机的。宁可整批
                # 拒绝，也不要把坏向量写进去——坏向量一旦落库，增量索引凭文本指纹
                # 永远查不出来。
                degenerate = _looks_degenerate(vectors)
                if degenerate:
                    embed_note = f"嵌入服务返回的向量不可用：{degenerate}"
                    vectors = []
        except EmbeddingError as exc:
            embed_note = f"向量化失败：{str(exc)[:160]}"
            vectors = []

    now = time.time()
    with db.transaction() as conn:
        if removed:
            conn.executemany(
                "DELETE FROM studio_kb_chunks WHERE id = ?", [(cid,) for cid in removed]
            )
        if stale:
            conn.executemany(
                "INSERT INTO studio_kb_chunks"
                " (id, entry_id, part, ordinal, text, section, chapter, heading,"
                "  entry_hash, chunk_hash, vector, dim, model, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET"
                "  text=excluded.text, section=excluded.section, chapter=excluded.chapter,"
                "  heading=excluded.heading, entry_hash=excluded.entry_hash,"
                "  chunk_hash=excluded.chunk_hash, vector=excluded.vector,"
                "  dim=excluded.dim, model=excluded.model, updated_at=excluded.updated_at",
                [
                    (
                        part["chunk_id"],
                        entry_id,
                        part["part"],
                        int(part["ordinal"]),
                        part["text"],
                        part["section"],
                        part.get("chapter") or "",
                        part.get("heading") or "",
                        digest,
                        part["chunk_hash"],
                        # 没有向量就存空 BLOB、dim=0、model=''：这是一行「只有文本」
                        # 的块，字面检索看得见它，语义检索（按 dim 过滤）看不见它。
                        _encode(vectors[index]) if index < len(vectors) else b"",
                        len(vectors[index]) if index < len(vectors) else 0,
                        emb.model if index < len(vectors) and emb is not None else "",
                        now,
                    )
                    for index, part in enumerate(stale)
                ],
            )
        if not stale and not removed:
            conn.execute(
                "UPDATE studio_kb_chunks SET entry_hash = ? WHERE entry_id = ?", (digest, entry_id)
            )
    # 建索引是「收了一本书」之后的第一个动作，也是最容易悄悄降级的一步：文本
    # 落盘了、语义层没落，界面只看得到「检索不到」。这里把两件事分开记。
    if embed_note:
        logs.event(
            "index.degraded", level="warn", entry=entry_id, reason=embed_note,
            chunks=len(parts), embedded=len(vectors),
        )
    else:
        logs.event(
            "index.done", entry=entry_id, chunks=len(parts), embedded=len(vectors),
            removed=len(removed), model=(emb.model if emb is not None else ""),
        )
    return {
        # `ok` 的含义是「语义索引完成了」。嵌入失败时它是 False，但文本块已经
        # 落盘（`text_saved`），字面检索照常能命中——调用方要能分辨这两件事，
        # 否则要么谎报成功，要么让用户以为整本书都没索引。
        "ok": not embed_note,
        "degraded": bool(embed_note),
        "reason": embed_note,
        "text_saved": bool(stale),
        "chunks": len(parts),
        "embedded": len(vectors),
        "removed": len(removed),
        "model": (emb.model if emb is not None else "") if not embed_note else "",
        "dim": len(vectors[0]) if vectors else _dominant_dim(),
    }


def _existing_chunks(entry_id: str) -> dict[str, dict[str, Any]]:
    conn = db.connect()
    rows = conn.execute(
        "SELECT id, chunk_hash, model, dim FROM studio_kb_chunks WHERE entry_id = ?",
        (str(entry_id),),
    ).fetchall()
    return {
        str(row["id"]): {
            "chunk_hash": str(row["chunk_hash"] or ""),
            "model": str(row["model"] or ""),
            "dim": int(row["dim"] or 0),
        }
        for row in rows
    }


def _update_entry_hash(entry_id: str, digest: str) -> None:
    with db.transaction() as conn:
        conn.execute(
            "UPDATE studio_kb_chunks SET entry_hash = ? WHERE entry_id = ?", (digest, entry_id)
        )


def remove_entry(entry_id: str) -> int:
    """删掉一个条目的全部向量。条目被弃用或删除时调用。"""
    with db.transaction() as conn:
        cursor = conn.execute("DELETE FROM studio_kb_chunks WHERE entry_id = ?", (str(entry_id),))
        return int(cursor.rowcount or 0)


def compact_vectors() -> dict[str, Any]:
    """把历史上存成 JSON 文本的向量就地改写成 BLOB。

    不调用嵌入接口：这是一次存储格式迁移，不是重建语义。为换格式去重新付一次
    向量化费用，既慢又多花钱，还会在限流时半途而废。
    """
    conn = db.connect()
    rows = conn.execute(
        "SELECT id, vector, dim FROM studio_kb_chunks WHERE typeof(vector) = 'text'"
    ).fetchall()
    converted = 0
    skipped = 0
    with db.transaction() as tx:
        for row in rows:
            vector = _decode(row["vector"])
            if vector is None or len(vector) == 0:
                skipped += 1
                continue
            tx.execute(
                "UPDATE studio_kb_chunks SET vector = ?, dim = ? WHERE id = ?",
                (_encode(vector), len(vector), row["id"]),
            )
            converted += 1
    return {"converted": converted, "skipped": skipped}


async def reindex_all(
    entries: Iterable[dict[str, Any]],
    *,
    embedder: Embedder | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """批量重建。逐条进行，单条失败不中断——一条坏条目不该让整个回填停住。"""
    # 先做一次零成本的格式迁移：老库里存的是 JSON 文本，改写成 BLOB 不需要
    # 重新嵌入。放在嵌入器检查之前，是因为这件事和有没有模型无关——没有模型时
    # 也该把已有数据收拾成正确格式。
    compacted = compact_vectors()

    emb = embedder or Embedder.from_settings()
    if emb is None:
        return {
            "ok": False,
            "degraded": True,
            "reason": "未配置向量化模型，无法重建索引",
            "compacted": compacted.get("converted", 0),
        }

    indexed = skipped = failed = chunks = 0
    failures: list[dict[str, str]] = []
    for entry in entries:
        result = await index_entry(entry, embedder=emb, force=force)
        if result.get("skipped"):
            skipped += 1
            continue
        if not result.get("ok"):
            failed += 1
            if len(failures) < 10:
                failures.append(
                    {"id": str(entry.get("id") or ""), "reason": str(result.get("reason") or "")}
                )
            continue
        indexed += 1
        chunks += int(result.get("chunks") or 0)
    return {
        "ok": True,
        "indexed": indexed,
        "skipped": skipped,
        "failed": failed,
        "chunks": chunks,
        "model": emb.model,
        "compacted": compacted.get("converted", 0),
        "failures": failures,
    }


# ============================================================================
# 读取
# ============================================================================


async def _embed_query(embedder: Any, term: str) -> list[float]:
    """取查询向量，兼容只实现了 `embed()` 的嵌入器。

    内置模型需要给查询加指令前缀，云端那家加不加都行，测试里的假嵌入器只有一个
    `embed()`。三个来源共用一个入口，调用方不必各自判断。
    """
    method = getattr(embedder, "embed_query", None)
    if callable(method):
        return await method(term)
    vectors = await embedder.embed([term])
    return vectors[0] if vectors else []


async def search(
    query: str,
    *,
    limit: int = 3,
    min_score: float | None = None,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """按语义召回条目。同一本书命中多块时，把那几块一并带回来。

    `min_score` 传 None 时按条目类型自动选阈值（卡片 0.55 / 书 0.48，理由见
    `MIN_SCORE_BOOK`）；显式传值则统一用它，测试和调参时方便。

    聚合到条目而不是只返回段落，是因为下游（`research()`）问的是「哪条知识」，
    段落只用来解释「凭哪一句命中」。但只返回段落会让调用方自己去重，而每个调用
    方都会把去重写错一次。
    """
    term = (query or "").strip()
    if not term:
        return {"ok": False, "hits": [], "reason": "检索词为空"}

    emb = embedder or Embedder.from_settings()
    if emb is None:
        return {**_degrade("未配置向量化模型，语义检索不可用", query=term), "hits": []}

    try:
        # 查询单列一个入口：内置模型要给查询加 bge 的指令前缀，而正文不加。
        # 调用方不必知道背后是哪一家模型。
        query_vector = await _embed_query(emb, term)
    except EmbeddingError as exc:
        return {**_degrade(f"向量化失败：{str(exc)[:160]}", query=term), "hits": []}
    if not query_vector:
        return {**_degrade("查询向量为空", query=term), "hits": []}

    dim = len(query_vector)
    # 只读维度匹配的行。放在 SQL 里过滤，而不是全读出来再筛：换过模型之后库里
    # 可能同时存在两种维度的向量，把不匹配的那些也反序列化一遍纯属浪费——而
    # 这个浪费正好发生在「收了一本书之后」最不该慢的时候。
    rows = _load_rows(dim)
    if not rows:
        if _total_chunks() == 0:
            return {"ok": True, "hits": [], "reason": "索引为空，尚未建立语义索引"}
        # 维度对不上说明嵌入模型换过，旧向量全部失效。这必须说清楚：检索结果
        # 变空是「索引过期」而不是「库里没有」，两者要采取的处置完全不同。
        return {
            "ok": True,
            "hits": [],
            "stale": True,
            "reason": f"索引维度（{_dominant_dim()}）与当前模型（{dim}）不一致，需要重建索引",
        }
    usable = rows
    ranked = _rank(usable, query_vector)

    # 按 entry 收集所有过阈的块，而不是「取每个 entry 的最高分块」。
    # 一本书里「1927 年的目录」和「南昌起义那一段」是两块不同的内容，只留最高分
    # 那块，等于让目录把正文挤掉——用户搜「南昌起义」，拿到的是一行事件名列表。
    #
    # 这里有两个分数，用途不同，不能混：
    #   · `raw`      —— 余弦原值，**只用来判断这一块够不够格算召回**
    #   · `weighted` —— 乘了 `_PART_WEIGHT` 之后的值，**只用来排序和对外报分**
    # 混用一个会把目录块判死：查「1927 年发生了什么」，目录块原值 0.5424 是明确
    # 的命中，乘 0.85 掉到 0.461 就会落空——而这个问题本来就该由目录来答。
    grouped: dict[str, list[tuple[dict[str, Any], float, float]]] = {}
    floor = min_score if min_score is not None else min(MIN_SCORE, MIN_SCORE_BOOK)
    for row, raw in ranked:
        if raw < floor:
            break  # ranked 已按分数降序，后续只会更低
        if min_score is None:
            threshold = MIN_SCORE_BOOK if row.get("kind") == "book" else MIN_SCORE
        else:
            threshold = min_score
        if raw < threshold:
            continue
        weighted = raw * _PART_WEIGHT.get(str(row.get("part") or ""), 1.0)
        grouped.setdefault(row["entry_id"], []).append((row, weighted, raw))

    # 每个条目内部也要按加权分重排：上面那个循环沿的是 `ranked` 的原始分数
    # 顺序，不重排的话 `candidates[0]` 仍然是原始分最高的那一块，折扣白打了。
    for bucket in grouped.values():
        bucket.sort(key=lambda item: -item[1])
    ordered = sorted(grouped.items(), key=lambda item: -item[1][0][1])
    hits: list[dict[str, Any]] = []
    for entry_id, candidates in ordered[: max(1, limit)]:
        top_row, top_score, _ = candidates[0]
        hit: dict[str, Any] = {
            "entry_id": entry_id,
            "score": round(float(top_score), 4),
            "text": top_row["text"],
            "part": top_row["part"],
            "ordinal": top_row["ordinal"],
            # 书的检索结果必须能说到「哪一章哪一节」——只说命中了某本书，等于
            # 没说。card 的这两项为空，前端按空值决定是否显示章节行。
            "section": top_row["section"],
            "chapter": top_row.get("chapter") or "",
            "heading": top_row.get("heading") or "",
        }
        if len(candidates) > 1:
            hit["chunks"] = [
                {
                    "score": round(float(weighted), 4),
                    "text": row["text"],
                    "part": row["part"],
                    "ordinal": row["ordinal"],
                    "chapter": row.get("chapter") or "",
                    "heading": row.get("heading") or "",
                }
                for row, weighted, _ in candidates[:BOOK_CHUNKS]
            ]
        # 最高分的正文块单独标出来：它是「直接回答问题的这一段」。
        # 目录块分数常年更高（它把所有事件名列了一遍），但那不是答案——所以
        # 这个字段不看主命中是什么，只要有过阈的正文块就给。
        body_rows = [item for item in candidates if item[0]["part"] == "body"]
        if body_rows:
            best_body, best_body_score, _ = body_rows[0]
            hit["best_body"] = {
                "score": round(float(best_body_score), 4),
                "text": best_body["text"],
                "chapter": best_body.get("chapter") or "",
                "heading": best_body.get("heading") or "",
                "ordinal": best_body["ordinal"],
            }
        hits.append(hit)
        if len(hits) >= max(1, limit):
            break

    # 「检索出来了但不对」和「一条都没召回」是两种不同的毛病：前者要调阈值或换
    # 模型，后者要看索引有没有落盘。分数和召回数放在一行里，才分得清。
    logs.event(
        "retrieve", query=term, hits=len(hits), scanned=len(usable),
        top=round(float(ordered[0][1][0][1]), 3) if ordered else None,
        model=emb.model,
    )
    return {"ok": True, "hits": hits, "model": emb.model, "scanned": len(usable)}


def _rank(
    rows: list[dict[str, Any]], query_vector: list[float]
) -> list[tuple[dict[str, Any], float]]:
    """精确扫描。书级规模（千级片段）下全表算一遍仍然比引入向量库便宜。

    归一化整矩阵一次算完，而不是逐行调用 `_normalise`：1000 行的逐行范数是 1000
    次小规模 numpy 调用，开销全在 Python 层；整矩阵算一次是同样的数学，快一个量级。
    """
    if np is not None:
        dim = len(query_vector)
        usable = [row for row in rows if len(row["vector"]) == dim]
        if not usable:
            return []
        matrix = np.vstack([row["vector"] for row in usable]).astype(np.float32, copy=False)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        query = np.asarray(_normalise(query_vector), dtype=np.float32)
        scores = matrix @ query
        order = scores.argsort()[::-1]
        return [(usable[int(i)], float(scores[int(i)])) for i in order]

    normalized = _normalise(query_vector)
    scored = [(row, _dot(_normalise(row["vector"]), normalized)) for row in rows]
    scored.sort(key=lambda item: -item[1])
    return scored


def _load_rows(dim: int = 0) -> list[dict[str, Any]]:
    """读出可用向量。

    连表过滤 `rejected` 是一道保险：条目被弃用时应当已经删过向量，但如果某次
    删除失败（或历史数据没回填过），这条过滤能保证已放弃的内容不会被当证据召回。
    """
    conn = db.connect()
    sql = (
        "SELECT c.entry_id, c.part, c.ordinal, c.text, c.section, c.chapter, c.heading,"
        " c.vector, e.kind"
        " FROM studio_kb_chunks c"
        " JOIN knowledge_entries e ON e.id = c.entry_id"
        " WHERE e.status != 'rejected'"
    )
    params: list[Any] = []
    if dim:
        sql += " AND c.dim = ?"
        params.append(int(dim))
    rows = conn.execute(sql, params).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        vector = _decode(row["vector"])
        # 用 len 而不是真值判断：numpy 数组的真值是歧义的，`if not arr` 会抛错。
        if vector is None or len(vector) == 0:
            continue
        out.append(
            {
                "entry_id": row["entry_id"],
                "part": row["part"],
                "ordinal": int(row["ordinal"] or 0),
                "text": row["text"],
                "section": row["section"],
                "chapter": row["chapter"] or "",
                "heading": row["heading"] or "",
                # 阈值按类型取，所以查询时要知道这条属于卡片还是书。
                "kind": row["kind"] or "card",
                "vector": vector,
            }
        )
    return out


def _normalise(vector: list[float]) -> Any:
    if np is None:
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr)) or 1.0
    return arr / norm


def _dot(a: Iterable[float], b: Iterable[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def rrf(*rankings: list[str], k: int = RRF_K) -> list[str]:
    """Reciprocal rank fusion —— 只按名次融合，不需要标定两套分数。

    主题检索的分数是字面重合率，向量检索是余弦，两者尺度不同，直接加权必须先
    用标注数据标定权重。RRF 绕开了这件事：它只问「在各自榜单上排第几」。
    """
    totals: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            totals[key] = totals.get(key, 0.0) + 1.0 / (k + rank)
    return [key for key, _ in sorted(totals.items(), key=lambda item: -item[1])]


# ============================================================================
# 状态
# ============================================================================


def stored_hash(entry_id: str) -> str:
    conn = db.connect()
    row = conn.execute(
        "SELECT entry_hash FROM studio_kb_chunks WHERE entry_id = ? LIMIT 1",
        (str(entry_id),),
    ).fetchone()
    return str(row["entry_hash"]) if row else ""


def count(entry_id: str) -> int:
    conn = db.connect()
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM studio_kb_chunks WHERE entry_id = ?", (str(entry_id),)
    ).fetchone()
    return int(row["n"]) if row else 0


def _total_chunks() -> int:
    conn = db.connect()
    row = conn.execute("SELECT COUNT(*) AS n FROM studio_kb_chunks").fetchone()
    return int(row["n"]) if row else 0


def _indexed_entries() -> int:
    conn = db.connect()
    row = conn.execute(
        "SELECT COUNT(DISTINCT entry_id) AS n FROM studio_kb_chunks"
    ).fetchone()
    return int(row["n"]) if row else 0


def _dominant_dim() -> int:
    conn = db.connect()
    row = conn.execute(
        "SELECT dim FROM studio_kb_chunks WHERE dim > 0"
        " GROUP BY dim ORDER BY COUNT(*) DESC LIMIT 1"
    ).fetchone()
    return int(row["dim"]) if row else 0


def stale_entry_ids(entries: Iterable[dict[str, Any]]) -> list[str]:
    """内容变过、或向量没补齐的条目。重建索引时只需处理这些。

    「有块但没向量」也算过期：那是一次嵌入故障留下的半成品（索引降级成纯文本），
    文本指纹和条目对得上，只比 fingerprint 会让它永远补不回来。
    """
    pending = _entries_missing_vectors()
    out: list[str] = []
    for entry in entries:
        entry_id = str(entry.get("id") or "").strip()
        if not entry_id:
            continue
        if str(entry.get("status") or "") == "rejected":
            continue
        if entry_id in pending or stored_hash(entry_id) != entry_hash(entry):
            out.append(entry_id)
    return out


def _entries_missing_vectors() -> set[str]:
    """哪些条目存在 dim=0 的块（只有文本、没有向量）。"""
    conn = db.connect()
    rows = conn.execute(
        "SELECT DISTINCT entry_id FROM studio_kb_chunks WHERE dim = 0 OR length(vector) = 0"
    ).fetchall()
    return {str(row["entry_id"]) for row in rows}


def _structure_mismatches() -> int:
    """索引结构不对的整本资料数量。

    「整本资料」应当切成 chapter / body 块。如果库里出现 `point` 块，说明它被
    按卡片结构切过一次（`key_points` 那一百个章节标题变成了独立向量）。

    这种数据不会让任何东西报错：检索照常返回，健康检查照常 ready，只是排序被
    一堆「1921 年」污染了。把它数出来放进 `health()`，是让这类损坏至少有个
    出口——用户能看到「1 本资料索引结构异常」，而不是对着一个搜不准的库发愣。
    """
    conn = db.connect()
    row = conn.execute(
        "SELECT COUNT(DISTINCT e.id) AS n FROM knowledge_entries e"
        " JOIN studio_kb_chunks c ON c.entry_id = e.id"
        " WHERE e.kind = 'book' AND c.part = 'point'"
    ).fetchone()
    return int(row["n"]) if row else 0


def outline(entry_id: str, *, order: list[str] | None = None, per_chapter: int = 4) -> list[dict[str, Any]]:
    """一本书的分章清单：章节标题 + 章内事件标题。

    面板原来只画章节标题。这本书的章节标题就是年份（「1921 年」），于是「这本
    书收了什么」在界面上退化成一列数字——而事件名其实早就在库里：每个事件块带
    一条 `heading`，章的摘要块甚至把当章事件写成了「1921 年：中共一大召开、…」。

    顺序按 `order`（也就是入库时存下的 `key_points`）走，不用 `ORDER BY chapter`：
    「第一章 / 第二章」按字典序会排成「第一章 / 第七章 / 第三章」，年份碰巧对，
    换个文档就错了。
    """
    conn = db.connect()
    rows = conn.execute(
        "SELECT chapter, heading FROM studio_kb_chunks WHERE entry_id = ?",
        (entry_id,),
    ).fetchall()

    by_title: dict[str, dict[str, Any]] = {}
    for row in rows:
        title = str(row["chapter"] or "").strip()
        if not title:
            continue
        chapter = by_title.setdefault(
            title, {"title": title, "events": [], "event_total": 0, "blocks": 0}
        )
        chapter["blocks"] += 1
        heading = str(row["heading"] or "").strip()
        # 章标题本身会作为一条块出现，它不是事件。
        if not heading or heading == title:
            continue
        chapter["event_total"] += 1
        if len(chapter["events"]) < per_chapter and heading not in chapter["events"]:
            chapter["events"].append(heading)

    ordered: list[dict[str, Any]] = []
    for title in order or []:
        chapter = by_title.pop(str(title).strip(), None)
        if chapter:
            ordered.append(chapter)
    # `key_points` 有 500 条上限，超出或没写进去的章节按字典序补在后面，
    # 总比整段消失好。
    ordered.extend(by_title[title] for title in sorted(by_title))
    return ordered


def health() -> dict[str, Any]:
    """语义检索现在到底能不能用。

    只检查配置，不发网络请求——这个函数会被状态接口高频调用，一次往返就够慢了。
    真正验证端点是否活着用 `probe()`。

    但「不发请求」不等于「什么都不读」：这里会真的解一行向量看看。原因是实际
    踩过——向量列从 JSON 文本改成 BLOB 之后，一个还在跑旧代码的进程每次走向量
    检索都抛 `UnicodeDecodeError`（它把 float32 字节当 JSON 解析），而当时这个
    接口照样回答 `ready=true`。健康检查不碰数据，就会在数据已经读不出来的时候
    继续说自己健康。
    """
    emb = Embedder.from_settings()
    configured = emb is not None
    chunks = _total_chunks()
    mismatched = _structure_mismatches()
    unreadable = 0 if chunks == 0 else (0 if _reads_any_vector() else 1)
    missing = _text_only_chunks()
    if configured and chunks == 0:
        reason = "已配置向量模型但没有索引，需重建索引"
    elif not configured:
        reason = "未配置向量化模型，语义检索不可用（主题检索不受影响）"
    elif unreadable:
        reason = "索引里的向量读不出来（存的是二进制、读的一方按文本解析）：请重启后端；仍失败则重建索引"
    else:
        reason = "语义检索可用"
    if mismatched:
        reason += f"；{mismatched} 本资料的索引结构异常（被按卡片切分），需重建索引"
    if missing:
        # 这类块是嵌入故障留下的半成品：字面检索能用，语义召回完全看不到它们。
        # 不说出来的话，用户只会觉得「这本书搜得不全」，而修法只有一句「重建索引」。
        reason += f"；{missing} 块只有文本、没有向量（上次嵌入失败），需重建索引"
    return {
        "configured": configured,
        "provider": getattr(emb, "provider", "") if emb else "",
        "model": emb.model if emb else "",
        "base_url": emb.base_url if emb else "",
        "chunks": chunks,
        "text_only_chunks": missing,
        "indexed_entries": _indexed_entries(),
        "dim": _dominant_dim(),
        "structure_mismatches": mismatched,
        "unreadable": bool(unreadable),
        "ready": configured and chunks > 0 and not mismatched and not unreadable and not missing,
        "reason": reason,
    }


def _text_only_chunks() -> int:
    """有多少块只有文本、没有向量。见 `index_entry` 里的降级说明。"""
    try:
        row = db.connect().execute(
            "SELECT COUNT(*) AS n FROM studio_kb_chunks WHERE dim = 0 OR length(vector) = 0"
        ).fetchone()
    except Exception:  # noqa: BLE001 - 表还没建就是 0
        return 0
    return int(row["n"]) if row else 0


def _reads_any_vector() -> bool:
    """能否把库里任意一行向量解出来。读不出来就说明这层实际不可用。

    取一行而不是全表：健康检查会被频繁调用，而这一行足以区分「格式对不上」和
    「只是没数据」。异常一律吞掉转成 False——健康检查本身不该抛。
    """
    try:
        row = db.connect().execute(
            "SELECT vector FROM studio_kb_chunks WHERE length(vector) > 0 LIMIT 1"
        ).fetchone()
    except Exception:  # noqa: BLE001 - 表还没建也算「不可读」
        return False
    if row is None:
        return True  # 没有数据可读，不算故障
    try:
        value = _decode(row["vector"])
    except Exception:  # noqa: BLE001
        return False
    # 用 len 而不是真值判断：numpy 数组的真值是歧义的。
    return value is not None and len(value) > 0


async def probe(text: str = "中国共产党的诞生") -> dict[str, Any]:
    """真调一次嵌入接口，证明端点通不通。状态页用它做「测试连接」。"""
    emb = Embedder.from_settings()
    if emb is None:
        return {"ok": False, "reason": "未配置向量化模型"}
    try:
        vectors = await emb.embed([text])
    except EmbeddingError as exc:
        return {"ok": False, "model": emb.model, "reason": f"调用失败：{str(exc)[:200]}"}
    return {
        "ok": True,
        "model": emb.model,
        "base_url": emb.base_url,
        "dim": len(vectors[0]) if vectors else 0,
        "latency_text": text[:20],
    }
