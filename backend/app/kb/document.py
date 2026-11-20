"""整本书的解析与切块：从原始 txt / md 到可召回的片段。

和 `vector.split_entry` 的区别：

  `split_entry` 处理的是「一条主题卡片」——标题 + 几条要点 + 一段正文，三块
  内容语义上平行。书不是这样：它有层级（章 > 节 > 段），而且**层级本身是检索
  信号**。问「1927 年发生了什么」时，能回答的是那一章，不是某个孤立段落。
  所以这里先还原结构，再在结构内部切块，而不是对全文做统一的按字切分。

三个原则，每一个都是为了避免一类具体的错误：

1. **片段绝不跨章**。跨章拼出来的片段无法给出可信出处，而「有出处」是这个
   知识库存在的理由。宁可让一个片段偏短。

2. **结构边界优先于字数边界**。一个 200 字的事件是一个完整语义单元；为了凑
   到 320 字把它和相邻的另一个事件拼起来，会让「讲的是哪件事」变得含糊，检索
   分数随之失去区分度。字数只在下限（太短没有检索价值）和上限（超出就按句切）
   起作用。

3. **带标题的段落不和别的段落合并**。书里每个带标题的段落是一个独立单元，
   合并等于制造一个「讲了两个主题」的片段——这种片段对任何一方都是噪声。

解析失败不抛异常，降级成「整本一章」。用户上传一本书却因为格式不认识而拿到
一个报错，比拿到一本没分章的书糟糕得多：后者仍然可用，前者什么也做不了。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..platform.rag.embedder import SplitConfig, split_block


#: 片段目标长度。书里一个论点常要一整段才讲完，比卡片的 320 略长。
TARGET_CHARS = 360
MAX_CHARS = 520
#: 低于这个长度、又不带标题的段落，单独成块没有检索价值（几个词无法承载
#: 一个可被问到的意思），会和相邻的同类段落合并。
MIN_CHARS = 80

_SPLIT = SplitConfig(target=TARGET_CHARS, max_chars=MAX_CHARS)

#: 章节标记。覆盖中文书最常见的四种写法，从宽到严依次匹配。
_CHAPTER_PATTERNS = (
    re.compile(r"^第\s*[0-9一二三四五六七八九十百千零〇两]+\s*[章回卷篇部]\s*[:：、.\s]*(.*)$"),
    re.compile(r"^Chapter\s+\d+\s*[:：.\s]*(.*)$", re.IGNORECASE),
)
#: 节标记。比章低一级，不单独成章，但会进标题路径。
_SECTION_PATTERNS = (
    re.compile(r"^第\s*[0-9一二三四五六七八九十百千零〇两]+\s*节\s*[:：、.\s]*(.*)$"),
    re.compile(r"^[一二三四五六七八九十]+\s*[、.]\s*(.{1,40})$"),
    re.compile(r"^\d+\.\d+\s+(.{1,40})$"),
)
#: Markdown 的行内加粗标题：「**中共一大召开**：正文」。
_BOLD_ITEM = re.compile(r"^\*\*(.+?)\*\*\s*[：:]\s*(.+)$")
#: 转换残缺：加粗标题被单独拆成一行，冒号开头的正文落在下一行。
_ORPHAN_COLON = re.compile(r"^[：:]\s*(.+)$")


@dataclass
class Block:
    """章内的一个语义单元。`heading` 为空表示它是一段没有小标题的正文。"""

    text: str
    heading: str = ""


@dataclass
class Chapter:
    title: str
    blocks: list[Block] = field(default_factory=list)
    #: 多级标题时的完整路径，如「第一章 > 第一节」。纯文本没有多级时等于 title。
    path: str = ""


@dataclass
class Document:
    title: str
    chapters: list[Chapter] = field(default_factory=list)
    note: str = ""

    @property
    def char_count(self) -> int:
        return sum(len(b.text) for c in self.chapters for b in c.blocks)


def _clean(line: str) -> str:
    """去掉行首尾的空白和 markdown 修饰，保留内容。

    行内的 `**` 一律去掉：它在原文里是排版标记，进了片段就会出现在引用里，
    读起来像文件损坏。`>` 引用标记同理。
    """
    text = line.strip()
    text = re.sub(r"^>\s*", "", text)
    text = text.replace("**", "")
    return text.strip()


def _looks_like_markdown(text: str) -> bool:
    """有 ATX 标题就算 markdown。

    不用扩展名判断：用户可能把 md 存成 .txt，也可能反过来。内容比后缀可信。
    """
    return bool(re.search(r"^#{1,3}\s+\S", text, re.MULTILINE))


def parse_markdown(text: str, fallback_title: str = "") -> Document:
    """解析 markdown，还原「书名 > 章 > 小节/事件」的层级。

    状态机而不是正则一把梭，是因为 h3 的含义取决于它后面跟什么：这本书里有两个
    `### 标题` 被 Word 转换拆成了「标题一行、正文另起一行以冒号开头」，那些 h3
    其实是条目标题而不是章节标题。只有看下一行才能判断。
    """
    lines = text.split("\n")
    doc = Document(title="")
    note_lines: list[str] = []
    # 已解析出至少一个二级标题之后，`>` 引用就不再是前言，按正文处理。
    in_body = False
    pending_heading = ""
    buffer: list[str] = []
    # 遇到 h3 时先记下它的标题；下一行若是「：正文」，就合并成一个 block。
    awaiting_heading_text = ""

    def flush() -> None:
        nonlocal buffer
        piece = " ".join(x for x in buffer if x).strip()
        buffer = []
        if not piece:
            return
        if not doc.chapters:
            doc.chapters.append(Chapter(title="全文", path="全文"))
        doc.chapters[-1].blocks.append(Block(text=piece, heading=pending_heading))

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            flush()
            continue

        if stripped.startswith("# "):
            flush()
            doc.title = doc.title or _clean(stripped[2:])
            pending_heading = ""
            continue
        if stripped.startswith("## "):
            flush()
            in_body = True
            title = _clean(stripped[3:]) or f"第{len(doc.chapters) + 1}章"
            doc.chapters.append(Chapter(title=title, path=title))
            pending_heading = ""
            awaiting_heading_text = ""
            continue
        if stripped.startswith("### "):
            flush()
            in_body = True
            heading = _clean(stripped[4:])
            if not doc.chapters:
                doc.chapters.append(Chapter(title="全文", path="全文"))
            pending_heading = heading
            awaiting_heading_text = heading
            continue
        if stripped.startswith(">") and not in_body:
            note_lines.append(_clean(stripped))
            continue
        if stripped.startswith(("- ", "* ", "+ ")):
            buffer.append(_clean(stripped[2:]))
            continue
        if stripped.startswith("|"):
            # 表格行拼成一句话，丢掉分隔行（|---|---|）。
            if set(stripped) <= set("|-: "):
                continue
            cells = [_clean(c) for c in stripped.strip("|").split("|") if _clean(c)]
            if cells:
                buffer.append(" ".join(cells))
            continue

        item = _BOLD_ITEM.match(stripped)
        if item:
            flush()
            if not doc.chapters:
                doc.chapters.append(Chapter(title="全文", path="全文"))
            doc.chapters[-1].blocks.append(
                Block(text=_clean(item.group(2)), heading=_clean(item.group(1)))
            )
            pending_heading = ""
            awaiting_heading_text = ""
            continue

        orphan = _ORPHAN_COLON.match(stripped)
        if orphan and awaiting_heading_text:
            # `### 标题` 后面紧跟「：正文」——把标题还给这一块，不要让它变成
            # 一个空的章节标题，也不要让正文丢掉它属于哪件事。
            if not doc.chapters:
                doc.chapters.append(Chapter(title="全文", path="全文"))
            doc.chapters[-1].blocks.append(
                Block(text=_clean(orphan.group(1)), heading=awaiting_heading_text)
            )
            pending_heading = ""
            awaiting_heading_text = ""
            continue

        buffer.append(_clean(stripped))

    flush()
    doc.title = doc.title or fallback_title or "未命名资料"
    doc.note = " ".join(note_lines).strip()
    return doc


def parse_plain_text(text: str, fallback_title: str = "") -> Document:
    """解析纯文本，用「第X章 / 一、 / 1.1」这类标记识别结构。

    识别不出来时把全文当作一章——这是刻意的降级。一本没分章的书仍然可以按段落
    切块检索，只是引用只能说到「第几段」。
    """
    doc = Document(title=fallback_title or "", note="")
    seen_heading = False
    pending_heading = ""
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        piece = " ".join(x for x in buffer if x).strip()
        buffer = []
        if not piece:
            return
        if not doc.chapters:
            doc.chapters.append(Chapter(title="全文", path="全文"))
        doc.chapters[-1].blocks.append(Block(text=piece, heading=pending_heading))

    chapter_hit = False
    for raw in text.split("\n"):
        stripped = _clean(raw)
        if not stripped:
            flush()
            continue

        matched_chapter = ""
        for pattern in _CHAPTER_PATTERNS:
            m = pattern.match(stripped)
            if m:
                matched_chapter = stripped
                break
        if matched_chapter and len(matched_chapter) <= 48:
            flush()
            chapter_hit = True
            title = matched_chapter
            doc.chapters.append(Chapter(title=title, path=title))
            pending_heading = ""
            continue

        matched_section = ""
        for pattern in _SECTION_PATTERNS:
            m = pattern.match(stripped)
            if m:
                matched_section = stripped if len(stripped) <= 48 else ""
                break
        if matched_section:
            flush()
            seen_heading = True
            pending_heading = matched_section
            continue

        buffer.append(stripped)

    flush()

    if not chapter_hit:
        # 没识别出任何章标记：整本一章，但把「一、」「1.1」这类识别到的节标题
        # 保留在 heading 上，至少引用能说到小标题。
        doc.chapters = [c for c in doc.chapters if c.blocks] or [
            Chapter(title=fallback_title or "全文", path=fallback_title or "全文")
        ]
        for chapter in doc.chapters:
            chapter.path = chapter.title
    if not doc.title:
        doc.title = fallback_title or "未命名资料"
    _ = seen_heading
    return doc


def decode(raw: bytes, *, filename: str = "") -> tuple[str, str]:
    """把上传的字节还原成文本，返回 (文本, 实际用的编码)。

    中文 txt 有一半不是 UTF-8：Windows 记事本存的是 GBK，老资料是 GB18030，
    从 Word 复制出来的偶尔带 BOM。一律按 UTF-8 解会把整本书变成乱码，而乱码
    入库之后**不会报错**——它会安静地建成一个搜不到任何东西的索引，用户只会
    觉得「这本书怎么什么都搜不出来」。所以这里必须显式检测。

    GB18030 放在 GB2312/GBK 之前：它是后两者的超集，用超集解码永远不会因为
    生僻字失败，而已知是 GB2312 时按 GB18030 解也是同样的结果。
    """
    if not raw:
        return "", "utf-8"

    # BOM 最可靠，先处理。UTF-8 BOM 会被当成正文第一个字符，导致书名变成
    # 「\ufeff中国共产党……」，在界面上看不出问题但按书名查重会失效。
    for bom, encoding in (
        (b"\xef\xbb\xbf", "utf-8-sig"),
        (b"\xff\xfe\x00\x00", "utf-32-le"),
        (b"\x00\x00\xfe\xff", "utf-32-be"),
        (b"\xff\xfe", "utf-16-le"),
        (b"\xfe\xff", "utf-16-be"),
    ):
        if raw.startswith(bom):
            try:
                return raw.decode(encoding), encoding
            except UnicodeDecodeError:
                break

    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass

    # UTF-8 失败之后必须先探测，不能直接硬解 GB18030：GB18030 几乎接受任何
    # 字节序列，硬解一定「成功」，于是 Big5 或 Shift-JIS 的文本会被安静地解成
    # 一堆错字。探测给出低置信度时再退回 GB18030——那是中文 Windows 最可能的
    # 编码，比乱码好。
    try:
        import chardet  # type: ignore

        guess = chardet.detect(raw)
        encoding = str(guess.get("encoding") or "").lower()
        confidence = float(guess.get("confidence") or 0.0)
        if encoding in ("gb2312", "gbk", "gb18030"):
            encoding = "gb18030"
        if encoding and confidence >= 0.6:
            return raw.decode(encoding, errors="replace"), encoding
    except Exception:  # noqa: BLE001 - chardet 是可选依赖
        pass

    try:
        return raw.decode("gb18030"), "gb18030"
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), "utf-8(replace)"


def parse(text: str, *, filename: str = "") -> Document:
    """按内容判断格式再解析。扩展名只用来兜底取名。"""
    stem = re.sub(r"\.[^.]+$", "", (filename or "").split("/")[-1].split("\\")[-1]).strip()
    body = (text or "").lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    if not body.strip():
        return Document(title=stem or "未命名资料")
    if _looks_like_markdown(body):
        return parse_markdown(body, fallback_title=stem)
    return parse_plain_text(body, fallback_title=stem)


# ============================================================================
# 切块
# ============================================================================


def _document_block(doc: Document) -> dict[str, Any]:
    """整本书的一览块：书名 + 说明 + 章名清单。

    它的作用是让「这本书讲了什么」这种问题能命中。章名清单截断到 300 字，
    否则一本 100 章的书会生成一个几千字的块，把向量稀释成一个平均值。
    """
    names = [c.title for c in doc.chapters if c.title]
    listing = "、".join(names)
    if len(listing) > 300:
        listing = listing[:300] + " 等"
    segments = [doc.title]
    if doc.note:
        segments.append(doc.note)
    if listing:
        segments.append(f"全书共 {len(names)} 章：{listing}")
    # 拼句子而不是直接加句号：原文的说明本身就以「。」结尾，硬拼会得到「。。」，
    # 这个块会进向量库，脏标点会一直留在引用里。
    text = "。".join(s.strip().rstrip("。") for s in segments if s.strip()) + "。"
    return {
        "part": "title",
        "ordinal": 0,
        "text": text,
        "section": doc.title,
        "chapter": "",
        "heading": doc.title,
        "embed_text": text,
    }


def _chapter_block(doc: Document, chapter: Chapter, ordinal: int) -> dict[str, Any]:
    """一章的一览块：章标题 + 本章各小标题。

    这本书按年份编排，一章就是一年。只索引章节正文会让「1927 年有哪些大事」
    这种问题命中若干条彼此无关的片段；先索引一份本章目录，命中后就能顺着章号
    把整章展开——这是「小-大检索」里「大」的那一半。

    `ordinal` 恒为 0，不传章序号：块 ID 里已经带了章节哈希，章序号是多余的，
    而且**有害**——删掉第一章会让后面所有章的序号前移，于是每一章的目录块都被
    判定成「新块」而重嵌。这个函数收到 `ordinal` 参数只是为了让调用点看起来
    一致，值一律忽略。
    """
    _ = ordinal
    heads = [b.heading for b in chapter.blocks if b.heading]
    listing = "、".join(heads)
    if len(listing) > 400:
        listing = listing[:400] + " 等"
    text = f"{chapter.path or chapter.title}"
    if listing:
        text = f"{text}：{listing}"
    return {
        "part": "chapter",
        "ordinal": 0,
        "text": text,
        "section": f"{doc.title} · {chapter.title}" if doc.title != chapter.title else doc.title,
        "chapter": chapter.path or chapter.title,
        "heading": "",
        "embed_text": f"《{doc.title}》{text}",
    }


def _block_chunks(
    doc: Document, chapter: Chapter, block: Block, ordinal: int
) -> list[dict[str, Any]]:
    """把一个语义单元变成一到多个片段。"""
    text = block.text.strip()
    if not text:
        return []
    section = f"{doc.title} · {chapter.title}" if doc.title != chapter.title else doc.title
    chapter_path = chapter.path or chapter.title
    pieces = [text] if len(text) <= MAX_CHARS else split_block(text, _SPLIT)
    out: list[dict[str, Any]] = []
    for index, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue
        # 续块保留原标题并标注序号：引用时不能把第二段说成是另一个事件，
        # 但也要让读者知道它是同一事件的后半部分。
        heading = block.heading if index == 0 else (
            f"{block.heading}（续{index}）" if block.heading else ""
        )
        head = heading or chapter_path
        out.append(
            {
                "part": "body",
                "ordinal": ordinal + index,
                "text": piece,
                "section": section,
                "chapter": chapter_path,
                "heading": heading,
                "embed_text": f"《{doc.title}》{chapter_path}·{head}：{piece}",
            }
        )
    return out


def split_document(doc: Document) -> list[dict[str, Any]]:
    """整本书切成片段。

    `ordinal` 是**章内**序号，不是全书序号。这一点看着无关紧要，但决定了增量
    索引能不能用：

      块 ID 由 (entry_id, part, ordinal) 组成，而增量索引判的是「这个 ID 的指纹
      变没变」。如果 ordinal 是全书递增的，删掉第一章会让后面所有块的序号前移，
      于是几百块全部被判定为「新块」，一次删章等于整本重嵌——实测删掉一章触发了
      386 块重嵌，而真正受影响的只有那 4 块。

      按章内计数，删除或插入一章只影响该章自己的块。代价是章内插入仍会让本章
      后续块重嵌，但一章只有两三个事件，这个级联可以接受。

    `chapter` 同时进块 ID，否则每章都从 0 开始编号，块 ID 会撞。
    """
    chunks: list[dict[str, Any]] = []
    if doc.chapters:
        chunks.append(_document_block(doc))

    for chapter_index, chapter in enumerate(doc.chapters):
        if chapter.blocks:
            chunks.append(_chapter_block(doc, chapter, chapter_index))

        # 章内相邻的短小无标题段落先合并，再切块。合并只发生在同一章内部，
        # 且两者都没有标题——有标题的段落是独立事件，拼起来会变成「讲了两个
        # 主题」的片段。
        merged: list[Block] = []
        carry = ""
        for block in chapter.blocks:
            text = block.text.strip()
            if not text:
                continue
            if block.heading:
                if carry:
                    merged.append(Block(text=carry))
                    carry = ""
                merged.append(block)
                continue
            if len(text) >= MIN_CHARS:
                if carry:
                    merged.append(Block(text=carry))
                    carry = ""
                merged.append(block)
                continue
            if carry and len(carry) + len(text) + 1 <= TARGET_CHARS:
                carry = f"{carry} {text}"
                continue
            if carry:
                merged.append(Block(text=carry))
            carry = text
        if carry:
            merged.append(Block(text=carry))

        chapter_ordinal = 0
        for block in merged:
            pieces = _block_chunks(doc, chapter, block, chapter_ordinal)
            chunks.extend(pieces)
            chapter_ordinal += len(pieces)

    return chunks


def preview(doc: Document, *, limit: int = 8) -> dict[str, Any]:
    """给上传接口返回的结构摘要：让用户在入库前看到「切成了什么样」。"""
    samples: list[dict[str, Any]] = []
    for chapter in doc.chapters[:limit]:
        samples.append(
            {
                "chapter": chapter.path or chapter.title,
                "blocks": len(chapter.blocks),
                "headings": [b.heading for b in chapter.blocks if b.heading][:4],
            }
        )
    return {
        "title": doc.title,
        "note": doc.note,
        "chapters": len(doc.chapters),
        "blocks": sum(len(c.blocks) for c in doc.chapters),
        "chars": doc.char_count,
        "samples": samples,
    }
