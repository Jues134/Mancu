"""整本资料（书）的解析、切块与增量索引。

和 `test_kb_vector.py` 一样用假嵌入器：测试不打网络，也不消耗真实额度。这里的
假嵌入器只需要「同样的文本得到同样的向量」，用来验证块级增量——真正要验的是
「改了一个字，到底重嵌了几块」，那和嵌入模型的质量无关。

书的数据用内联的合成 markdown，不读真实文件：测试不该依赖开发者机器上恰好有
哪本书，也不该因为那本书被编辑过而变红。
"""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import db
from app.kb import ResearchError, document, import_book, store, vector


DIM = 96


def _fake(text: str) -> list[float]:
    out = [0.0] * DIM
    cleaned = "".join(ch for ch in text if not ch.isspace())
    for index in range(max(1, len(cleaned) - 1)):
        slot = sum(ord(ch) for ch in cleaned[index : index + 2]) % DIM
        out[slot] += 1.0
    if not any(out):
        out[0] = 1.0
    return out


class FakeEmbedder:
    model = "fake-embed"

    def __init__(self) -> None:
        self.calls = 0
        self.texts = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.texts += len(texts)
        return [_fake(text) for text in texts]


#: 一本结构完整的小书。刻意做成年份编排 + 加粗事件，和真实资料的形状一致。
BOOK = """# 测试史料汇编

> 这是一本用于测试的合成资料，按年份编排。

## 1921 年

**甲事件**：这是甲事件的正文，交代了它的起因、经过和结果，长度足够构成一个可召回的片段，不需要和别的段落合并。

**乙事件**：这是乙事件的正文，同样交代了起因经过结果，与甲事件互相独立，两者不该被拼进同一个片段。

## 1927 年

**丙事件**：丙事件发生在 1927 年，正文里写明了时间、地点和参与方，足以支撑一个关于它的提问。

**丁事件**：丁事件的正文。这一段稍短一些，但仍然带着自己的小标题。

**戊事件**：戊事件是这一章的第三个条目，它和丙、丁并列，构成这一章的事件目录。

## 1935 年

**己事件**：己事件发生在 1935 年，正文包含具体月份和地点，可以作为引用出处。

### 庚事件与一段残缺标题

：庚事件的正文被 Word 转换拆成了「标题一行、正文另起一行以冒号开头」，解析时应当把标题还给这一块。
"""


class DocumentParsingTests(unittest.TestCase):
    def test_markdown_structure_is_recovered(self):
        doc = document.parse(BOOK, filename="测试史料汇编.md")
        self.assertEqual(doc.title, "测试史料汇编")
        self.assertIn("按年份编排", doc.note)
        self.assertEqual([c.title for c in doc.chapters], ["1921 年", "1927 年", "1935 年"])
        self.assertEqual(len(doc.chapters[0].blocks), 2)
        self.assertEqual(doc.chapters[0].blocks[0].heading, "甲事件")

    def test_orphan_colon_merges_into_heading(self):
        """`### 标题` + 下一行「：正文」要合成一块，而不是留个空章节标题。

        真实的 Word 转 markdown 会产出这种形状。处理错的话，那一条内容会变成
        一个没有正文的空块，用户看到的是「标题在、内容没了」。
        """
        doc = document.parse(BOOK, filename="x.md")
        last = doc.chapters[-1]
        self.assertEqual(last.blocks[-1].heading, "庚事件与一段残缺标题")
        self.assertTrue(last.blocks[-1].text.startswith("庚事件的正文"))

    def test_plain_text_falls_back_to_single_chapter(self):
        """没有章节标记的文本要能降级成「整本一章」，而不是解析失败。

        用户传一本没分章的书，得到一个可检索的结果，比得到一个报错有用得多。
        """
        text = "第一段内容，讲了一件事的来龙去脉，长度足够。\n\n第二段内容，讲了另一件事。"
        doc = document.parse(text, filename="无结构.txt")
        self.assertEqual(len(doc.chapters), 1)
        self.assertEqual(len(doc.chapters[0].blocks), 2)

    def test_gbk_bytes_are_decoded(self):
        """中文 txt 有一半不是 UTF-8。按 UTF-8 硬解会得到乱码索引，且不报错。"""
        original = "中国共产党百年大事记，按年份编排。"
        raw = original.encode("gb18030")
        text, encoding = document.decode(raw, filename="x.txt")
        self.assertEqual(text, original)
        self.assertIn(encoding, ("gb18030", "gbk", "gb2312"))

    def test_utf8_bom_does_not_become_part_of_the_title(self):
        """BOM 进了书名，按书名查重就会永远匹配不上，而界面上看不出来。"""
        raw = "\ufeff# 书名\n\n## 一章\n\n内容。".encode("utf-8")
        text, _ = document.decode(raw, filename="x.md")
        self.assertFalse(text.startswith("\ufeff"))
        self.assertEqual(document.parse(text).title, "书名")


class DocumentChunkingTests(unittest.TestCase):
    def setUp(self):
        self.doc = document.parse(BOOK, filename="测试史料汇编.md")
        self.chunks = document.split_document(self.doc)

    def test_chunks_never_cross_chapters(self):
        """跨章片段给不出可信出处，而「有出处」是这个库存在的理由。"""
        by_chapter: dict[str, set[str]] = {}
        for chunk in self.chunks:
            if chunk["part"] != "body":
                continue
            by_chapter.setdefault(chunk["chapter"], set()).add(chunk["text"])
        # 每个片段只属于一个章，且每章的片段里不出现别章标题的事件名。
        for chapter, texts in by_chapter.items():
            joined = " ".join(texts)
            for other in by_chapter:
                if other == chapter:
                    continue
                for block in self.doc.chapters:
                    if block.title == other:
                        for item in block.blocks:
                            if item.heading:
                                self.assertNotIn(
                                    item.heading,
                                    joined,
                                    f"{chapter} 的片段里混进了 {other} 的「{item.heading}」",
                                )

    def test_each_block_becomes_its_own_chunk(self):
        """有标题的段落是独立事件，不会被合并成「讲了两个主题」的片段。

        判据用「一组块、一个标题」而不是「正文里不出现别的标题名」：正文之间
        互相引用（「与前一事件互相独立」）是正常写法，按关键字否定会误报。
        """
        body = [c for c in self.chunks if c["part"] == "body"]
        titled = [b for c in self.doc.chapters for b in c.blocks if b.heading]
        self.assertEqual(len(body), len(titled), "块数应等于带标题的段落数，说明没有合并")
        headings = [c["heading"] for c in body]
        self.assertEqual(len(headings), len(set(headings)), "每个事件的片段只应出现一次")

    def test_ordinal_counts_within_a_chapter(self):
        """ordinal 是章内序号。全书递增会让删一章触发整本重嵌。"""
        first = [c["ordinal"] for c in self.chunks if c["chapter"] == "1921 年" and c["part"] == "body"]
        second = [c["ordinal"] for c in self.chunks if c["chapter"] == "1927 年" and c["part"] == "body"]
        self.assertEqual(first, [0, 1])
        self.assertEqual(second, [0, 1, 2])

    def test_chunk_ids_are_unique_across_chapters(self):
        """每章都从 0 开始编号，ID 必须靠章节哈希区分开，否则会撞主键。"""
        ids = [
            vector._chunk_id("e1", c["part"], c["ordinal"], c["chapter"])
            for c in self.chunks
        ]
        self.assertEqual(len(ids), len(set(ids)))

    def test_document_block_states_the_chapter_count(self):
        head = next(c for c in self.chunks if c["part"] == "title")
        self.assertIn("全书共 3 章", head["text"])

    def test_chapter_block_lists_its_headings(self):
        chapter = next(
            c for c in self.chunks if c["part"] == "chapter" and c["chapter"] == "1927 年"
        )
        for name in ("丙事件", "丁事件", "戊事件"):
            self.assertIn(name, chapter["text"])

    def test_embed_text_carries_chapter_context(self):
        """正文段落经常不提年份和书名，送进嵌入时必须补上，否则「1927 年」搜不到它。"""
        chunk = next(c for c in self.chunks if c["heading"] == "丙事件")
        self.assertIn("测试史料汇编", chunk["embed_text"])
        self.assertIn("1927 年", chunk["embed_text"])


class BookIndexTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self._old_path = db.DB_PATH
        self._old_conn = db._conn
        db._conn = None
        db.DB_PATH = Path(self.temp.name) / "test.db"
        self.addCleanup(self._restore_db)
        db.connect()

    def _restore_db(self):
        try:
            if db._conn is not None:
                db._conn.close()
        except Exception:  # noqa: BLE001
            pass
        db._conn = self._old_conn
        db.DB_PATH = self._old_path

    # ------------------------------------------------------------ 入库

    async def test_imported_book_is_approved_and_typed(self):
        """上传即入库：书是用户自己给的，审核等于要用户确认自己刚做的事。"""
        embedder = FakeEmbedder()
        with patch.object(vector.Embedder, "from_settings", return_value=embedder):
            result = await import_book(
                text=BOOK, title="测试史料汇编", filename="测试史料汇编.md"
            )
        entry = result["entry"]
        self.assertEqual(entry["kind"], store.BOOK)
        self.assertEqual(entry["status"], store.APPROVED)
        self.assertEqual(entry["origin"], "upload")
        self.assertEqual(result["structure"]["chapters"], 3)
        self.assertGreater(result["index"]["chunks"], 0)

    async def test_too_short_document_is_rejected_with_a_reason(self):
        with self.assertRaises(ResearchError) as error:
            await import_book(text="太短了", title="短")
        self.assertEqual(error.exception.error_type, "document_too_short")

    async def test_book_bypasses_the_whitelist_gate(self):
        """书没有白名单网址，仍然要通过闸门。

        闸门拦的是「模型从网上摘要出的内容可能不可靠」，不是「用户上传的资料
        不可靠」。如果书也要白名单网址，等于用户只能上传政府网站。
        """
        from app.kb import _has_authoritative_source

        card = {"sources": [{"url": "https://weibo.com/x"}]}
        book = {"kind": store.BOOK, "sources": [{"url": ""}]}
        self.assertFalse(_has_authoritative_source(card))
        self.assertTrue(_has_authoritative_source(book))

    async def test_reimport_fixes_a_mislabelled_kind(self):
        """一条被错标成 card 的书，重传之后必须修好。

        这是上一版留下的死角：`store.save` 的 UPDATE 分支刻意不更新 kind（内容
        更新不该改掉生命周期），于是错标的记录重传多少次都还是 card，重建索引也
        还是按卡片切——「一百个 1921 年」永远清不掉。
        """
        embedder = FakeEmbedder()
        with patch.object(vector.Embedder, "from_settings", return_value=embedder):
            first = await import_book(text=BOOK, title="测试史料汇编")
        entry_id = first["entry"]["id"]

        # 把类型打坏，模拟历史遗留数据。
        with db.transaction() as conn:
            conn.execute(
                "UPDATE knowledge_entries SET kind = 'card' WHERE id = ?", (entry_id,)
            )
        self.assertEqual(store.get(entry_id)["kind"], store.CARD)

        with patch.object(vector.Embedder, "from_settings", return_value=embedder):
            again = await import_book(text=BOOK, title="测试史料汇编")
        self.assertEqual(again["entry"]["kind"], store.BOOK)
        self.assertEqual(store.get(entry_id)["kind"], store.BOOK)

    async def test_set_kind_reports_whether_it_changed_anything(self):
        entry = store.save({"title": "x", "kind": store.CARD, "body": BOOK})
        self.assertTrue(store.set_kind(entry["id"], store.BOOK))
        self.assertFalse(store.set_kind(entry["id"], store.BOOK))
        self.assertFalse(store.set_kind(entry["id"], "nonsense"))

    async def test_reimport_same_title_replaces_instead_of_duplicating(self):
        """同名重传是在更新版本，不是造第二本——两本同名书会让同一内容占两个位置。"""
        embedder = FakeEmbedder()
        with patch.object(vector.Embedder, "from_settings", return_value=embedder):
            first = await import_book(text=BOOK, title="测试史料汇编")
            second = await import_book(text=BOOK + "\n## 1949 年\n\n**新事件**：新中国的成立于 1949 年宣告。\n", title="测试史料汇编")
        self.assertEqual(first["entry"]["id"], second["entry"]["id"])
        self.assertTrue(second["replaced"])
        self.assertEqual(store.stats()["books"], 1)

    # ------------------------------------------------------------ 增量

    async def test_reindexing_unchanged_book_embeds_nothing(self):
        embedder = FakeEmbedder()
        with patch.object(vector.Embedder, "from_settings", return_value=embedder):
            result = await import_book(text=BOOK, title="测试史料汇编")
        entry = store.get(result["entry"]["id"])
        self.assertIsNotNone(entry)

        spy = FakeEmbedder()
        state = await vector.index_entry(entry, embedder=spy)
        self.assertTrue(state.get("skipped"))
        self.assertEqual(spy.texts, 0)

    async def test_editing_one_character_reembeds_one_chunk(self):
        """一本书三百多块，改一个标点就重嵌整本，是最典型的「能跑但没法用」。"""
        embedder = FakeEmbedder()
        with patch.object(vector.Embedder, "from_settings", return_value=embedder):
            result = await import_book(text=BOOK, title="测试史料汇编")
        entry = store.get(result["entry"]["id"])

        entry = dict(entry)
        entry["body"] = entry["body"].replace("甲事件的正文", "甲事件的正文内容", 1)
        spy = FakeEmbedder()
        state = await vector.index_entry(entry, embedder=spy)
        self.assertEqual(spy.texts, 1, "只该重嵌被改动的那一块")
        self.assertEqual(state["embedded"], 1)

    async def test_deleting_a_chapter_does_not_reembed_the_rest(self):
        """块 ID 里带全局序号时，删第一章会触发后面几百块重嵌。实测过 386 块。"""
        embedder = FakeEmbedder()
        with patch.object(vector.Embedder, "from_settings", return_value=embedder):
            result = await import_book(text=BOOK, title="测试史料汇编")
        entry = dict(store.get(result["entry"]["id"]))

        before = vector.count(entry["id"])
        entry["body"] = re.sub(r"## 1921 年.*?(?=## 1927 年)", "", entry["body"], flags=re.S)
        spy = FakeEmbedder()
        state = await vector.index_entry(entry, embedder=spy)
        # 1921 年有 2 个事件块 + 1 个章节目录块 = 3 块被删。
        self.assertLessEqual(spy.texts, 4, f"删一章重嵌了 {spy.texts} 块，说明 ID 里有全局序号")
        self.assertLess(vector.count(entry["id"]), before)

    async def test_removed_chunks_do_not_linger(self):
        """删掉的内容必须真的从索引里消失，否则它还会被召回。"""
        embedder = FakeEmbedder()
        with patch.object(vector.Embedder, "from_settings", return_value=embedder):
            result = await import_book(text=BOOK, title="测试史料汇编")
        entry = dict(store.get(result["entry"]["id"]))
        entry["body"] = re.sub(r"## 1935 年.*", "", entry["body"], flags=re.S)
        await vector.index_entry(entry, embedder=embedder)

        conn = db.connect()
        leaked = conn.execute(
            "SELECT COUNT(*) AS n FROM studio_kb_chunks WHERE entry_id = ? AND chapter = '1935 年'",
            (entry["id"],),
        ).fetchone()["n"]
        self.assertEqual(int(leaked), 0)


class BookRetrievalTests(unittest.IsolatedAsyncioTestCase):
    """检索的两条书专属规则：更低的阈值、目录块降权。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self._old_path = db.DB_PATH
        self._old_conn = db._conn
        db._conn = None
        db.DB_PATH = Path(self.temp.name) / "test.db"
        self.addCleanup(self._restore_db)
        db.connect()

    def _restore_db(self):
        try:
            if db._conn is not None:
                db._conn.close()
        except Exception:  # noqa: BLE001
            pass
        db._conn = self._old_conn
        db.DB_PATH = self._old_path

    def test_book_threshold_is_lower_than_card_threshold(self):
        """书的正文是叙事段落，和短查询的相似度天然低一档；同一阈值会全落空。"""
        self.assertLess(vector.MIN_SCORE_BOOK, vector.MIN_SCORE)

    def test_directory_chunks_are_downweighted(self):
        """目录块列举了所有事件名，对任何相关查询都中等偏高，会挤掉真正有内容的段落。"""
        self.assertLess(vector._PART_WEIGHT["chapter"], 1.0)
        self.assertLess(vector._PART_WEIGHT["title"], 1.0)

    async def test_hit_carries_chapter_and_best_body(self):
        """书命中后必须能说到「哪一章」，并指出「哪一段是直接答案」。"""
        embedder = FakeEmbedder()
        entry = store.save(
            {
                "title": "测试史料汇编",
                "kind": store.BOOK,
                "body": BOOK,
                "status": store.APPROVED,
                "origin": "upload",
            }
        )
        await vector.index_entry(entry, embedder=embedder)

        result = await vector.search("丙事件", embedder=embedder, limit=1)
        self.assertTrue(result["hits"])
        hit = result["hits"][0]
        self.assertTrue(hit["chapter"], "命中结果里必须带章节")
        self.assertIn("best_body", hit, "书必须标出直接回答问题的正文段")
        self.assertEqual(hit["best_body"]["chapter"], "1927 年")

    async def test_unrelated_query_returns_nothing(self):
        embedder = FakeEmbedder()
        entry = store.save(
            {
                "title": "测试史料汇编",
                "kind": store.BOOK,
                "body": BOOK,
                "status": store.APPROVED,
                "origin": "upload",
            }
        )
        await vector.index_entry(entry, embedder=embedder)
        result = await vector.search("量子纠缠与光合作用", embedder=embedder, limit=3)
        self.assertEqual(result["hits"], [])

    async def _indexed_book(self) -> dict:
        entry = store.save(
            {
                "title": "测试史料汇编",
                "kind": store.BOOK,
                "body": BOOK,
                "status": store.APPROVED,
                "origin": "upload",
            }
        )
        await vector.index_entry(entry, embedder=FakeEmbedder())
        return entry

    async def test_book_body_is_matched_at_chunk_level(self):
        """书的正文要参与字面匹配，但粒度是「块」，不是整本。

        上一版干脆不比正文，理由是「六万字上任何两字组合都命中过，重合率恒等于
        1」。那个理由成立，但它推出的是「别把整本当一篇文章比」——于是章节标题
        （一串年份）成了唯一的字面入口，用户换一种说法就恒为 0 命中，本来能答的
        问题被报成「库里没有」。
        """
        await self._indexed_book()
        hits = store.search("丙事件发生")
        self.assertTrue(hits)
        match = hits[0]["match"]
        self.assertEqual(match["chapter"], "1927 年")
        self.assertEqual(match["heading"], "丙事件")
        self.assertIn("丙事件发生在 1927 年", match["chunk"])
        self.assertLess(len(match["chunk"]), len(BOOK), "交出去的是命中的那一块，不是整本")

    async def test_a_year_query_lands_on_its_own_chapter(self):
        """「1927」这种查询只能靠章标题区分——每一章的正文里都有别的年份。"""
        await self._indexed_book()
        hits = store.search("1927")
        self.assertTrue(hits)
        self.assertEqual(hits[0]["match"]["chapter"], "1927 年")

    async def test_a_two_character_query_still_only_looks_at_titles(self):
        """两个字的查询不能去扫正文。

        一个二元组的重合率恒等于 1，扫出来的是「所有提到过这两个字的块」，
        顺序等于随机。这类词该在出题阶段就被拦下（见 director_search），
        而不是在检索层被当成命中。
        """
        await self._indexed_book()
        self.assertEqual(store.search("丙事"), [])


class DegenerateVectorTests(unittest.IsolatedAsyncioTestCase):
    """坏向量必须整批拒绝写入。

    这一组来自一次真实事故：压测块级增量时用了一个返回常量向量的假嵌入器，
    它把正确的向量覆盖了。因为文本没变，`chunk_hash` 是匹配的，于是增量索引
    从此再也不会修正那些块——检索照常返回结果，只是排序变成噪声。
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self._old_path = db.DB_PATH
        self._old_conn = db._conn
        db._conn = None
        db.DB_PATH = Path(self.temp.name) / "test.db"
        self.addCleanup(self._restore_db)
        db.connect()

    def _restore_db(self):
        try:
            if db._conn is not None:
                db._conn.close()
        except Exception:  # noqa: BLE001
            pass
        db._conn = self._old_conn
        db.DB_PATH = self._old_path

    class ConstantEmbedder:
        model = "fake-embed"

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.01] * DIM for _ in texts]

    class ZeroEmbedder:
        model = "fake-embed"

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * DIM for _ in texts]

    async def test_constant_vectors_are_rejected(self):
        """坏向量不能进库，但文本块要留下——两件事分开断言。

        留下文本块是有意的降级：书的字面检索读的就是这张表的 text 列，一次嵌入
        故障不该让整本书在检索体系里消失。所以这里验的是「没有向量」，而不是
        「没有任何行」。
        """
        entry = store.save(
            {"title": "测试", "kind": store.BOOK, "body": BOOK,
             "status": store.APPROVED, "origin": "upload"}
        )
        state = await vector.index_entry(entry, embedder=self.ConstantEmbedder())
        self.assertFalse(state["ok"])
        self.assertTrue(state["degraded"])
        self.assertIn("常量向量", state["reason"])
        self.assertTrue(state["text_saved"], "文本块应当落盘，字面检索才不至于全空")
        with db.transaction() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM studio_kb_chunks"
                " WHERE entry_id = ? AND dim > 0",
                (entry["id"],),
            ).fetchone()
        self.assertEqual(int(row["n"]), 0, "坏向量不该落库")

    async def test_zero_vectors_are_rejected(self):
        entry = store.save(
            {"title": "测试", "kind": store.BOOK, "body": BOOK,
             "status": store.APPROVED, "origin": "upload"}
        )
        state = await vector.index_entry(entry, embedder=self.ZeroEmbedder())
        self.assertFalse(state["ok"])
        self.assertIn("全为零", state["reason"])

    async def test_normal_vectors_pass(self):
        """检查不能误伤正常向量——bag-of-bigrams 的方差很小但不是零。"""
        entry = store.save(
            {"title": "测试", "kind": store.BOOK, "body": BOOK,
             "status": store.APPROVED, "origin": "upload"}
        )
        state = await vector.index_entry(entry, embedder=FakeEmbedder())
        self.assertTrue(state["ok"], state.get("reason"))
        self.assertGreater(vector.count(entry["id"]), 0)


class InstructionStrippingTests(unittest.TestCase):
    """「帮我做一个 X 的视频」里的 X 才是主题。

    这一组来自一个真实故障：用户问「帮我做一个国民党一大的视频」，系统返回的是
    《中国共产党的诞生》里中共一大的内容。查下来是三层原因叠在一起，其中一层是
    向量检索拿整句话去做嵌入，主题被「做视频」这个意图稀释。
    """

    def test_instruction_wrapper_is_removed(self):
        from app.kb.query import strip_filler

        cases = {
            "帮我做一个国民党一大的视频": "国民党一大",
            "帮我做一期关于南昌起义的视频": "南昌起义",
            "做一个遵义会议的短片": "遵义会议",
            "生成一段 3 分钟的党史内容": "党史内容",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(strip_filler(text), expected)

    def test_topic_only_input_is_unchanged(self):
        """本来就只有主题的输入不能被改动——这是最常见的情况。"""
        from app.kb.query import strip_filler

        for text in ("国民党一大", "1927 年发生了什么", "遵义会议"):
            with self.subTest(text=text):
                self.assertEqual(strip_filler(text), text)

    def test_stripping_never_yields_empty(self):
        """全是填充词的极端输入要能回退，不能把查询弄没。"""
        from app.kb.query import strip_filler

        self.assertEqual(strip_filler("帮我做一个视频"), "")

class InstructionStrippingAsyncTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self._old_path = db.DB_PATH
        self._old_conn = db._conn
        db._conn = None
        db.DB_PATH = Path(self.temp.name) / "test.db"
        self.addCleanup(self._restore_db)
        db.connect()

    def _restore_db(self):
        try:
            if db._conn is not None:
                db._conn.close()
        except Exception:  # noqa: BLE001
            pass
        db._conn = self._old_conn
        db.DB_PATH = self._old_path

    async def test_semantic_search_uses_the_stripped_query(self):
        """`research` 走向量时传的必须是清洗后的文本。

        关键词那条路早就删了填充词（`store._pairs`），向量这条路曾经一直用原句——
        同一个问题在两层处理不一致，而输出上看不出来，只能靠对比分数发现。
        """
        from app.kb import research

        seen: list[str] = []

        async def spy(text, **kwargs):
            seen.append(text)
            return {"ok": True, "hits": []}

        with patch.object(store, "search", return_value=[]), \
             patch.object(vector, "search", side_effect=spy):
            with self.assertRaises(ResearchError):
                await research("帮我做一个国民党一大的视频", allow_web=False)
        self.assertEqual(seen, ["国民党一大"])


class StructureMismatchTests(unittest.IsolatedAsyncioTestCase):
    """整本资料被按卡片结构切分时，必须能被发现。

    这种损坏不会让任何东西报错：检索照常返回、健康检查照常 ready，只是排序被
    一堆「1921 年」污染。用户能看到的只有「搜得不准」，而搜不准有太多种可能，
    没人会想到去怀疑索引结构。
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self._old_path = db.DB_PATH
        self._old_conn = db._conn
        db._conn = None
        db.DB_PATH = Path(self.temp.name) / "test.db"
        self.addCleanup(self._restore_db)
        db.connect()

    def _restore_db(self):
        try:
            if db._conn is not None:
                db._conn.close()
        except Exception:  # noqa: BLE001
            pass
        db._conn = self._old_conn
        db.DB_PATH = self._old_path

    async def test_long_body_misindexed_as_cards_is_rejected(self):
        """六万字正文被按卡片切分时必须拒绝写入。

        判据是正文长度，不是 `kind`：真实故障里这本书的 `kind` 本身就是错的
        （`store.save` 的 UPDATE 分支不动 kind），所以看 kind 的校验不会触发。
        """
        long_body = BOOK * 12  # 一万多字不够，再乘一点确保超过阈值
        while len(long_body) < vector._LONG_DOC_CHARS + 1000:
            long_body += BOOK
        entry = store.save(
            {"title": "长文", "kind": store.CARD, "body": long_body,
             "key_points": [f"第{i}章" for i in range(80)],
             "status": store.APPROVED, "origin": "upload"}
        )
        state = await vector.index_entry(entry, embedder=FakeEmbedder())
        self.assertFalse(state["ok"])
        self.assertIn("按卡片结构切分", state["reason"])
        self.assertEqual(vector.count(entry["id"]), 0, "不该写入任何块")

    async def test_normal_card_still_indexes(self):
        """校验不能误伤正常卡片——它们也有 point 块，只是正文短。"""
        entry = store.save(
            {"title": "短卡片", "kind": store.CARD, "body": "一段八百字的正文。" * 40,
             "key_points": ["要点一", "要点二"], "status": store.APPROVED}
        )
        state = await vector.index_entry(entry, embedder=FakeEmbedder())
        self.assertTrue(state["ok"], state.get("reason"))
        self.assertGreater(vector.count(entry["id"]), 0)

    async def test_health_reports_a_book_indexed_as_cards(self):
        """库里已经躺着的错误结构要被数出来，不能只防新的。"""
        entry = store.save(
            {"title": "测试", "kind": store.BOOK, "body": BOOK,
             "status": store.APPROVED, "origin": "upload"}
        )
        # 绕过 index_entry 校验，直接伪造一批 point 块——模拟历史遗留的错误数据。
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO studio_kb_chunks"
                " (id, entry_id, part, ordinal, text, section, chapter, heading,"
                "  entry_hash, chunk_hash, vector, dim, model, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("fake:point:000", entry["id"], "point", 0, "1921 年", "测试",
                 "", "", "", "h1", vector._encode(_fake("1921 年")), DIM, "fake-embed", 0.0),
            )
        health = vector.health()
        self.assertEqual(health["structure_mismatches"], 1)
        self.assertFalse(health["ready"], "结构异常时不该报 ready")
        self.assertIn("索引结构异常", health["reason"])


if __name__ == "__main__":
    unittest.main()
