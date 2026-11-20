"""知识库语义索引的测试。

全部使用假嵌入器：测试不该依赖网络，也不该消耗真实额度。假嵌入器把字符二元组
散列到固定维度的桶里，因此「用词相近 → 余弦相近」这个性质仍然成立——足以验证
排序、阈值、聚合这些逻辑，而不必真的调用一次模型。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app import db
from app.kb import research, store, vector
from app.kb import ResearchError


DIM = 96


def _fake(vector_text: str) -> list[float]:
    """Stable bag-of-bigrams embedding. Deterministic across processes."""
    out = [0.0] * DIM
    cleaned = "".join(ch for ch in vector_text if not ch.isspace())
    for index in range(max(1, len(cleaned) - 1)):
        gram = cleaned[index : index + 2]
        slot = sum(ord(ch) for ch in gram) % DIM
        out[slot] += 1.0
    if not any(out):
        out[0] = 1.0
    return out


class FakeEmbedder:
    model = "fake-embed"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [_fake(text) for text in texts]


def _entry(**overrides) -> dict:
    base = {
        "title": "中国共产党的诞生",
        "category": "历史知识",
        "period": "1921年",
        "body": (
            "1921年7月，中国共产党第一次全国代表大会在上海法租界望志路召开，"
            "后因受到干扰转移到浙江嘉兴南湖的游船上继续举行。大会通过了党的第一个纲领，"
            "确定党的名称为中国共产党，选举产生了党的领导机构。这次会议的召开标志着"
            "中国共产党正式成立，中国革命的面貌从此焕然一新。南湖红船也因此成为"
            "中国共产党诞生地的象征。"
        ),
        "key_points": ["中共一大在上海召开，后转移至嘉兴南湖", "大会宣告中国共产党正式成立"],
        "sources": [
            {"kind": "参考资料", "org": "中国政府网", "title": "中国共产党成立", "url": "https://www.gov.cn/xinwen/2021-06/28/content_1.htm"}
        ],
        "status": store.APPROVED,
        "origin": "manual",
        "query": "中国共产党的诞生",
        "note": "",
    }
    base.update(overrides)
    return base


class KbVectorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self._old_path = db.DB_PATH
        self._old_conn = db._conn
        db._conn = None
        db.DB_PATH = Path(self.temp.name) / "test.db"
        self.addCleanup(self._restore_db)
        db.connect()
        self.embedder = FakeEmbedder()

    def _restore_db(self):
        try:
            if db._conn is not None:
                db._conn.close()
        except Exception:  # noqa: BLE001 - cleanup must not mask the real failure
            pass
        db._conn = self._old_conn
        db.DB_PATH = self._old_path

    # ---------------------------------------------------------------- 存储格式

    async def test_vectors_are_stored_as_blob_not_json(self):
        """JSON 是 27 KB/块，BLOB 是 8 KB/块——书级规模下这是 26 倍检索差距。"""
        entry = store.save(_entry())
        await vector.index_entry(entry, embedder=self.embedder)

        conn = db.connect()
        rows = conn.execute(
            "SELECT typeof(vector) AS kind, length(vector) AS size, dim"
            " FROM studio_kb_chunks WHERE entry_id = ? LIMIT 1",
            (entry["id"],),
        ).fetchall()
        self.assertTrue(rows)
        self.assertEqual(rows[0]["kind"], "blob")
        # float32 × 96 维 = 384 字节，JSON 至少要翻好几倍
        self.assertEqual(rows[0]["size"], DIM * 4)
        self.assertEqual(rows[0]["dim"], DIM)

    async def test_legacy_json_rows_are_still_readable(self):
        """改存储格式不该要求把所有条目重新嵌入一遍。"""
        entry = store.save(_entry())
        legacy = _fake("中国共产党的诞生")  # 用同一套散列，保证是真实会命中的向量
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO studio_kb_chunks"
                " (id, entry_id, part, ordinal, text, section, entry_hash, vector, dim, model, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    f"{entry['id']}:title:000", entry["id"], "title", 0,
                    entry["title"], entry["title"], vector.entry_hash(entry),
                    json.dumps(legacy), DIM, "legacy", 0.0,
                ),
            )

        result = await vector.search("中国共产党的诞生", embedder=self.embedder, min_score=0.1)
        self.assertTrue(result["hits"], "旧格式的向量必须仍然可被检索")

    def test_compact_rewrites_legacy_rows_without_calling_the_model(self):
        entry = store.save(_entry())
        legacy = _fake("中国共产党")
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO studio_kb_chunks"
                " (id, entry_id, part, ordinal, text, section, entry_hash, vector, dim, model, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    f"{entry['id']}:title:000", entry["id"], "title", 0,
                    entry["title"], entry["title"], "h",
                    json.dumps(legacy), DIM, "legacy", 0.0,
                ),
            )

        # 迁移不该碰嵌入器：这里把 from_settings 改成会抛错的桩，一旦被调用就失败。
        class Exploding:
            @classmethod
            def from_settings(cls, *args, **kwargs):
                raise AssertionError("格式迁移不应该调用嵌入接口")

        with patch.object(vector.Embedder, "from_settings", Exploding.from_settings):
            result = vector.compact_vectors()

        self.assertEqual(result["converted"], 1)
        conn = db.connect()
        kind = conn.execute(
            "SELECT typeof(vector) AS t FROM studio_kb_chunks WHERE id = ?",
            (f"{entry['id']}:title:000",),
        ).fetchone()["t"]
        self.assertEqual(kind, "blob")

    # ---------------------------------------------------------------- 切块

    def test_split_entry_separates_title_points_and_body(self):
        parts = vector.split_entry(_entry())
        kinds = [p["part"] for p in parts]
        self.assertEqual(kinds[0], "title")
        self.assertEqual(kinds.count("point"), 2)
        self.assertGreaterEqual(kinds.count("body"), 1)
        # 标题块带上分类与年代，主题检索才有东西可匹配。
        self.assertIn("1921年", parts[0]["text"])

    def test_embed_text_carries_the_title_but_stored_text_does_not(self):
        """正文段落常不提主题名，嵌入时要补上；引用时不该重复。"""
        parts = vector.split_entry(_entry())
        body = [p for p in parts if p["part"] == "body"][0]
        self.assertIn("中国共产党的诞生", body["embed_text"])
        self.assertNotIn("《", body["text"])

    def test_entry_hash_ignores_unindexed_fields(self):
        entry = _entry()
        baseline = vector.entry_hash(entry)
        self.assertEqual(baseline, vector.entry_hash({**entry, "uses": 99}))
        self.assertEqual(baseline, vector.entry_hash({**entry, "status": store.PENDING}))
        self.assertNotEqual(baseline, vector.entry_hash({**entry, "body": entry["body"] + "补充。"}))

    # ---------------------------------------------------------------- 写入

    async def test_index_is_idempotent_until_content_changes(self):
        entry = store.save(_entry())
        first = await vector.index_entry(entry, embedder=self.embedder)
        self.assertTrue(first["ok"])
        chunks = first["chunks"]
        self.assertGreater(chunks, 0)

        again = await vector.index_entry(store.get(entry["id"]), embedder=self.embedder)
        self.assertTrue(again.get("skipped"), "内容没变不该重复嵌入")

        changed = store.save({**store.get(entry["id"]), "body": entry["body"] + "补充材料。"})
        reindexed = await vector.index_entry(changed, embedder=self.embedder)
        self.assertTrue(reindexed["ok"])
        self.assertFalse(reindexed.get("skipped"))
        self.assertEqual(vector.count(entry["id"]), reindexed["chunks"])

    async def test_index_reports_degradation_instead_of_raising(self):
        """没有向量模型是「这一层没生效」，不是「整个知识库坏了」。

        文本块仍然要落盘：字面检索读的就是这张表，一块都不写等于这本书在检索
        体系里不存在——而它明明在库里、正文也完整的。所以这里断言的是「没有
        向量」，而不是「没有行」。
        """
        entry = store.save(_entry())
        with patch.object(vector.Embedder, "from_settings", return_value=None):
            result = await vector.index_entry(entry)
        self.assertFalse(result["ok"])
        self.assertTrue(result["degraded"])
        self.assertTrue(result["text_saved"])
        with db.transaction() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM studio_kb_chunks"
                " WHERE entry_id = ? AND dim > 0",
                (entry["id"],),
            ).fetchone()
        self.assertEqual(int(row["n"]), 0)

    async def test_remove_entry_clears_vectors(self):
        entry = store.save(_entry())
        await vector.index_entry(entry, embedder=self.embedder)
        self.assertGreater(vector.count(entry["id"]), 0)
        removed = vector.remove_entry(entry["id"])
        self.assertGreater(removed, 0)
        self.assertEqual(vector.count(entry["id"]), 0)

    # ---------------------------------------------------------------- 召回

    async def test_search_aggregates_per_entry_and_keeps_best_chunk(self):
        entry = store.save(_entry())
        await vector.index_entry(entry, embedder=self.embedder)

        result = await vector.search("嘉兴南湖的船", embedder=self.embedder, min_score=0.1)
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["hits"]), 1, "同一条目只应出现一次")
        self.assertEqual(result["hits"][0]["entry_id"], entry["id"])
        self.assertIn(result["hits"][0]["part"], {"title", "point", "body"})

    async def test_search_threshold_rejects_unrelated_topics(self):
        entry = store.save(_entry())
        await vector.index_entry(entry, embedder=self.embedder)

        unrelated = await vector.search("量子纠缠原理与量子计算", embedder=self.embedder, min_score=0.9)
        self.assertEqual(unrelated["hits"], [])

    async def test_rejected_entry_is_not_recallable(self):
        entry = store.save(_entry())
        await vector.index_entry(entry, embedder=self.embedder)
        store.decide(entry["id"], store.REJECTED)

        result = await vector.search("中国共产党的诞生", embedder=self.embedder, min_score=0.1)
        self.assertEqual(result["hits"], [], "已放弃的条目不能被语义召回")

    async def test_stale_dimension_is_reported_not_silently_empty(self):
        """换过嵌入模型时，结果为空是「索引过期」而不是「库里没有」。"""
        entry = store.save(_entry())
        await vector.index_entry(entry, embedder=self.embedder)

        class OtherDim:
            model = "other"

            async def embed(self, texts):
                return [[1.0, 0.0, 0.0] for _ in texts]

        result = await vector.search("中国共产党的诞生", embedder=OtherDim(), min_score=0.1)
        self.assertEqual(result["hits"], [])
        self.assertTrue(result.get("stale"))
        self.assertIn("重建索引", result["reason"])

    # ---------------------------------------------------------------- 融合

    def test_rrf_prefers_items_ranked_high_by_both_lists(self):
        topic = ["a", "b", "c"]
        semantic = ["b", "a", "d"]
        fused = vector.rrf(topic, semantic)
        self.assertEqual(fused[:2], ["a", "b"])
        self.assertIn("d", fused)

    # ---------------------------------------------------------------- 闸门

    async def test_semantic_hit_still_passes_through_the_authority_gate(self):
        """语义召回不能绕过闸门：没有白名单来源的条目，两条路都不许当证据。

        这是这次改造里最需要钉住的一条。语义层返回的是「相似的段落」，如果直接
        把它交给下游，就等于给未核实内容开了一条侧门——而侧门只在特定召回路径
        上打开，最不容易在测试里被发现。
        """
        entry = store.save(
            _entry(
                title="网传说法汇编",
                sources=[],
                origin="web",
                origin_note="",
            )
        )
        await vector.index_entry(entry, embedder=self.embedder)

        with patch.object(store, "search", return_value=[]), \
             patch.object(vector, "search", new_callable=AsyncMock) as semantic:
            semantic.return_value = {
                "ok": True,
                "hits": [
                    {
                        "entry_id": entry["id"],
                        "score": 0.88,
                        "text": entry["body"][:60],
                        "part": "body",
                        "section": entry["title"],
                    }
                ],
            }
            with self.assertRaises(ResearchError) as caught:
                await research("党的诞生地", allow_web=False)
        self.assertEqual(caught.exception.error_type, "local_not_found")
        # 报错里要带上语义层的情况，否则「真没有」和「索引挂了」看起来一样。
        self.assertIn("语义检索", str(caught.exception))

    async def test_semantic_hit_is_used_when_it_passes_the_gate(self):
        entry = store.save(_entry())
        await vector.index_entry(entry, embedder=self.embedder)

        with patch.object(store, "search", return_value=[]), \
             patch.object(vector, "search", new_callable=AsyncMock) as semantic:
            semantic.return_value = {
                "ok": True,
                "hits": [
                    {
                        "entry_id": entry["id"],
                        "score": 0.71,
                        "text": "后转移至嘉兴南湖的游船上继续举行",
                        "part": "body",
                        "section": entry["title"],
                    }
                ],
            }
            result = await research("南湖那条船", allow_web=False)

        self.assertEqual(result["source"], "kb")
        self.assertEqual(result["retrieval"], "semantic")
        self.assertEqual(result["entry"]["id"], entry["id"])
        self.assertIn("语义召回", result["trace"])

    async def test_topic_hit_short_circuits_before_any_embedding_call(self):
        """命中主题检索时必须不再嵌入查询——这条快路径是知识库的核心价值。"""
        entry = store.save(_entry())
        await vector.index_entry(entry, embedder=self.embedder)

        with patch.object(vector, "search", new_callable=AsyncMock) as semantic:
            result = await research("中国共产党的诞生", allow_web=False)
        self.assertEqual(result["retrieval"], "topic")
        semantic.assert_not_awaited()

    async def test_web_branch_indexes_the_entry_it_just_built(self):
        """联网建出来的条目要立刻可被语义召回。

        隔一段时间再建索引，就会存在「条目已经在库里、但语义检索搜不到」的窗口。
        这类不一致最难被想到去查，因为界面上一切正常。
        """
        from types import SimpleNamespace

        from app.kb import websearch
        from app import kb as kb_pkg

        draft = SimpleNamespace(
            title="遵义会议",
            category="历史知识",
            period="1935年",
            body=(
                "1935年1月，中共中央在贵州遵义召开政治局扩大会议，纠正了此前军事指挥上的"
                "错误，改组了中央领导机构，确立了毛泽东在党中央和红军的领导地位。"
            ),
            key_points=["纠正了军事指挥上的错误", "改组了中央领导机构", "确立新的领导核心"],
        )
        hit = websearch.SearchResult(title="遵义会议", url="https://www.gov.cn/zzhy.htm")

        with patch.object(kb_pkg, "search_available", return_value=True), \
             patch.object(kb_pkg, "search", new_callable=AsyncMock, return_value=[hit]), \
             patch.object(kb_pkg, "_fetch", new_callable=AsyncMock, return_value="1935年1月遵义会议召开。"), \
             patch.object(kb_pkg, "summarize", new_callable=AsyncMock, return_value=draft), \
             patch.object(vector.Embedder, "from_settings", return_value=self.embedder):
            result = await research("遵义会议", allow_web=True)

        self.assertEqual(result["source"], "web")
        self.assertTrue(result["indexed"], result.get("index_note"))
        self.assertGreater(vector.count(result["entry"]["id"]), 0)


if __name__ == "__main__":
    unittest.main()
