"""素材没找到时那张问卡：选项从哪来、回答怎么变成一次取材。

规则是从讨论里定下来的：

  · 有主题但本地没有 → 列出「贴正文 / 联网 / 飞书 / 不做了」，由用户点。
  · 选项按能力生成：没连飞书就不出现飞书，不写「敬请期待」。
  · 选的那条路也失败 → 回到同一张卡，把原因写上，让他再决定（不限次数）。
  · 「先不做了」是取消，不是失败。
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from app.doc import lookup
from app.doc.intake import Intake


def missed(**kw) -> Intake:
    base = dict(
        intent="topic_request", needs_knowledge=True, topic="习近平",
        desired_seconds=60, ready=False, reason="本地资料库 没有「习近平」的可用材料",
    )
    base.update(kw)
    return Intake(**base)


class OptionTests(unittest.TestCase):
    def test_feishu_is_absent_when_it_is_not_connected(self):
        with patch.object(lookup, "feishu_available", return_value=False):
            ids = [o["id"] for o in lookup.options(topic="习近平")]
        self.assertEqual(ids, ["provide", "web", "stop"])

    def test_feishu_appears_when_it_can_actually_be_used(self):
        with patch.object(lookup, "feishu_available", return_value=True):
            ids = [o["id"] for o in lookup.options(topic="习近平")]
        self.assertEqual(ids, ["provide", "web", "feishu", "stop"])

    def test_no_topic_means_no_web_option(self):
        """没有主题时「联网搜」没有意义：不知道该搜什么。"""
        ids = [o["id"] for o in lookup.options(topic="", feishu=False)]
        self.assertEqual(ids, ["provide", "stop"])

    def test_provide_is_the_only_option_that_needs_text(self):
        needs = [o for o in lookup.options(topic="习近平") if o.get("needs_text")]
        self.assertEqual([o["id"] for o in needs], ["provide"])


class QuestionTests(unittest.TestCase):
    def test_missing_material_names_the_topic_and_what_was_tried(self):
        intake = missed(tool_calls=[{"name": "search_knowledge", "args": {"allow_web": False}, "failed": True}])
        card = lookup.for_material(intake)
        self.assertIn("习近平", card["title"])
        self.assertEqual([t["source"] for t in card["tried"]], ["本地资料库"])
        self.assertFalse(card["tried"][0]["ok"])

    def test_broken_retrieval_is_not_reported_as_no_results(self):
        """检索挂了和「库里没有」是两件事：一个要去修配置，一个要换来源。"""
        intake = missed(error={"error_type": "local_error", "message": "向量化失败"})
        card = lookup.for_material(intake)
        self.assertIn("检索没跑通", card["title"])
        self.assertNotIn("现成材料", card["title"])

    def test_missing_topic_offers_only_provide_and_stop(self):
        intake = Intake(intent="insufficient", ready=False, reason="原文只有制作要求")
        card = lookup.for_missing_topic(intake)
        self.assertEqual([o["id"] for o in card["options"]], ["provide", "stop"])
        self.assertIn("没看出要做什么", card["title"])

    def test_reason_is_trimmed_to_one_sentence(self):
        """检索失败的原因常常拖着「任务影响／下一步」，卡上装不下。"""
        intake = missed(reason="本地没有可用材料；任务影响：已在生成前暂停；建议：请重试")
        card = lookup.for_material(intake)
        self.assertNotIn("任务影响", card["reason"])


class ResolveTests(unittest.IsolatedAsyncioTestCase):
    async def test_provided_text_becomes_the_material(self):
        intake = missed()
        resolved = await lookup.resolve(
            intake, {"choice": "provide", "text": "2012 年 11 月，习近平当选总书记。"}
        )
        self.assertTrue(resolved.ready)
        self.assertEqual(resolved.source, "user")
        self.assertEqual(resolved.material(), "2012 年 11 月，习近平当选总书记。")
        self.assertFalse(resolved.needs_knowledge, "拿到了正文就不该再去查库")
        # 用户素材不进库，这条边界要在卡片上也说得清。
        self.assertIn("未写入知识库", resolved.entry["note"])

    async def test_empty_text_asks_again_instead_of_guessing(self):
        with self.assertRaises(lookup.AskUser) as caught:
            await lookup.resolve(missed(), {"choice": "provide", "text": "   "})
        self.assertIn("还没收到正文", caught.exception.question["title"])

    async def test_stop_is_a_cancellation_not_a_failure(self):
        with self.assertRaises(lookup.StopRun):
            await lookup.resolve(missed(), {"choice": "stop"})

    async def test_unknown_choice_asks_again(self):
        with self.assertRaises(lookup.AskUser) as caught:
            await lookup.resolve(missed(), {"choice": "dingtalk"})
        self.assertIn("没看懂这个选择", caught.exception.question["title"])

    async def test_web_failure_returns_to_the_same_question_with_the_reason(self):
        intake = missed()
        with patch.object(
            lookup, "_search_knowledge_once",
            new_callable=AsyncMock,
            return_value=(None, {"reason": "HTTP 404，已换源仍未找到"}),
        ):
            with self.assertRaises(lookup.AskUser) as caught:
                await lookup.resolve(intake, {"choice": "web"})
        question = caught.exception.question
        self.assertIn("404", question["reason"])
        self.assertIn("web", [o["id"] for o in question["options"]], "还能再试一次")

    async def test_web_hit_is_accepted(self):
        intake = missed()
        entry = {"id": "e1", "title": "习近平", "body": "权威正文", "sources": [{"url": "https://x"}]}
        with patch.object(
            lookup, "_search_knowledge_once",
            new_callable=AsyncMock,
            return_value=(entry, {"found": True, "source": "web"}),
        ):
            resolved = await lookup.resolve(intake, {"choice": "web"})
        self.assertTrue(resolved.ready)
        self.assertEqual(resolved.source, "web")


class SnapshotTests(unittest.TestCase):
    def test_snapshot_roundtrips_through_json(self):
        import json

        original = missed(tool_calls=[{"name": "search_knowledge", "args": {"query": "习近平"}}])
        restored = lookup.restore(json.loads(json.dumps(lookup.snapshot(original), ensure_ascii=False)))
        self.assertEqual(restored.intent, original.intent)
        self.assertEqual(restored.topic, original.topic)
        self.assertEqual(restored.tool_calls, original.tool_calls)
        self.assertFalse(restored.ready)


if __name__ == "__main__":
    unittest.main()
