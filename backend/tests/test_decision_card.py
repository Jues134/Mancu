"""导演决策卡的三条硬规则。

这张卡存在的理由是把「输入判定」从一句结论变成可以核对的过程。它最容易出的问题
不是长得不好看，而是**看起来可信**：一句编造的引用、一个第一段就下的结论，读起来
都比空白更像在思考。
"""
from __future__ import annotations

import unittest

from app.doc.decision import build
from app.doc.intake import Intake


TOPIC = "帮我做一个国民党一大的视频，1 分钟左右"
MATERIAL = (
    "1935 年 1 月，遵义会议召开。这次会议纠正了之前错误的军事指挥，"
    "确立了毛泽东同志在党中央和红军的领导地位。"
)


def card(text: str, **kw) -> dict:
    intake = Intake(**kw)
    return build(text=text, intake=intake)


def topic_card(**kw) -> dict:
    base = dict(
        intent="topic_request", needs_knowledge=True, topic="国民党一大",
        desired_seconds=60, ready=False, reason="只给了主题，没有正文",
    )
    base.update(kw)
    return card(TOPIC, **base)


class ReceivedSectionTests(unittest.TestCase):
    """第一段只分诊，不下结论。"""

    def test_first_section_never_states_the_verdict(self):
        received = topic_card()["received"]
        blob = received["line"] + received["triage"]
        # 「是素材还是主题」属于判定段；第一段说了，一张卡上就有两处判定。
        for word in ("主题请求", "素材", "需要查资料", "可以直接改编"):
            self.assertNotIn(word, blob, f"第一段出现了结论词：{word}")

    def test_first_section_says_what_arrived(self):
        received = topic_card()["received"]
        self.assertIn("20 字", received["line"])
        self.assertIn("无附件", received["line"])

    def test_link_input_is_triaged_to_parse_first(self):
        received = card(
            "根据 https://acme.feishu.cn/docx/abc123 做一个视频",
            intent="feishu_request", needs_external_material=True, ready=False,
            source_query="abc123",
        )["received"]
        self.assertIn("链接", received["triage"])


class QuoteRuleTests(unittest.TestCase):
    """带「」的引用必须是原文子串——不是就丢掉，而不是改写成看起来像原文。"""

    def test_every_quote_is_a_substring_of_the_input(self):
        for payload in (
            topic_card(),
            card(MATERIAL, intent="material", ready=True, reason="原文含事件叙述"),
            card("做一个关于南昌起义的短片，30 秒", intent="topic_request",
                 needs_knowledge=True, topic="南昌起义", desired_seconds=30),
        ):
            for row in payload["understanding"]:
                if not row.get("quote"):
                    continue
                self.assertIn(
                    row["quote"], payload["source"]["text"],
                    f"{row['label']} 的引用不是原文片段：{row['quote']}",
                )

    def test_topic_that_is_not_in_the_text_gets_no_quote(self):
        # 模型归纳出来的主题可能原封不动不在原文里。可以显示它，但不能加引号
        # 冒充引用。
        payload = card(
            "讲讲那次在广州开的会", intent="topic_request", needs_knowledge=True,
            topic="国民党一大",
        )
        row = next(r for r in payload["understanding"] if r["label"] == "题材")
        self.assertEqual(row["value"], "国民党一大")
        self.assertEqual(row["quote"], "")

    def test_missing_topic_is_stated_not_hidden(self):
        row = next(
            r for r in card(MATERIAL, intent="material", ready=True)["understanding"]
            if r["label"] == "题材"
        )
        self.assertEqual(row["value"], "未从原文识别到")


class VerdictTests(unittest.TestCase):
    def test_verdict_follows_the_intent(self):
        self.assertTrue(topic_card()["verdict"]["need_kb"])
        material = card(MATERIAL, intent="material", ready=True)
        self.assertFalse(material["verdict"]["need_kb"])
        self.assertIn("不需要查资料", material["verdict"]["title"])

    def test_basis_separates_measurement_from_the_model_claim(self):
        basis = topic_card()["verdict"]["basis"]
        sources = [item["from"] for item in basis]
        self.assertEqual(sources[0], "程序测量", "测量放第一条：它不依赖模型")
        self.assertIn("规则判断", sources)
        self.assertIn("导演自述", sources)
        # 测量那一句必须是真的数出来的，不是模型说的。
        self.assertIn("20 字", basis[0]["text"])

    def test_explicit_no_research_is_quoted_as_a_rule(self):
        payload = card(
            "1935 年 1 月遵义会议召开，不要查资料。", intent="material", ready=True,
        )
        rules = [i for i in payload["verdict"]["basis"] if i["from"] == "规则判断"]
        self.assertIn("不要查资料", rules[0]["text"])

    def test_next_step_reports_the_hit_and_stays_honest_when_it_fails(self):
        hit = topic_card(entry={"title": "中国共产党百年大事记"}, source="kb")
        self.assertIn("已命中《中国共产党百年大事记》", hit["next"]["line"])

        # 没查到时不能说「不能开拍」，也不能说「补一句具体的人或事」——用户已经
        # 给了主题。它要说明的是「已停下，等你选从哪儿找」。
        miss = card(
            TOPIC, intent="topic_request", needs_knowledge=True, topic="国民党一大",
            ready=False, error={"message": "本地未命中"},
        )
        self.assertIn("等你选从哪儿找", miss["next"]["line"])
        self.assertNotIn("不能开拍", miss["next"]["line"])


class SearchingStateTests(unittest.TestCase):
    """分类完成、检索未完时的中间态卡。

    检索搬进识别阶段之后，判定卡要等检索结束才「有资格」下结论——但用户不能
    在那之前先看一串工具调用。中间态卡就是开头那一下：告诉我你把我的话读成了
    什么、现在在干什么。它绝不能把「还没查」说成「没有」。
    """

    def test_searching_card_does_not_pretend_the_library_is_empty(self):
        payload = build(
            text=TOPIC,
            intake=Intake(
                intent="topic_request", needs_knowledge=True, topic="国民党一大",
                ready=False, reason="只给了主题",
            ),
            searching=True,
        )
        self.assertIn("正在检索", payload["verdict"]["title"])
        self.assertIn("更新", payload["next"]["hint"])
        # 「没有」「等你选」都是检索结束后的结论，中间态一个都不许出现。
        self.assertNotIn("没有现成正文", payload["next"]["line"])
        self.assertNotIn("等你选从哪儿找", payload["next"]["line"])

    def test_searching_card_keeps_triage_sections(self):
        payload = build(
            text=TOPIC,
            intake=Intake(intent="topic_request", needs_knowledge=True, topic="国民党一大"),
            searching=True,
        )
        self.assertTrue(payload["verdict"]["need_kb"])
        self.assertTrue(payload["received"]["line"])
        self.assertTrue(payload["understanding"])


if __name__ == "__main__":
    unittest.main()
