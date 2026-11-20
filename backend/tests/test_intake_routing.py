"""输入路由回归：不访问真实模型、知识库数据库或生成服务。"""
from __future__ import annotations

import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from app.doc import director_search, intake, lookup, media_mode, pipeline_six
from app import kb


MATERIAL = (
    "1935 年 1 月，遵义会议召开。这次会议纠正了之前错误的军事指挥，"
    "确立了毛泽东同志在党中央和红军的领导地位，在极端危急的历史关头，"
    "挽救了党、挽救了红军、挽救了中国革命，是中国共产党历史上一个生死攸关的转折点。"
    "从此，红军逐步摆脱被动局面，踏上继续北上的征程党史学习教..."
)
TOPIC = "帮我做一个关于遵义会议的30秒视频"
MISS = {"found": False, "error_type": "local_not_found", "reason": "本地未命中"}


def classification(intent="material", topic="", seconds=0):
    return {"intent": intent, "topic": topic, "desired_seconds": seconds, "reason": "test"}


def found(source="kb"):
    return {
        "found": True, "source": source, "material_ready": True,
        "entry": {
            "id": "test-entry", "title": "遵义会议", "body": "权威正文，仅用于模拟检索。",
            "key_points": [], "status": "pending" if source == "web" else "approved",
            "sources": [{"url": "https://www.12371.cn/2021/01/01/ARTItest.shtml"}],
        },
    }


class IntakeRoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(intake, "_config", return_value=object()))
        self.model = self.stack.enter_context(patch.object(
            intake.llm_tools, "chat_json", new_callable=AsyncMock,
            return_value=classification(),
        ))
        self.tool = AsyncMock(side_effect=AssertionError("unexpected retrieval"))
        self.server = self.stack.enter_context(patch.object(
            intake, "outbound_server", return_value=SimpleNamespace(call=self.tool),
        ))
        # 出题这一步默认走空题单：它平时读的是本机 providers.json，
        # 别人克隆下来跑测试时那里是空的，`_think` 会抛 PlanError，
        # 于是「工具坏了」被误判成「规划没跑起来」。测试不该依赖本机配置。
        self.plan = self.stack.enter_context(patch.object(
            director_search, "_think", new_callable=AsyncMock,
            return_value={"questions": []},
        ))

    async def test_supplied_material_never_creates_tool_client(self):
        # Even contradictory fields from a model cannot grant tools to material.
        self.model.return_value = {**classification(), "needs_knowledge": True, "ready": False}
        result = await intake.judge(MATERIAL)
        self.assertTrue(result.ready)
        self.assertFalse(result.needs_knowledge)
        self.assertEqual(result.tool_calls, [])
        self.assertIsNone(result.entry)
        self.server.assert_not_called()
        self.assertNotIn("tools", self.model.call_args.kwargs)

    async def test_uploaded_book_is_usable_even_without_a_source_url(self):
        """整本上传的资料没有网址，但它必须能当材料用。

        这条曾经是断的：检索层为 `kind=book` 放行了白名单校验，流程层
        （`_usable_entry`）又按「sources 里必须有白名单网址」判了一遍，于是书
        收进去了、检索也命中了，界面却说「检索结果缺少可用正文或可追溯的权威
        来源」。同一份判据写两遍，结果就是一边放行、一边拦截。
        """
        book = {
            "found": True, "source": "kb", "unverified": False, "kind": "book",
            "entry": {
                "id": "b1", "title": "中国共产党百年大事记", "kind": "book",
                "body": "1924 年 1 月，国民党一大在广州召开，实行联俄、联共、扶助农工三大政策。",
                "sources": [{"kind": "上传文件", "org": "", "title": "百年大事记.md", "url": ""}],
            },
        }
        self.assertIsNotNone(intake._usable_entry(book))

        # 卡片仍然必须过白名单：放宽的边界只给书。
        card = {
            "found": True, "source": "kb", "unverified": False, "kind": "card",
            "entry": {
                "id": "c1", "title": "某摘要", "kind": "card", "body": "一段没有出处的摘要。",
                "sources": [{"kind": "参考资料", "org": "", "title": "", "url": ""}],
            },
        }
        self.assertIsNone(intake._usable_entry(card))
        card["entry"]["sources"] = [{"url": "https://www.gov.cn/xinwen/2021.htm"}]
        self.assertIsNotNone(intake._usable_entry(card))

    async def test_uploaded_book_reaches_director_without_a_website_source(self):
        """整本上传的资料必须能进生成，不能因为「来源不是网址」被拦在导演这一步。

        这条规则在代码里曾经有三份拷贝：检索层闸门、`_usable_entry`、以及
        `pipeline_six` 的导演分支。前两处修好之后，书仍然会在第三处被清空材料，
        报「MCP 返回的材料没有可追溯的权威网站来源」——检索说命中了、界面说没材料。
        """
        called = self.mock_pipeline_agents()
        self.model.return_value = classification("topic_request", "国民党一大召开")
        book_hit = {
            "found": True, "source": "kb", "unverified": False, "kind": "book",
            "locator": "1924 年 · 国民党一大召开",
            "entry": {
                "id": "book-1", "title": "中国共产党百年大事记", "kind": "book",
                "body": "1924 年 1 月，国民党一大在广州召开，实行联共政策。",
                "key_points": [], "status": "approved",
                "sources": [{"kind": "上传文件", "org": "", "title": "百年大事记.md", "url": ""}],
            },
        }
        self.tool.side_effect = [book_hit]
        result = await self.run_pipeline("帮我做一个国民党一大召开的视频。")

        self.assertIn("script_adaptor", [item[0] for item in called])
        self.assertTrue(all("1924 年 1 月" in item[1] for item in called))
        self.assertEqual(result["state"]["material_origin"], "kb")
        # 来源清单要说出「这份材料是谁给的」，而不是显示 0 条。
        self.assertEqual(len(result["state"]["knowledge"]["sources"]), 1)
        self.assertEqual(result["state"]["knowledge"]["sources"][0]["title"], "百年大事记.md")

    async def test_short_story_keeps_its_input_route(self):
        self.model.return_value = classification("story")
        result = await intake.judge("我爷爷1949年在码头扛包，那天他没回家。")
        self.assertTrue(result.ready)
        self.server.assert_not_called()

    async def test_material_with_production_request_preserves_duration(self):
        self.model.return_value = classification(seconds=30)
        result = await intake.judge("请把以下素材改成30秒视频：" + MATERIAL)
        self.assertEqual(result.desired_seconds, 30)
        self.server.assert_not_called()

    async def test_topic_local_hit_stops_before_network(self):
        self.model.return_value = classification("topic_request", "遵义会议", 30)
        self.tool.side_effect = [found()]
        order = []
        result = await intake.judge(
            TOPIC, on_classified=lambda value: order.append("classified"),
            on_tool=lambda *args: order.append("tool"),
        )
        self.assertEqual(order, ["classified", "tool"])
        self.tool.assert_awaited_once_with("search_knowledge", {"query": "遵义会议", "allow_web": False})
        self.assertEqual(result.source, "kb")
        self.assertTrue(result.ready)

    async def test_topic_queries_local_only(self):
        """识别阶段只查本地。联网要等用户在问卡上点了才发。

        以前这里会连着查一次联网：用户明明手上有材料，却要先等几十秒的抓取，
        最后拿到的还是一段网上找来的概述。

        模型调用从 1 次变成 2 次：一次分诊，一次由导演想该查什么。这个数不该再
        用「等于 1」钉住——真正要守的是「工具那条路只允许 allow_web=False」。
        """
        self.model.return_value = classification("topic_request", "遵义会议")
        self.tool.side_effect = [MISS]
        result = await intake.judge(TOPIC)
        self.assertEqual(
            [c.args[1]["allow_web"] for c in self.tool.await_args_list], [False]
        )
        self.assertFalse(result.ready)
        self.assertTrue(result.needs_knowledge, "没拿到材料也不能改判成素材")
        self.assertEqual(result.topic, "遵义会议")
        self.assertLessEqual(self.model.await_count, 2)

    async def test_web_is_used_only_when_the_user_asks(self):
        self.model.return_value = classification("topic_request", "遵义会议")
        self.tool.side_effect = [MISS]
        missed = await intake.judge(TOPIC)

        self.tool.side_effect = None
        self.tool.return_value = found("web")
        resolved = await lookup.resolve(missed, {"choice": "web"})
        self.assertEqual(
            [c.args[1]["allow_web"] for c in self.tool.await_args_list], [False, True]
        )
        self.assertTrue(resolved.ready)
        self.assertEqual(resolved.entry["status"], "pending")
        self.assertEqual(resolved.source, "web")

    async def test_web_failure_asks_again_instead_of_ending_the_run(self):
        self.model.return_value = {**classification("topic_request", "遵义会议"), "ready": True}
        self.tool.side_effect = [MISS]
        missed = await intake.judge(TOPIC)

        self.tool.side_effect = None
        self.tool.return_value = {
            "found": False, "reason": "HTTP 404，已换源仍未找到", "action": "更换主题",
        }
        with self.assertRaises(lookup.AskUser) as caught:
            await lookup.resolve(missed, {"choice": "web"})
        question = caught.exception.question
        # 失败原因要写在卡上：超时、404、反爬是三种完全不同的下一步，混成
        # 「联网失败」会让人一直点重试，而其中两种重试多少次都没用。
        self.assertIn("404", question["reason"])
        self.assertIn("web", [item["id"] for item in question["options"]], "还能再试一次")
        self.assertFalse(caught.exception.intake.ready)
        self.assertTrue(caught.exception.intake.needs_knowledge)

    async def test_unusable_or_untrusted_tool_body_cannot_start_generation(self):
        self.model.return_value = classification("topic_request", "遵义会议")
        for body, sources in [("", found()["entry"]["sources"]), ("无来源正文", [])]:
            with self.subTest(body=body):
                invalid = found("web")
                invalid["entry"].update(body=body, sources=sources)
                self.tool.side_effect = [invalid]
                result = await intake.judge(TOPIC)
                self.assertFalse(result.ready)
                self.assertIsNone(result.entry)

    async def test_local_system_error_does_not_masquerade_as_cache_miss(self):
        self.model.return_value = classification("topic_request", "遵义会议")
        self.plan.return_value = {
            "anchor": {"kind": "event", "subject": "遵义会议", "when": "1935 年"},
            "questions": [
                {"ask": "遵义会议的时间与内容", "slots": ["when"], "queries": ["遵义会议"]}
            ],
        }
        self.tool.side_effect = RuntimeError("database unavailable")
        result = await intake.judge(TOPIC)
        self.assertEqual(self.tool.await_count, 1)
        self.assertFalse(result.ready)
        self.assertIn("database unavailable", result.reason)

    async def test_insufficient_input_does_not_search(self):
        self.model.return_value = classification("insufficient")
        result = await intake.judge("帮我做个视频")
        self.assertFalse(result.ready)
        self.server.assert_not_called()

    async def test_explicit_feishu_request_uses_only_feishu_tools(self):
        self.model.return_value = {
            "intent": "feishu_request", "source_query": "Q4产品方案",
            "desired_seconds": 30, "reason": "用户明确要求读取飞书文档",
        }
        search = {
            "ok": True,
            "results": [{"title": "Q4产品方案", "url": "https://acme.feishu.cn/docx/abc", "type": "docx"}],
        }
        fetch = {
            "ok": True, "title": "Q4产品方案", "url": "https://acme.feishu.cn/docx/abc",
            "text": "这是飞书中的产品方案正文，只供本次任务使用。", "chars": 22,
        }
        self.tool.side_effect = [search, fetch]
        result = await intake.judge("用飞书里的《Q4产品方案》做一个30秒视频")
        self.assertTrue(result.ready)
        self.assertTrue(result.needs_external_material)
        self.assertFalse(result.needs_knowledge)
        self.assertEqual(result.source, "feishu")
        self.assertEqual([call.args[0] for call in self.tool.await_args_list], ["search_feishu_docs", "fetch_feishu_doc"])
        self.assertNotIn("search_knowledge", [call.args[0] for call in self.tool.await_args_list])

    async def test_feishu_miss_stops_without_falling_back_to_knowledge_or_web(self):
        self.model.return_value = {
            "intent": "feishu_request", "source_query": "不存在的方案",
            "desired_seconds": 0, "reason": "用户明确要求读取飞书文档",
        }
        self.tool.side_effect = [{"ok": True, "results": [], "reason": "飞书中未找到"}]
        result = await intake.judge("从飞书找《不存在的方案》")
        self.assertFalse(result.ready)
        self.assertIn("未找到", result.reason)
        self.tool.assert_awaited_once_with("search_feishu_docs", {"query": "不存在的方案"})

    async def test_direct_feishu_link_skips_search_and_fetches_exact_document(self):
        self.tool.side_effect = [{
            "ok": True, "url": "https://acme.feishu.cn/docx/abc123",
            "text": "指定飞书文档的正文，可直接用于本次任务。", "chars": 21,
        }]
        result = await intake.judge(
            "根据 https://acme.feishu.cn/docx/abc123 制作30秒视频"
        )
        self.assertTrue(result.ready)
        self.assertEqual(result.external_url, "https://acme.feishu.cn/docx/abc123")
        self.tool.assert_awaited_once_with(
            "fetch_feishu_doc", {"url": "https://acme.feishu.cn/docx/abc123"}
        )
        self.model.assert_not_awaited()

    async def test_mentioning_possible_feishu_material_requires_confirmation(self):
        result = await intake.judge("我在飞书里可能有一份产品方案")
        self.assertTrue(result.needs_clarification)
        self.assertFalse(result.ready)
        self.assertIn("确认", result.reason)
        self.server.assert_not_called()
        self.model.assert_not_awaited()

    async def test_feishu_as_subject_does_not_grant_account_access(self):
        self.model.return_value = classification("material")
        result = await intake.judge("飞书是一款企业协作产品，这段文字介绍它的发展历程和主要功能。")
        self.assertTrue(result.ready)
        self.assertFalse(result.needs_external_material)
        self.server.assert_not_called()

    async def test_classifier_outage_keeps_supplied_paragraph(self):
        self.model.side_effect = RuntimeError("offline")
        result = await intake.judge(MATERIAL)
        self.assertTrue(result.ready)
        self.assertFalse(result.needs_knowledge)
        self.server.assert_not_called()

    async def test_classifier_outage_handles_explicit_and_bare_topics(self):
        self.model.side_effect = RuntimeError("offline")
        for text in (TOPIC, "遵义会议", "党是怎么出来的"):
            with self.subTest(text=text):
                self.tool.reset_mock()
                self.tool.side_effect = [found()]
                result = await intake.judge(text)
                self.assertTrue(result.needs_knowledge)
                self.assertFalse(self.tool.await_args.args[1]["allow_web"])

    async def test_invalid_classification_cannot_open_tools_for_material(self):
        self.model.return_value = {"reason": "missing intent", "needs_knowledge": True}
        result = await intake.judge(MATERIAL)
        self.assertTrue(result.ready)
        self.server.assert_not_called()

    async def test_local_only_research_never_calls_search_or_fetch(self):
        # 必须把语义检索也 mock 掉。以前只 mock 了主题检索，测试是靠「库里恰好
        # 没有这条」通过的——收进一本书之后，「遵义会议」立刻被语义层召回，这个
        # 用例就红了。真正的意图是「本地什么都没有 + 禁止联网 → 报错且不联网」，
        # 那就该把「本地什么都没有」显式摆出来，而不是指望生产库碰巧是空的。
        with patch.object(kb.store, "search", return_value=[]), \
             patch.object(kb.vector, "search", new_callable=AsyncMock) as semantic, \
             patch.object(kb, "search", new_callable=AsyncMock) as search, \
             patch.object(kb, "_fetch", new_callable=AsyncMock) as fetch:
            semantic.return_value = {"ok": True, "hits": []}
            with self.assertRaises(kb.ResearchError) as error:
                await kb.research("遵义会议", allow_web=False)
            self.assertEqual(error.exception.error_type, "local_not_found")
            search.assert_not_awaited()
            fetch.assert_not_awaited()

    def mock_pipeline_agents(self):
        called = []

        def resolve(name):
            async def run(ctx, definition, options):
                called.append((name, ctx.state["chapter_text"], ctx.state["source_content"]))
                if name == "director":
                    return {"segments": [{"id": "seg1", "title": "test", "order": 0}], "review_mode": "NORMAL"}
                if name == "quality_review":
                    return {"critic_verdict": {"approved": True}}
                return {}
            return None, run

        self.stack.enter_context(patch.object(pipeline_six.registry, "catalog", return_value=[{"id": "script_adaptor"}]))
        self.stack.enter_context(patch.object(pipeline_six.registry, "resolve", side_effect=resolve))
        self.stack.enter_context(patch.object(media_mode, "prompt_only", return_value=True))
        self.stack.enter_context(patch.object(pipeline_six, "_assemble", new_callable=AsyncMock, return_value=""))
        for name, value in [("_video_capabilities", {}), ("_image_capabilities", {}),
                            ("_image_model_name", ""), ("_video_model_name", "")]:
            self.stack.enter_context(patch.object(pipeline_six, name, return_value=value))
        return called

    async def run_pipeline(self, text):
        return await pipeline_six.run_pipeline_six(
            run_id="test-input-routing", chapter_text=text, target_seconds=30,
            model_settings=None, max_rework=0,
        )

    async def test_material_reaches_director_and_adaptor_unchanged(self):
        called = self.mock_pipeline_agents()
        result = await self.run_pipeline(MATERIAL)
        self.assertIn("script_adaptor", [item[0] for item in called])
        self.assertTrue(all(item[1] == MATERIAL and item[2] == MATERIAL for item in called))
        self.assertEqual(result["state"]["material_origin"], "user")
        self.server.assert_not_called()

    async def test_failed_local_search_stops_and_asks_instead_of_dispatching(self):
        """本地没有就停下来问，不再自动联网、也不再直接判死这一轮。

        以前这里会连着查一次联网，然后把整轮以「已中断生成」结束。两步都不对：
        联网该由用户决定（可能要的是他自己那份材料），而「本地没有」不等于
        「做不了」，界面上却只剩一句失败。
        """
        called = self.mock_pipeline_agents()
        self.model.return_value = classification("topic_request", "遵义会议")
        self.tool.side_effect = [MISS]
        with self.assertRaises(lookup.AskUser) as caught:
            await self.run_pipeline(TOPIC)
        question = caught.exception.question
        self.assertIn("遵义会议", question["title"])
        self.assertEqual(
            [item["id"] for item in question["options"]], ["provide", "web", "stop"]
        )
        self.assertEqual(called, [], "没拿到材料就分发，等于拿提问当正文")
        self.assertEqual(self.tool.call_count, 1, "联网要等用户点了才发")

    async def test_search_ledger_keeps_the_process_when_run_stops_to_ask(self):
        """停下来问用户之前，检索过程必须先落袋为安。

        过程框跑的时候在 live 桶里看得见，run 一结束桶就清了。账本消息要赶在
        问卡之前发出、并且把那串过程带上——否则回看时只剩「没找到」，它是怎么
        找的、哪个词失败、为什么失败，一点痕迹都没有。
        """
        self.mock_pipeline_agents()
        self.model.return_value = classification("topic_request", "遵义会议")
        self.tool.side_effect = [MISS]
        events: list[dict] = []

        async def collect(event):
            events.append(event)

        with self.assertRaises(lookup.AskUser):
            await pipeline_six.run_pipeline_six(
                run_id="test-ledger", chapter_text=TOPIC, target_seconds=30,
                model_settings=None, max_rework=0, on_event=collect,
            )
        ledger = [e for e in events if e.get("msg_key") == "director:search"]
        self.assertEqual(len(ledger), 1, "停下来问之前，账本必须已经发出")
        steps = (ledger[0].get("payload") or {}).get("steps") or []
        self.assertTrue(steps, "账本必须带上过程框，否则过程跟着 run 一起消失")
        blob = json.dumps(steps, ensure_ascii=False)
        self.assertIn("search_knowledge", blob)
        # 事件流里账本排在所有消息之后、AskUser 抛出之前——也就是界面上它一定
        # 在问卡上面，用户读到问题时排查依据已经在了。
        self.assertEqual(events.index(ledger[0]), len(events) - 1)

    async def test_retrieved_material_dispatched_but_original_request_retained(self):
        called = self.mock_pipeline_agents()
        self.model.return_value = classification("topic_request", "遵义会议")
        self.tool.side_effect = [found("web")]
        result = await self.run_pipeline(TOPIC)
        self.assertIn("script_adaptor", [item[0] for item in called])
        # 材料本体进了下游，原始请求没被当正文。断言用「包含」而不是「等于」：
        # 导演交上来的是一份材料包，每段前面带着「【片段 n】《书名》出处」——
        # 没有那一行，写脚本的 agent 说不出依据在哪一章，只能写「据记载」。
        self.assertTrue(all(found()["entry"]["body"] in item[1] for item in called))
        self.assertTrue(all(item[2] == TOPIC for item in called))
        self.assertEqual(result["state"]["knowledge"]["status"], "pending")

    async def test_feishu_document_is_ephemeral_material_not_knowledge(self):
        called = self.mock_pipeline_agents()
        self.model.return_value = {
            "intent": "feishu_request", "source_query": "Q4产品方案",
            "desired_seconds": 30, "reason": "用户明确要求读取飞书文档",
        }
        self.tool.side_effect = [
            {"ok": True, "results": [{"title": "Q4产品方案", "url": "https://acme.feishu.cn/docx/abc"}]},
            {"ok": True, "title": "Q4产品方案", "url": "https://acme.feishu.cn/docx/abc", "text": "飞书方案正文，可直接用于本次视频改编。", "chars": 20},
        ]
        result = await self.run_pipeline("用飞书里的《Q4产品方案》做一个30秒视频")
        self.assertIn("script_adaptor", [item[0] for item in called])
        self.assertTrue(all(item[1].startswith("飞书方案正文") for item in called))
        self.assertEqual(result["state"]["material_origin"], "feishu")
        self.assertFalse(result["state"]["external_material"]["persisted"])
        self.assertNotIn("knowledge", result["state"])


class InsufficientGuardTests(unittest.TestCase):
    """模型判「空泛」时，规则层要能兜住确凿的主题请求。

    真实案例：同一句「做一个建国一个相关的视频」，一轮判成主题请求正常跑完，
    下一轮判成 insufficient 停下来问用户——分类是概率的，而这句话的「请求 +
    主题」是规则层能确定的。规则说了算的部分不该跟着模型抖。
    """

    def test_clear_request_is_not_downgraded_by_a_jittery_model(self):
        result = intake._to_intake(
            {"intent": "insufficient", "reason": "输入空泛，属于空泛请求。"},
            body="做一个建国一个相关的视频",
        )
        self.assertEqual(result.intent, "topic_request")
        self.assertTrue(result.needs_knowledge)
        self.assertTrue(result.topic)

    def test_genuinely_empty_request_stays_insufficient(self):
        result = intake._to_intake(
            {"intent": "insufficient", "reason": "没有主题。"},
            body="帮我做个视频",
        )
        self.assertEqual(result.intent, "insufficient")
        self.assertFalse(result.needs_knowledge)
        self.assertEqual(result.topic, "")


class OpeningNarrationTests(unittest.TestCase):
    """开场那句话：只能讲此刻已确定的事，不许提前承诺还没发生的动作。

    用户的话说得很直白——「先去本地库里查……后面必须接上他在找的行动」。开场这
    一刻出题还没发生、检索词一个都还没有，所以它只能说「我拿到了什么、这片子要
    讲什么」；一旦它开始列「我要去找 ①②③」，那句话在接下来几十秒里没有对应的
    动作，就是一句听起来很真诚的假话。
    """

    def test_missing_narration_stays_empty_instead_of_being_filled_in(self):
        # 宁可少一句，也不要假的一句：模型没写就留空，由调用方决定不显示。
        self.assertEqual(intake._narration({}), "")
        self.assertEqual(intake._narration({"narration": "   "}), "")

    def test_it_strips_the_voice_the_model_should_not_use(self):
        self.assertEqual(
            intake._narration({"narration": "作为导演，你给的是一个主题——建党。"}),
            "你给的是一个主题——建党。",
        )
        self.assertEqual(
            intake._narration({"narration": "narration: 你给的是一个主题。"}),
            "你给的是一个主题。",
        )

    def test_it_is_cut_so_one_sentence_cannot_become_a_page(self):
        self.assertEqual(len(intake._narration({"narration": "我" * 900})), 600)

    def test_the_prompt_forbids_promising_a_search_it_cannot_start(self):
        # 提示词里必须留着这条禁令。它被改掉时，开场话就会重新变成「我要去找
        # ①②③」＋一段静默——这正是用户这次要拆掉的东西。
        self.assertIn("不要列材料清单", intake.SYSTEM)
        self.assertIn("说了就是空话", intake.SYSTEM)

    def test_the_plan_prompt_tells_the_model_to_stop_at_the_current_action(self):
        from app.doc import director_search

        # 第一轮：分析贴出来之后紧接着就是检索回执，所以不许预告「之后还会补查」。
        self.assertIn("现在就按这几个词去查", director_search.PLAN_SYSTEM)
        self.assertIn("不要预告后面", director_search.PLAN_SYSTEM)
        # 第二轮：stop=true 时程序不会再查，话里就不许写「我再查一次」。
        self.assertIn("绝对不许", director_search.REVIEW_SYSTEM)


if __name__ == "__main__":
    unittest.main()
