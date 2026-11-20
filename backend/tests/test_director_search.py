"""导演出题：想什么、查什么、够不够。

这个模块要防的是「换一个主题就废」。所以这里的测试大多不是断言「它查到了什么」，
而是断言**程序守住了哪几条线**：

- 检索词太短就不许查（两个字的词在中文里命中率恒等于 1，那不是检索）。
- 「从材料里抄的」和「按常识补的」由程序判，不采信模型自述。
- 覆盖率由程序算，模型说的不算。
- 缺材料时不能把「模型没跑起来」说成「库里没有」。

最后一条特别重要：三种失败（库里没有／导演不会提问／模型挂了）在界面上长得一样，
但下一步完全不同——用户要是分不清，就只能一直点重试。
"""

from __future__ import annotations

import unittest

from app.doc import director_search as ds


def clip(text: str, locator: str = "1949 年 · 中华人民共和国成立", title: str = "百年大事记") -> dict:
    return {
        "text": text,
        "locator": locator,
        "title": title,
        "entry_id": "book-1",
        "kind": "book",
        "status": "approved",
        "sources": [{"title": "百年大事记.md", "url": ""}],
        "score": 0.9,
    }


def finder(mapping: dict[str, list[dict]]):
    """按词返回命中。没列在表里的词一律未命中。"""
    calls: list[str] = []

    async def search(query: str) -> dict:
        calls.append(query)
        hits = mapping.get(query) or []
        return {
            "found": bool(hits),
            "clips": hits,
            "reason": "" if hits else "本地资料库 没有这条",
            "error_type": "" if hits else "local_not_found",
        }

    search.calls = calls  # type: ignore[attr-defined]
    return search


def thinker(*answers):
    """依次返回给定的题单，同时记下被问了几次。"""
    asked: list[tuple[str, str]] = []
    queue = list(answers)

    async def think(system: str, user: str, **_kwargs):
        asked.append((system, user))
        if not queue:
            raise AssertionError("多问了一轮")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    think.asked = asked  # type: ignore[attr-defined]
    return think


class QuestionTests(unittest.TestCase):
    def test_two_character_query_is_refused(self):
        """「建国」这种词不许直接查。

        书里「建国」出现 19 次，全是「建国初期」这种修饰语；而 1949 那一块的
        小标题逐字是「中华人民共和国成立」。两个字的词查不出这个区别，只会给
        一堆同分的结果——所以程序拦下它，逼导演换一个说法。
        """
        questions, dropped = ds._accept_questions(
            [{"ask": "新中国怎么来的", "slots": ["what"], "queries": ["建国", "中华人民共和国成立"]}],
            pool="", round_no=1, seen=set(),
        )
        self.assertEqual([q["queries"][0]["text"] for q in questions], ["中华人民共和国成立"])
        self.assertEqual(dropped[0]["query"], "建国")
        self.assertIn("3 个字", dropped[0]["why"])

    def test_filler_queries_are_refused(self):
        """「历史背景」不是任何一份资料里的说法，拿它查必然是 0，然后被记成缺口。"""
        questions, dropped = ds._accept_questions(
            [{"ask": "x", "queries": ["历史背景", "事件经过", "做视频需要的画面"]}],
            pool="", round_no=1, seen=set(),
        )
        self.assertEqual(questions, [])
        self.assertEqual(len(dropped), 3)

    def test_duplicate_queries_are_queried_once(self):
        questions, dropped = ds._accept_questions(
            [
                {"ask": "a", "queries": ["开国大典"]},
                {"ask": "b", "queries": ["开国大典"]},
            ],
            pool="", round_no=1, seen=set(),
        )
        self.assertEqual(len(questions), 1)
        self.assertIn("已经查过", dropped[0]["why"])

    def test_origin_is_decided_by_the_program(self):
        """词的来源由程序判：模型自报一律是 grounded。

        四个汉字的连续重合才算抄来的——「中国」「人民」这种词到处都是，按两字判
        会让每个词都变成 grounded。
        """
        pool = ds._compact("1949 年 10 月 1 日，庆祝中华人民共和国中央人民政府成立典礼")
        questions, _ = ds._accept_questions(
            [{"ask": "x", "queries": ["中华人民共和国中央人民政府", "金瓶掣签制度"]}],
            pool=pool, round_no=1, seen=set(),
        )
        origins = {q["text"]: q["origin"] for q in questions[0]["queries"]}
        self.assertEqual(origins["中华人民共和国中央人民政府"], "grounded")
        self.assertEqual(origins["金瓶掣签制度"], "inferred")

    def test_a_plan_without_queries_is_dropped_entirely(self):
        questions, dropped = ds._accept_questions(
            [{"ask": "背景是什么", "queries": []}], pool="", round_no=1, seen=set()
        )
        self.assertEqual(questions, [])
        self.assertEqual(dropped, [])

    def test_unknown_slots_are_ignored(self):
        """模型偶尔会自造验收位（「情绪」「节奏」），不能让它污染覆盖率的分母。"""
        questions, _ = ds._accept_questions(
            [{"ask": "x", "slots": ["when", "mood", "情绪"], "queries": ["开国大典"]}],
            pool="", round_no=1, seen=set(),
        )
        self.assertEqual(questions[0]["slots"], ["when"])


class CoverageTests(unittest.TestCase):
    def test_missing_slots_are_counted_by_the_program(self):
        questions = [
            {"id": "q1", "slots": ["when", "what"], "hits": ["1949 年 · 成立"]},
            {"id": "q2", "slots": ["why"], "hits": []},
        ]
        cover = ds.coverage(questions)
        self.assertEqual(cover["covered"], ["when", "what"])
        self.assertEqual(cover["missing"], ["where", "who", "why"])
        self.assertEqual(cover["answered"], 1)
        # 缺三个，超了上限：这条片子还立不起来。
        self.assertFalse(cover["enough"])

    def test_two_missing_slots_is_still_enough(self):
        questions = [
            {"id": "q1", "slots": ["when", "what", "why"], "hits": ["某段"]},
        ]
        self.assertTrue(ds.coverage(questions)["enough"])

    def test_no_hits_is_never_enough(self):
        """一个片段都没有时，「缺得少」不代表够——那是没查。"""
        self.assertFalse(ds.coverage([{"id": "q1", "slots": ["when"], "hits": []}])["enough"])


class PackTests(unittest.TestCase):
    def test_every_clip_carries_its_locator(self):
        text = ds.pack([clip("1949 年 10 月 1 日，典礼在天安门广场举行。")])
        self.assertIn("【片段 1】", text)
        self.assertIn("百年大事记", text)
        self.assertIn("1949 年 · 中华人民共和国成立", text)

    def test_a_clip_without_a_locator_is_dropped(self):
        """没有出处的片段不进材料包。

        进去了，写脚本那一步就会把它写成「据记载」——一句谁也不用负责的话。
        """
        text = ds.pack([clip("某段没有出处的话", locator="", title="")])
        self.assertEqual(text, "")

    def test_pack_stops_at_the_budget(self):
        many = [clip("甲" * 500, locator=f"第 {i} 章") for i in range(30)]
        self.assertLessEqual(len(ds.pack(many)), ds.MATERIAL_CHARS + ds.CLIP_CHARS)


class BudgetTests(unittest.TestCase):
    def test_a_long_question_list_is_cut_instead_of_shown_as_misses(self):
        """裁掉的题不能留在清单上写着「未命中」——它一次都没查过。

        模型被要求出 3–6 个问题、每题 1–3 个词，很容易出到十几条。不裁的话，
        清单上有六条、后三条标着未命中，用户读到的是「查过没有」，事实是「没查」。
        """
        questions = [
            {"id": f"q{i}", "queries": [{"text": f"词{i}"}]} for i in range(10)
        ]
        kept, cut = ds._within_budget(questions, room=4)
        self.assertEqual(len(kept), 4)
        self.assertEqual(len(cut), 6)

    def test_the_first_question_always_survives(self):
        """预算只剩 1 也要留下一题。一条都不查，等于没有出题。"""
        questions = [{"id": "q1", "queries": [{"text": "甲"}, {"text": "乙"}]}]
        kept, cut = ds._within_budget(questions, room=0)
        self.assertEqual(len(kept), 1)
        self.assertEqual(cut, [])


class TemplateSmellTests(unittest.TestCase):
    def test_a_generic_question_list_is_flagged(self):
        questions = [
            {"ask": "这件事的历史背景是什么"},
            {"ask": "它的经过是怎样的"},
            {"ask": "有什么意义"},
        ]
        self.assertTrue(ds._template_smell(questions))

    def test_specific_questions_are_not_flagged(self):
        questions = [
            {"ask": "政协之前国共谈判为什么破裂"},
            {"ask": "定都北京是什么时候定的"},
            {"ask": "阅兵的影像资料从哪儿来"},
        ]
        self.assertEqual(ds._template_smell(questions), "")

    def test_two_questions_are_not_enough_to_call_it_a_template(self):
        self.assertEqual(ds._template_smell([{"ask": "背景"}, {"ask": "意义"}]), "")


class PlanTextTests(unittest.TestCase):
    """贴在对话里的那几行字。它存在的唯一目的是让用户判断「该不该换个说法再试」。"""

    def test_every_query_is_named_with_its_result(self):
        from app.doc.pipeline_six import _plan_text

        text = _plan_text({
            "anchor": {"subject": "中华人民共和国成立", "when": "1949 年"},
            "questions": [
                {
                    "ask": "那天发生了什么",
                    "queries": [
                        {"text": "中华人民共和国成立", "origin": "grounded"},
                        {"text": "开国大典", "origin": "inferred"},
                    ],
                    "hits": ["1949 年 · 中华人民共和国成立"],
                },
                {"ask": "政权从哪来", "queries": [{"text": "渡江战役", "origin": "grounded"}], "hits": []},
            ],
            "tried": [
                {"query": "中华人民共和国成立", "origin": "grounded", "found": 1,
                 "reason": "", "error_type": ""},
                {"query": "开国大典", "origin": "inferred", "found": 0,
                 "reason": "向量化失败：connection refused", "error_type": "local_error"},
                {"query": "渡江战役", "origin": "grounded", "found": 0,
                 "reason": "本地未命中", "error_type": "local_not_found"},
            ],
            "dropped": [{"query": "建国", "why": "少于 3 个字"}],
            "coverage": {"covered": ["when", "what"], "missing": ["why"]},
            "notes": [],
        })
        self.assertIn("中华人民共和国成立", text)
        self.assertIn("命中 1 段", text)
        self.assertIn("未命中", text)
        # 检索坏了和「没找到」必须分开写：用户看到前者该去修配置，看到后者才该换词。
        self.assertIn("没查成", text)
        self.assertNotIn("向量化失败", text)  # 长报错不进账本行，在过程框里
        self.assertIn("「建国」", text, "被拦下的词也要说出来，否则用户以为它没想到")
        self.assertIn("还缺 为何重要", text)
        self.assertIn("*", text, "按常识补的说法要标出来")

    def test_the_director_speaks_in_its_own_words(self):
        """卡片主体是模型自己写的分析，不是程序填的固定句式。

        这是这张卡存在的意义：换了主题，它讲的话要跟着变。程序只接账目——两段
        分析原文照贴，轮与轮之间插一行分隔，末尾才是核对过的数字。
        """
        from app.doc.pipeline_six import _plan_text

        text = _plan_text({
            "anchor": {"subject": "中共一大", "when": "1921 年"},
            "analysis": [
                {"round": 1, "text": "我怀疑库里不会叫「中共一大」，正式记载可能写全称。"},
                {"round": 2, "text": "「上海转移」命中了，说明书是按事件经过写的。"},
            ],
            "questions": [
                {
                    "round": 1, "ask": "为什么转移到南湖",
                    "queries": [{"text": "中共一大上海转移", "origin": "grounded"}],
                    "hits": ["1921 年 · 中共一大转移"],
                },
                {
                    "round": 2, "ask": "谁参加了会议",
                    "queries": [{"text": "南陈北李", "origin": "inferred"}],
                    "hits": [],
                },
            ],
            "tried": [
                {"query": "中共一大上海转移", "origin": "grounded", "found": 1,
                 "reason": "", "error_type": ""},
                {"query": "南陈北李", "origin": "inferred", "found": 0,
                 "reason": "本地未命中", "error_type": "local_not_found"},
            ],
            "dropped": [],
            "coverage": {"covered": ["where", "what"], "missing": ["who"]},
            "notes": [],
        })
        self.assertIn("我怀疑库里不会叫「中共一大」", text)
        self.assertIn("「上海转移」命中了", text)
        self.assertIn("—— 第 1 轮结果回来了 ——", text, "轮与轮之间要分开，读者才知道后一段是看过结果说的")
        self.assertIn("程序核对：", text, "程序算的账目要单独一行，不许混进自述")
        # 自述在前、账目在后：先读它的判断，再读核对过的数字。
        self.assertLess(text.index("我怀疑库里"), text.index("程序核对："))

    def test_live_snapshot_says_pending_not_dropped(self):
        """跑到一半的快照不能说「没查」——那是在宣布它放弃了。

        进度卡每轮都会覆写一次，第一轮刚出完题、词还没查时，每个词都还没有结论。
        这时写「没查」读者会以为它不打算查了。
        """
        from app.doc.pipeline_six import _plan_text

        text = _plan_text({
            "questions": [
                {"round": 1, "ask": "为什么转移到南湖",
                 "queries": [{"text": "中共一大上海转移", "origin": "grounded"}], "hits": []},
            ],
            "tried": [],
            "coverage": {},
        }, live=True)
        self.assertIn("正在查…", text)
        self.assertNotIn("没查", text.replace("没查成", ""))


class RunTests(unittest.IsolatedAsyncioTestCase):
    async def test_it_rewrites_the_topic_instead_of_querying_it_raw(self):
        """「建国」→ 换成库里的说法再查。这是这个模块存在的全部理由。"""
        plan = {
            "anchor": {"kind": "event", "subject": "中华人民共和国成立", "when": "1949 年"},
            "questions": [{
                "ask": "1949 年 10 月 1 日发生了什么",
                "slots": ["when", "where", "what"],
                "queries": ["建国", "中华人民共和国成立"],
                "why": "正片的主体",
            }],
        }
        search = finder({"中华人民共和国成立": [clip("1949 年 10 月 1 日，典礼在天安门广场举行。")]})
        result = await ds.run("建国", search=search, think=thinker(plan))

        self.assertEqual(search.calls, ["中华人民共和国成立"], "两个字那个词不该被查")
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["clips"]), 1)
        self.assertIn("中华人民共和国成立", result["material"])
        self.assertEqual(result["coverage"]["covered"], ["when", "where", "what"])
        self.assertEqual(result["dropped"][0]["query"], "建国")

    async def test_progress_is_pushed_after_every_single_query(self):
        """卡片要边跑边长：出完题推一次，**每查完一个词再推一次**。

        只在整轮结束时推一次，中间仍有一段静默：话里说「现在去查这三个」，接着
        什么都没有，然后突然出现三条结论——读起来还是「说了没做」。逐词推送才
        是「说完就在做」：第 n 次快照里正好有前 n 个词的结果。
        """
        first = {
            "analysis": "我怀疑库里不会叫「建国」，正式记载应该写全称。",
            "questions": [{
                "ask": "1949 年 10 月 1 日发生了什么",
                "slots": ["when", "where", "what"],
                "queries": ["中华人民共和国成立", "开国大典", "天安门城楼"],
            }],
        }
        search = finder({
            "中华人民共和国成立": [clip("典礼在天安门广场举行。")],
            "开国大典": [clip("三十万人参加了典礼。")],
            "天安门城楼": [],
        })
        snapshots: list[dict] = []

        def progress(snapshot: dict) -> None:
            snapshots.append(snapshot)

        result = await ds.run("建国", search=search, think=thinker(first), on_progress=progress)

        # 1 次（出完题、还没查）+ 3 次（每查完一个词）+ 1 次（本轮收尾）
        self.assertEqual(len(snapshots), 5, "每个词都要推一次，不能等整轮查完才给")
        self.assertEqual(snapshots[0]["tried"], [], "第一次推的时候还没查，tried 必须是空的")
        self.assertIn("我怀疑库里不会叫「建国」", snapshots[0]["analysis"][0]["text"])
        # 逐词递增：第 2 次 1 个词、第 3 次 2 个、第 4 次 3 个。
        self.assertEqual(
            [len(item["tried"]) for item in snapshots[1:4]], [1, 2, 3],
            "回执要一个词一个词地长出来",
        )
        self.assertEqual(snapshots[2]["tried"][1]["found"], 1, "第二个词的回执要带上命中数")
        self.assertEqual(snapshots[3]["tried"][2]["found"], 0, "未命中也要如实报 0")
        self.assertEqual(result["analysis"][0]["round"], 1)

    async def test_a_second_round_runs_only_when_slots_are_missing(self):
        first = {
            "questions": [{
                "ask": "典礼在哪举行", "slots": ["where", "what"],
                "queries": ["中华人民共和国成立"],
            }],
        }
        second = {
            "reviews": [{"ask": "典礼在哪举行", "verdict": "answered", "note": "拿到了地点和经过"}],
            "gaps": ["when", "who", "why"],
            "questions": [{
                "ask": "为什么这件事重要", "slots": ["why"],
                "queries": ["新中国的成立的历史意义"],
            }],
            "surprise": "材料说下午三点开始，和预想不同",
        }
        search = finder({
            "中华人民共和国成立": [clip("典礼在天安门广场举行。")],
            "新中国的成立的历史意义": [clip("结束了半殖民地半封建社会历史。", "1949 年 · 成立")],
        })
        think = thinker(first, second)
        result = await ds.run("建国", search=search, think=think)

        self.assertEqual(len(think.asked), 2)
        self.assertIn("还没覆盖的验收位", think.asked[1][1])
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["coverage"]["covered"], ["where", "what", "why"])
        self.assertIn("和预想不同", result["surprise"])
        # 排查结论也要落到 notes 上：第二轮不只是补查，它同时说明了「拿回来的
        # 东西回答了什么」。
        self.assertIn("拿到了地点和经过", "".join(result["notes"]))

    async def test_it_stops_after_one_round_when_the_slots_are_full(self):
        plan = {
            "questions": [{
                "ask": "整件事", "slots": list(ds.SLOTS),
                "queries": ["中华人民共和国成立"],
            }],
        }
        search = finder({"中华人民共和国成立": [clip("1949 年 10 月 1 日，天安门广场。")]})
        think = thinker(plan)
        result = await ds.run("建国", search=search, think=think)
        self.assertEqual(len(think.asked), 1)
        self.assertEqual(result["coverage"]["missing"], [])

    async def test_a_broken_model_does_not_become_a_missing_material_verdict(self):
        """模型挂了，要说「没跑通」，不能说「库里没有」。

        上一版在这里会说「导演试过 1 个检索词，本地没有可用材料」——用户据此去
        联网找材料，而真正的毛病是他没配模型。
        """
        search = finder({})
        result = await ds.run("建国", search=search, think=thinker(RuntimeError("no model")))
        self.assertFalse(result["ok"])
        self.assertIn("no model", result["plan_error"])
        self.assertIn("没跑起来", result["reason"])
        self.assertEqual(search.calls, ["建国"], "兜底要按原句查一次")

    async def test_a_tool_failure_is_reported_as_a_failure(self):
        async def broken(_query: str) -> dict:
            return {"found": False, "clips": [], "reason": "database unavailable", "error_type": "tool_error"}

        result = await ds.run("遵义会议", search=broken, think=thinker({"questions": []}))
        self.assertIn("没跑通", result["reason"])
        self.assertIn("database unavailable", result["reason"])

    async def test_a_miss_says_which_words_were_tried(self):
        """卡上要写出查过哪些词。

        「查了『建国』没命中」和「查了『中华人民共和国成立』没命中」是两种处境：
        前者该换个说法，后者该去联网。只说「没找到」，用户无法判断。
        """
        search = finder({})
        result = await ds.run("开国大典", search=search, think=thinker(
            {"questions": [{"ask": "典礼怎么举行的", "slots": ["what"], "queries": ["开国大典纪实"]}]},
            # 一轮没覆盖住五问，会再问一次；第二次也没提出新词。
            {"questions": []},
        ))
        self.assertIn("开国大典纪实", result["reason"])
        self.assertFalse(result["ok"])

    async def test_an_empty_topic_never_queries(self):
        search = finder({})
        result = await ds.run("   ", search=search)
        self.assertEqual(search.calls, [])
        self.assertFalse(result["ok"])

    async def test_tool_calls_are_recorded_for_the_trace(self):
        seen: list[tuple] = []
        search = finder({"中华人民共和国成立": [clip("正文")]})
        await ds.run(
            "建国",
            search=search,
            think=thinker({"questions": [{"ask": "x", "slots": ["what"], "queries": ["中华人民共和国成立"]}]}),
            on_tool=lambda name, args, result: seen.append((name, args["query"], result["found"])),
        )
        self.assertEqual(seen, [("search_knowledge", "中华人民共和国成立", True)])
