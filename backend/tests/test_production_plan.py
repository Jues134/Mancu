"""任务清单：开工前贴出来、边跑边推进、没有「合成」这一步。

锁的三件事，都是讨论里定下来又被推回过的：

1. 清单是三步（取材 / 分段 / 逐段生产），没有合成。合成确实在后台发生，所以
   边界写在 `beyond` 里，而不是装作不存在。
2. 段数只有切完段才知道。开工前写「切成 3 段」是编的。
3. 不能跳过。段失败就是失败，不存在「降级完成」——那条路已经在
   `test_stage_failure_no_skip.py` 里单独守着了。
"""
from __future__ import annotations

import unittest

from app.doc import production_plan as pp

STAGES = [
    {"id": "script_adaptor", "label": "内容改编"},
    {"id": "storyboard", "label": "分镜"},
    {"id": "consistency", "label": "一致性"},
    {"id": "video_gen", "label": "视频生成"},
]


def started(route: str = "kb", **material):
    base = {"route": route, "chars": 860, "title": "中国共产党百年大事记"}
    base.update(material)
    return pp.start(material=base, stages=STAGES, rework_cap=1)


class PlanShapeTests(unittest.TestCase):
    def test_three_steps_and_no_assembly(self):
        card = started()
        titles = [s["title"] for s in card["steps"]]
        self.assertEqual(titles, ["取材", "分段", "逐段生产"])
        # 合成不在清单里，但必须有一句话说明它在后台发生——否则会出现
        # 「清单都做完了，用户手里多出一个清单没写的东西」。
        self.assertNotIn("合成", titles)
        self.assertIn("后台合成", card["beyond"])

    def test_every_step_has_owner_output_and_dependency(self):
        """少任何一件就不是任务，只是备注。"""
        for step in started()["steps"]:
            self.assertTrue(step["owner"], step["title"])
            self.assertTrue(step["output"], step["title"])
            if step["index"] > 1:
                self.assertTrue(step["depends"], f"{step['title']} 没写等谁")

    def test_order_is_derived_not_chosen(self):
        """顺序是 key 依赖的结果：取材→分段→逐段生产。"""
        steps = started()["steps"]
        self.assertEqual(steps[0]["depends"], "")
        self.assertIn("取材", steps[1]["depends"])
        self.assertIn("分段", steps[2]["depends"])

    def test_segment_count_is_not_written_before_segmentation(self):
        card = started()
        segment = next(s for s in card["steps"] if s["id"] == "segment")
        produce = next(s for s in card["steps"] if s["id"] == "produce")
        self.assertEqual(segment["status"], "running")
        self.assertEqual(produce["segments"], [])
        self.assertNotIn("3 段", segment["summary"])

    def test_blank_retrieval_is_recorded_on_the_step(self):
        card = started(chapter="第 12 章", locator="国民党一大召开")
        material = card["steps"][0]
        self.assertEqual(material["status"], "done")
        self.assertIn("第 12 章", material["result"])
        self.assertIn("国民党一大召开", material["result"])

    def test_raw_input_marks_material_step_as_not_needed(self):
        """「不需要」不是「没做」。两种说法在进度上一样，在解释上不同。"""
        card = started(route="user")
        material = card["steps"][0]
        self.assertEqual(material["status"], "not_needed")
        self.assertIn("直接用你给的原文", material["summary"])
        # 不阻塞进度：完成数照样算它。
        self.assertEqual(card["done"], 1)

    def test_missing_material_fails_the_first_step(self):
        card = started(route="none")
        self.assertEqual(card["steps"][0]["status"], "failed")
        self.assertEqual(card["state"], "failed")


class PlanProgressTests(unittest.TestCase):
    def test_segments_are_attached_after_director_returns(self):
        card = started()
        pp.begin_segments(
            card,
            [{"id": "seg1", "title": "三大政策"}, {"id": "seg2", "title": "国共合作"}],
        )
        segment = next(s for s in card["steps"] if s["id"] == "segment")
        produce = next(s for s in card["steps"] if s["id"] == "produce")
        self.assertEqual(segment["status"], "done")
        self.assertIn("seg1", segment["output"])
        self.assertEqual([s["id"] for s in produce["segments"]], ["seg1", "seg2"])
        self.assertEqual(produce["seg_total"], 2)

    def test_stage_dots_move_from_pending_to_done(self):
        card = started()
        pp.begin_segments(card, [{"id": "seg1", "title": "一"}])
        pp.touch(card, "seg1", "script_adaptor", "running")
        seg = card["steps"][2]["segments"][0]
        self.assertEqual(seg["stages"]["script_adaptor"], "running")
        self.assertEqual(seg["status"], "running")
        pp.touch(card, "seg1", "script_adaptor", "done")
        self.assertEqual(seg["stages"]["script_adaptor"], "done")

    def test_settled_segment_stops_moving(self):
        card = started()
        pp.begin_segments(card, [{"id": "seg1", "title": "一"}])
        pp.settle(card, "seg1", approved=True)
        seg = card["steps"][2]["segments"][0]
        self.assertEqual(seg["status"], "done")
        # 通过之后剩下的圆点补成完成，否则界面会停在「进行中」而这一段已经交付。
        self.assertTrue(all(v == "done" for v in seg["stages"].values()))

    def test_approval_wipes_the_old_failure_note(self):
        # 重现线上 bug：失败时挂上「卡在分镜设计」，重跑通过后备注没人擦，
        # 任务板满屏红字，看起来像没跑完——状态刷新了、备注没刷。
        card = started()
        pp.begin_segments(card, [{"id": "seg1", "title": "一"}])
        pp.settle(card, "seg1", approved=False, note="卡在「分镜设计」")
        pp.settle(card, "seg1", approved=True)
        seg = card["steps"][2]["segments"][0]
        self.assertEqual(seg["status"], "done")
        self.assertEqual(seg.get("note"), "", "通过后旧失败备注必须清空")

    def test_one_failed_segment_does_not_hide_the_others(self):
        card = started()
        pp.begin_segments(
            card, [{"id": "seg1", "title": "一"}, {"id": "seg2", "title": "二"}]
        )
        pp.touch(card, "seg1", "consistency", "failed", note="审核拦下")
        pp.settle(card, "seg1", approved=False)
        pp.settle(card, "seg2", approved=True)
        produce = card["steps"][2]
        self.assertEqual(produce["status"], "failed", "有一段没完成，这一步就不能算完成")
        self.assertEqual(produce["seg_done"], 2)
        self.assertEqual(card["state"], "failed")

    def test_headline_shows_segment_progress_while_running(self):
        card = started()
        pp.begin_segments(card, [{"id": "seg1", "title": "一"}, {"id": "seg2", "title": "二"}])
        pp.settle(card, "seg1", approved=True)
        self.assertIn("分段 1/2", pp.headline(card))

    def test_finish_never_leaves_a_step_running(self):
        card = started()
        pp.begin_segments(card, [{"id": "seg1", "title": "一"}])
        pp.touch(card, "seg1", "storyboard", "running")
        pp.finish(card, reason="2 段没有完成")
        for step in card["steps"]:
            self.assertNotIn(step["status"], ("pending", "running"))
        for seg in card["steps"][2]["segments"]:
            self.assertEqual(seg["status"], "failed")
        self.assertEqual(card["state"], "failed")

    def test_finish_is_idempotent(self):
        card = started()
        pp.begin_segments(card, [{"id": "seg1", "title": "一"}])
        pp.settle(card, "seg1", approved=True)
        before = pp.headline(card)
        pp.finish(card)
        self.assertEqual(pp.headline(card), before, "全部完成时收尾不该把完成改成未完成")


class ReopenTests(unittest.TestCase):
    """重跑沿用上一轮的清单：通过的段不动，点名的段回到等着跑。"""

    def make_finished_card(self) -> dict:
        card = pp.start(
            material={"route": "user", "chars": 500},
            stages=[{"id": "script_adaptor", "label": "剧本改编"}],
        )
        pp.begin_segments(card, [{"id": "seg1", "title": "一"}, {"id": "seg2", "title": "二"}])
        # 两段的剧本改编都做完了；seg2 死在后面的分镜上。
        pp.touch(card, "seg1", "script_adaptor", "done")
        pp.touch(card, "seg2", "script_adaptor", "done")
        pp.touch(card, "seg2", "storyboard", "failed", note="卡在「分镜设计」")
        pp.settle(card, "seg1", approved=True)
        pp.settle(card, "seg2", approved=False, note="卡在「分镜设计」")
        pp.finish(card)
        return card

    def test_retry_segment_returns_to_pending(self):
        card = pp.reopen(self.make_finished_card(), retry_ids={"seg2"})
        produce = next(s for s in card["steps"] if s["id"] == "produce")
        by_id = {s["id"]: s for s in produce["segments"]}
        self.assertEqual(by_id["seg1"]["status"], "done", "已通过段不该被重置")
        self.assertEqual(by_id["seg2"]["status"], "pending")
        self.assertEqual(by_id["seg2"]["stages"]["script_adaptor"], "done")
        self.assertEqual(card["state"], "running")

    def test_only_named_segments_are_reset(self):
        card = pp.reopen(self.make_finished_card(), retry_ids={"seg1"})
        produce = next(s for s in card["steps"] if s["id"] == "produce")
        by_id = {s["id"]: s for s in produce["segments"]}
        self.assertEqual(by_id["seg2"]["status"], "failed", "没点名的失败段保持原样")

    def test_reopened_segment_clears_the_rerun_note_when_done(self):
        # 重跑的段身上写着「准备重跑」；跑通过之后这句必须消失，
        # 否则任务板显示完成的同时还挂着红字，自相矛盾。
        card = pp.reopen(self.make_finished_card(), retry_ids={"seg2"})
        produce = next(s for s in card["steps"] if s["id"] == "produce")
        by_id = {s["id"]: s for s in produce["segments"]}
        self.assertEqual(by_id["seg2"]["note"], "准备重跑")
        pp.settle(card, "seg2", approved=True)
        self.assertEqual(by_id["seg2"]["status"], "done")
        self.assertEqual(by_id["seg2"].get("note"), "", "重跑通过后「准备重跑」必须清掉")

    def test_a_missing_card_returns_none(self):
        self.assertIsNone(pp.reopen(None, retry_ids={"seg1"}))
        self.assertIsNone(pp.reopen({}, retry_ids={"seg1"}))


if __name__ == "__main__":
    unittest.main()
