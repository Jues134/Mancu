"""回答问卡的接口：答对了就续跑，答错了别把任务弄坏。

这里只测接口的契约（谁能在什么时候答、答错的后果），真正应用回答的逻辑在
`test_lookup_ask.py` 里——两处分开，是因为它们坏起来的表现完全不同：接口坏是
「点了没反应」，逻辑坏是「跑出来的东西不对」。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app import db
from app.api import doc as doc_api
from app.main import app


class AnswerRouteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        for name, value in (("DB_PATH", base / "studio.db"), ("_conn", None)):
            p = patch.object(db, name, value)
            p.start()
            self.addCleanup(p.stop)
        db.connect()  # 建表 + 迁移
        self.client = TestClient(app)
        self.rid = "run-answer-test"
        db.create_run({"chapter_text": "做一个习近平的生平视频", "options": {"chapter_text": "做一个习近平的生平视频"}}, run_id=self.rid)

    def tearDown(self):
        # Windows 上没关掉的连接会让临时目录删不掉（WinError 32），而这会以
        # "清理失败" 的名义把一条本该通过的用例标红。
        conn, db._conn = db._conn, None
        try:
            if conn is not None:
                conn.close()
        except Exception:  # noqa: BLE001 - 已经关掉就算了
            pass

    def park(self, *, options=None) -> dict:
        """把 run 放到「停在等回答」的状态。"""
        question = {
            "id": "q1",
            "phase": "material",
            "topic": "习近平",
            "title": "本地没有「习近平」的现成材料，接下来从哪儿找？",
            "options": options or [
                {"id": "provide", "label": "我给你正文", "needs_text": True},
                {"id": "web", "label": "你去联网搜"},
                {"id": "stop", "label": "先不做了"},
            ],
        }
        pending = {"phase": "material", "question": question, "intake": {"topic": "习近平"}, "answer": None}
        db.update_run(self.rid, status="waiting", pending=pending)
        return pending

    def test_a_waiting_run_accepts_an_answer_and_requeues(self):
        self.park()
        with patch.object(doc_api, "_execute", new_callable=AsyncMock) as execute:
            resp = self.client.post(
                f"/api/doc/runs/{self.rid}/answer",
                json={"question_id": "q1", "choice": "provide", "text": "2012 年 11 月……"},
            )
        self.assertEqual(resp.status_code, 200)
        execute.assert_awaited_once()

        run = db.get_run(self.rid)
        self.assertEqual(run["status"], "queued")
        self.assertEqual(run["pending"]["answer"]["choice"], "provide")
        self.assertIn("2012", run["pending"]["answer"]["text"])
        # 回答要落回那张卡，否则历史里会留着一张还在问的卡，而任务早就往下走了。
        cards = [m for m in db.list_messages(self.rid) if m.get("kind") == "ask"]
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["payload"]["answer"]["choice"], "provide")

    def test_stop_cancels_without_running_anything(self):
        """选了「先不做了」就不该再入队——否则管线会先把能力报告重发一遍。"""
        self.park()
        with patch.object(doc_api, "_execute", new_callable=AsyncMock) as execute:
            resp = self.client.post(f"/api/doc/runs/{self.rid}/answer", json={"choice": "stop"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "cancelled")
        execute.assert_not_awaited()
        self.assertEqual(db.get_run(self.rid)["status"], "cancelled")

    def test_running_run_cannot_be_answered(self):
        db.update_run(self.rid, status="running")
        resp = self.client.post(f"/api/doc/runs/{self.rid}/answer", json={"choice": "web"})
        self.assertEqual(resp.status_code, 409)
        self.assertIn("不在等待回答", resp.text)

    def test_choice_must_come_from_the_question(self):
        """选项列表是那次提问给的。别的值（包括拼错的）不能被当成默认路去跑。"""
        self.park()
        resp = self.client.post(f"/api/doc/runs/{self.rid}/answer", json={"choice": "dingtalk"})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(db.get_run(self.rid)["status"], "waiting")

    def test_provide_without_text_is_rejected_before_anything_runs(self):
        self.park()
        with patch.object(doc_api, "_execute", new_callable=AsyncMock) as execute:
            resp = self.client.post(
                f"/api/doc/runs/{self.rid}/answer",
                json={"choice": "provide", "text": "  "},
            )
        self.assertEqual(resp.status_code, 400)
        execute.assert_not_awaited()
        self.assertEqual(db.get_run(self.rid)["status"], "waiting")

    def test_a_stale_question_id_is_rejected(self):
        """两个窗口都开着时，后提交的那次不该把已经续跑的任务再放一遍。"""
        self.park()
        resp = self.client.post(
            f"/api/doc/runs/{self.rid}/answer",
            json={"question_id": "q-old", "choice": "web"},
        )
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(db.get_run(self.rid)["status"], "waiting")

    def test_a_waiting_run_can_be_cancelled(self):
        """「等你回复」也是能取消的。

        它不算「执行中」——管线早就返回了，没人在替它往下走——但它同样不是
        「已结束」：那条记录还占着一个没答的问题。前端把 waiting 当活动状态、
        按钮叫「取消任务」，接口却回一句「无需取消」，用户就被卡在一个自己
        明确不想要的任务上。
        """
        self.park()
        with patch.object(doc_api, "_execute", new_callable=AsyncMock) as execute:
            resp = self.client.post(f"/api/doc/runs/{self.rid}/cancel")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"], body)
        run = db.get_run(self.rid)
        self.assertEqual(run["status"], "cancelled")
        self.assertEqual(run["pending"], {}, "挂着的那张问题要一起丢掉，否则还能被回答")
        self.assertIn("cancel", [m.get("kind") for m in db.list_messages(self.rid)])
        execute.assert_not_awaited()

    def test_a_question_stops_being_answerable_after_cancel(self):
        """取消之后那张卡上的按钮就该失效，不能点了才拿到 409。"""
        self.park()
        self.client.post(f"/api/doc/runs/{self.rid}/cancel")
        resp = self.client.post(f"/api/doc/runs/{self.rid}/answer", json={"choice": "web"})
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(db.get_run(self.rid)["status"], "cancelled")

    def test_a_finished_run_refuses_cancel_in_plain_words(self):
        """真结束了才该说「不用取消」，而且要说人话，不是把 slug 抛给用户。"""
        db.update_run(self.rid, status="done")
        resp = self.client.post(f"/api/doc/runs/{self.rid}/cancel")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertFalse(body["ok"])
        self.assertIn("已完成", body["reason"])
        self.assertNotIn("done", body["reason"])
        self.assertEqual(db.get_run(self.rid)["status"], "done")


class RetryRouteTests(unittest.TestCase):
    """「继续生成」接口：只重跑没完成的段，跑完的任务才能继续。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        for name, value in (("DB_PATH", base / "studio.db"), ("_conn", None)):
            p = patch.object(db, name, value)
            p.start()
            self.addCleanup(p.stop)
        db.connect()
        self.client = TestClient(app)
        self.rid = "run-retry-test"
        db.create_run({"chapter_text": "做一个建国相关的视频"}, run_id=self.rid)

    def tearDown(self):
        conn, db._conn = db._conn, None
        try:
            if conn is not None:
                conn.close()
        except Exception:  # noqa: BLE001
            pass

    def park_finished(self, *, status="failed") -> None:
        """把 run 放成「跑完但有段失败」的状态：清单卡 + 持久化 state。"""
        db.update_run(
            self.rid,
            status=status,
            pending={},
            state={"segments": [{"id": "seg1", "title": "一"}, {"id": "seg2", "title": "二"}]},
        )
        db.add_message(
            self.rid,
            role="agent",
            agent="director",
            kind="plan",
            content="未完成 2/3",
            payload={
                "steps": [
                    {"id": "material", "title": "取材", "status": "done"},
                    {"id": "segment", "title": "分段", "status": "done"},
                    {
                        "id": "produce",
                        "title": "逐段生产",
                        "status": "failed",
                        "segments": [
                            {"id": "seg1", "title": "一", "status": "done"},
                            {"id": "seg2", "title": "二", "status": "failed"},
                        ],
                    },
                ]
            },
            msg_key="director:plan",
        )

    def test_a_finished_run_requeues_with_failed_segments_only(self):
        self.park_finished()
        with patch.object(doc_api, "_execute", new_callable=AsyncMock) as execute:
            resp = self.client.post(f"/api/doc/runs/{self.rid}/retry")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["retry_segments"], ["seg2"], "只点名失败段")
        execute.assert_awaited_once()
        retry_arg = execute.await_args.args[2]
        self.assertEqual(retry_arg["segment_ids"], ["seg2"])
        self.assertEqual(retry_arg["prev_approved"], ["seg1"], "已通过段要带上，汇报总数用")
        self.assertIn("segments", retry_arg["state"])
        self.assertEqual(db.get_run(self.rid)["status"], "running")

    def test_a_running_run_cannot_be_retried(self):
        db.update_run(self.rid, status="running")
        resp = self.client.post(f"/api/doc/runs/{self.rid}/retry")
        self.assertEqual(resp.status_code, 409)

    def test_no_failed_segments_means_nothing_to_continue(self):
        self.park_finished(status="done")
        # 清单上两段都完成了：没有需要继续的东西。
        plan = db.get_message_by_key(self.rid, "director:plan")
        produce = next(s for s in plan["payload"]["steps"] if s["id"] == "produce")
        for seg in produce["segments"]:
            seg["status"] = "done"
        db.add_message(
            self.rid, role="agent", agent="director", kind="plan",
            content="已完成 3/3", payload=plan["payload"], msg_key="director:plan",
        )
        resp = self.client.post(f"/api/doc/runs/{self.rid}/retry")
        self.assertEqual(resp.status_code, 409)
        self.assertIn("没有未完成的段", resp.text)

    def test_missing_state_cannot_resume(self):
        """没有持久化 state 的老任务没法断点重跑——如实拒绝，不猜。"""
        self.park_finished()
        db.update_run(self.rid, state={})
        resp = self.client.post(f"/api/doc/runs/{self.rid}/retry")
        self.assertEqual(resp.status_code, 409)
        self.assertIn("分段存档", resp.text)


if __name__ == "__main__":
    unittest.main()
