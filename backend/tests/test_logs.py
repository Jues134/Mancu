"""日志层自己的测试。

只测两件「坏了会很难看」的事：

  · **脱敏**。一条日志把 Key 写进文件，而这个文件是设计成「发给别人」的，
    所以掩码必须覆盖 header、JSON 正文和裸字符串三种写法。这是唯一一处
    出错会造成实际损害的地方。
  · **不重复装 handler**。测试和多入口都会调 `setup()`；装两遍就是每行日志
    打印两次，而那看起来像业务代码在重复执行。
"""

import json
import unittest

from app import logs


class MaskTests(unittest.TestCase):
    def test_bare_key_is_replaced(self):
        text = logs.mask("请求头里是 sk-ofm-testkey0000000000000000000000000000000")
        self.assertNotIn("testkey0000", text)
        self.assertIn("sk-***", text)

    def test_bearer_header_is_replaced(self):
        text = logs.mask("Authorization: Bearer abcdef1234567890abcdef")
        self.assertNotIn("abcdef1234567890", text)
        self.assertIn("***", text)

    def test_json_body_key_is_replaced(self):
        """供应商标配错误体里就带着 key，它是最常被整段打进日志的东西。"""
        text = logs.mask('{"api_key": "sk-abcdefghijklmnop12345", "model": "glm-4.5"}')
        self.assertNotIn("abcdefghijklmnop", text)
        self.assertIn('"model": "glm-4.5"', text, "只该掩掉 Key，不该连模型名一起吃掉")

    def test_plain_text_survives(self):
        """别把正常句子也掩掉：过度脱敏会让日志失去排错价值。"""
        text = "检索词：中共一大；召回 3 条；最高分 0.812"
        self.assertEqual(logs.mask(text), text)


class EventTests(unittest.TestCase):
    def test_event_roundtrip_and_run_filter(self):
        logs.event("test.tick", run_id="run-one", note="第一条")
        logs.event("test.tick", run_id="run-two", note="第二条")

        mine = logs.read_events(limit=50, run_id="run-one")
        self.assertTrue(mine, "刚写的事件应该读得回来")
        self.assertTrue(all(item["run"] == "run-one" for item in mine))
        self.assertEqual(mine[0]["event"], "test.tick")

    def test_event_masks_its_fields(self):
        logs.event("test.secret", run_id="run-secret", body="key=sk-abcdefghijklmnop12345")
        found = logs.read_events(limit=20, run_id="run-secret")
        self.assertTrue(found)
        self.assertNotIn("abcdefghijklmnop", json.dumps(found, ensure_ascii=False))

    def test_event_never_raises_on_odd_values(self):
        """事件字段来自任意异常对象；序列化失败不能反过来把调用方炸掉。"""
        class Weird:
            def __repr__(self) -> str:
                return "<weird>"

        logs.event("test.weird", run_id="run-weird", payload=Weird(), nested={"a": Weird()})


class SetupTests(unittest.TestCase):
    def test_setup_is_idempotent(self):
        import logging

        logs.setup()
        logs.setup()
        ours = [h for h in logging.getLogger().handlers if getattr(h, "_mancu", False)]
        self.assertEqual(len(ours), 3, "控制台 + app.log + error.log，装两遍就是六条")


if __name__ == "__main__":
    unittest.main()
