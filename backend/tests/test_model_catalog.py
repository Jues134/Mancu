"""模型能力分类的回归用例。

分类错了不会在设置页报错，只会在生成阶段以「模型不支持图片」「不是视频模型」
这种远处的方式炸出来，所以每个判断都固定成一条用例。
"""

import unittest

from app import model_catalog as mc


class ClassifyTests(unittest.TestCase):
    def test_minimax_chat_models_are_text(self):
        """MiniMax 首先是聊天厂商。裸的 minimax 标记曾把三个聊天模型塞进生视频。"""
        for mid in ("trae/minimax-m2.7", "trae/minimax-m3.1-flash", "minimax-m3"):
            self.assertEqual(mc.classify(mid)[0], mc.CAP_TEXT, mid)

    def test_minimax_video_products_are_still_video(self):
        for mid in ("MiniMax-H3", "minimax-video-01", "hailuo-02"):
            self.assertEqual(mc.classify(mid)[0], mc.CAP_VIDEO, mid)

    def test_vision_suffix_is_not_pinned_to_4v(self):
        """视觉后缀是一代一代的：写死 4v，glm-5v 就会被当成文本模型。"""
        for mid in ("trae/glm-5v-turbo", "glm-4v-flash", "qwen-2.5-vl"):
            self.assertEqual(mc.classify(mid)[0], mc.CAP_VISION, mid)

    def test_image_family_is_image(self):
        for mid in ("Kwai-Kolors/Kolors", "cogview-4-250304", "flux-1.1-pro"):
            self.assertEqual(mc.classify(mid)[0], mc.CAP_IMAGE, mid)

    def test_unknown_id_falls_back_to_text(self):
        """不认识的名字按文本处理：猜错的代价最小，第一次调用就会失败。"""
        self.assertEqual(mc.classify("big-pickle")[0], mc.CAP_TEXT)

    def test_build_keeps_every_scanned_id(self):
        """扫到多少个就要在界面里出现多少个——过滤掉会变成「扫描成功但选不到」。"""
        ids = ["trae/glm-5", "trae/glm-5v-turbo", "Kwai-Kolors/Kolors"]
        cat = mc.build(ids)
        self.assertEqual([e.id for e in cat.entries], ids)


if __name__ == "__main__":
    unittest.main()
