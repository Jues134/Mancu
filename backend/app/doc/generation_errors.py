"""Turn provider failures into actionable messages.

The requirement: when the key or model does not support text-to-image,
image-to-video, or the endpoint shape differs, the user must be told *what is
unsupported* rather than shown a bare status code.

Current behaviour raises `文生图失败 HTTP 404：...` — accurate but useless: 404
could mean a wrong base URL, a model that does not exist, or an endpoint that the
provider does not implement. The user cannot tell, and the fix differs for each.

This module classifies a failure into a small set of causes with a next step:

  · endpoint_missing   — the provider has no such route (404 on the path)
  · model_unsupported  — the key works but the model is not available (404/400
                         naming the model, or "model not found")
  · auth               — 401/403, the key is wrong or expired
  · quota              — 429 with a balance/quota message
  · content            — the prompt was rejected (400 with a safety message)
  · unknown            — nothing recognisable; the raw body is included

Each message names the operation (生图 / 生视频), the model attempted, and what to
do — because the answer differs per case and a generic "调用失败" forces the user
to guess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


#: Operation labels, so a message says which capability failed.
OPERATION_LABELS = {
    "image": "文生图",
    "video": "图生视频",
    "status": "查询任务",
}


@dataclass
class Diagnosis:
    cause: str
    message: str
    hint: str

    def as_error_message(self) -> str:
        return f"{self.message}｜建议：{self.hint}"


def _extract_message(body: str) -> str:
    """Pull a provider's own explanation out of a JSON or text error body."""
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return body.strip()[:200]

    if not isinstance(data, dict):
        return body.strip()[:200]

    for path in (
        ("error", "message"),
        ("error",),
        ("message",),
        ("msg",),
        ("detail",),
    ):
        node: Any = data
        for key in path:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(key)
        if isinstance(node, str) and node.strip():
            return node.strip()[:200]

    # Alibaba-style nested errors: {"code": "...", "message": "..."}
    if isinstance(data.get("code"), str) and isinstance(data.get("message"), str):
        return f"{data['code']}: {data['message']}"[:200]

    return json.dumps(data, ensure_ascii=False)[:200]


def diagnose(
    *,
    operation: str,
    status: int,
    body: str,
    model: str = "",
    base_url: str = "",
) -> Diagnosis:
    """Classify one failed call."""
    label = OPERATION_LABELS.get(operation, operation)
    detail = _extract_message(body)
    lowered = detail.lower()
    model_note = f"（模型 {model}）" if model else ""
    where = f"（地址 {base_url}）" if base_url else ""

    # ---- auth ----
    if status in (401, 403):
        return Diagnosis(
            "auth",
            f"{label}被拒：Key 无效或无权限{model_note}",
            "在设置里重新填写该供应商的 API Key，确认它对这个模型有权限",
        )

    # ---- quota / billing ----
    if status == 429 or "余额" in detail or "quota" in lowered or "insufficient" in lowered:
        if "余额" in detail or "insufficient" in lowered:
            return Diagnosis(
                "quota",
                f"{label}失败：账户余额或额度不足{model_note}",
                "充值或换成有额度的供应商；该错误重试无用",
            )
        return Diagnosis(
            "quota",
            f"{label}被限流{model_note}",
            "降低并发或稍后重试；免费额度通常有每分钟次数限制",
        )

    # ---- the model is not offered ----
    #
    # Checked BEFORE the generic 404 branch: a provider that does not know a model
    # often answers 404 with "model not found", and reporting that as a missing
    # endpoint would send the user to check their base URL when the real fix is
    # the model name.
    if "model" in lowered and (
        "not exist" in lowered
        or "not found" in lowered
        or "unsupported" in lowered
        or "no access" in lowered
        or "not available" in lowered
    ):
        return Diagnosis(
            "model_unsupported",
            f"{label}失败：该供应商不提供模型 {model}",
            "在设置里改成该供应商支持的模型名，或换一家支持此能力的供应商",
        )

    # ---- the provider has no such endpoint ----
    if status == 404 and ("not found" in lowered or "resource not found" in lowered
                          or "no such" in lowered or not detail):
        return Diagnosis(
            "endpoint_missing",
            f"{label}失败：该供应商没有这个接口{model_note}{where}",
            "确认该供应商支持此能力，并检查设置里的接口地址是否填对"
            "（应形如 https://…/v1）",
        )

    # ---- the model exists but cannot do this operation ----
    if status == 400 and any(
        k in lowered
        for k in ("not support", "unsupported", "not capable", "invalid model",
                  "does not support", "不支持", "暂不开放", "no such capability")
    ):
        return Diagnosis(
            "capability_unsupported",
            f"{label}失败：模型 {model} 不支持这项能力",
            "换一个支持该能力的模型（如文生图需 cogview 类，图生视频需 cogvideox 类）",
        )

    # ---- the prompt was refused ----
    if status == 400 and any(k in lowered for k in ("sensitive", "违规", "敏感", "safety", "policy")):
        return Diagnosis(
            "content",
            f"{label}失败：提示词被内容审核拒绝{model_note}",
            "改写提示词中的人物或场景描述后重试",
        )

    # ---- parameters rejected ----
    if status in (400, 422):
        return Diagnosis(
            "bad_request",
            f"{label}请求参数被拒{model_note}：{detail}",
            "检查尺寸、参考图 URL 是否仍然有效（签名会过期）",
        )

    # ---- unknown ----
    return Diagnosis(
        "unknown",
        f"{label}失败 HTTP {status}{model_note}：{detail}",
        "把这条错误发回开发者，附上供应商名称与模型名",
    )
