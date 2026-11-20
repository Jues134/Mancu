"""Chunking and embedding — turning a chapter into retrievable, citable pieces.

Chunking choices, and why they are not arbitrary:

  · **Split on paragraph boundaries first.** A chunk that starts mid-sentence
    retrieves badly and cites worse ("第3段" would be a lie).
  · **Target 200–400 characters.** Smaller chunks lose the context needed to judge
    a claim; larger ones dilute the similarity score with unrelated sentences. Chinese
    history prose carries dense meaning, so the window is smaller than for English.
  · **Keep section and paragraph on every chunk.** This is what makes a retrieval
    result usable as evidence rather than as vague context.
  · **Carry the chapter title into each chunk's section.** A paragraph about "立储
    之争" alone is ambiguous; the same paragraph under "万历十五年" is not.

Embedding: one batched request per run where possible. Providers limit batch size,
so the client chunks the batch and retries on a 429 with backoff — the failure mode
here is a rate limit, not a logic error, and a naive retry would just fail again
sooner.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

import httpx

#: Target characters per chunk. Tuned for Chinese historical prose: dense enough
#: that 200 characters usually contain one checkable assertion, small enough that
#: a query about one event does not pull in the whole chapter.
TARGET_CHARS = 320
#: Hard ceiling, so a paragraph with no punctuation cannot produce a giant chunk.
MAX_CHARS = 520


@dataclass
class SplitConfig:
    target: int = TARGET_CHARS
    max_chars: int = MAX_CHARS


def split_block(block: str, cfg: SplitConfig) -> list[str]:
    """Split one section into target-sized pieces on sentence boundaries.

    Falls back to punctuation-insensitive splitting when the text has no sentence
    terminators (a list of clauses separated by commas, or text pasted without
    punctuation). Without that fallback a whole section becomes one oversized
    chunk, which defeats retrieval: a query about one event would return the entire
    chapter and the similarity score would be meaningless.

    Public because the knowledge base indexes a different shape of text (an entry's
    body plus its key points) through the same boundary rules. Duplicating the
    sentence/clause fallbacks there would let the two drift apart, and chunking is
    exactly the kind of code that rots when copied.
    """
    sentences = [s for s in re.split(r"(?<=[。！？；])", block) if s.strip()]
    if not sentences:
        return []
    out: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) <= cfg.target:
            current += sentence
            continue
        if current:
            out.append(current)
            current = ""
        while len(sentence) > cfg.max_chars:
            out.append(sentence[: cfg.max_chars])
            sentence = sentence[cfg.max_chars:]
        current = sentence
    if current:
        out.append(current)

    # Any piece still over the target is split further on commas, then hard-cut.
    refined: list[str] = []
    for piece in out:
        if len(piece) <= cfg.target:
            refined.append(piece)
            continue
        clauses = [c for c in re.split(r"(?<=[，、])", piece) if c.strip()]
        buf = ""
        for clause in clauses:
            if len(buf) + len(clause) <= cfg.target:
                buf += clause
            else:
                if buf:
                    refined.append(buf)
                while len(clause) > cfg.max_chars:
                    refined.append(clause[: cfg.max_chars])
                    clause = clause[cfg.max_chars:]
                buf = clause
        if buf:
            refined.append(buf)

    return [p.strip() for p in refined if p.strip()]


# ============================================================================
# Embedding client
# ============================================================================


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced."""


class Embedder:
    """Generate embeddings for text, with batching and a bounded retry.

    Deliberately provider-agnostic: any OpenAI-compatible `/embeddings` endpoint
    works. That matters because the project already supports several providers,
    and hard-coding one would make RAG unavailable whenever the user runs on a
    different vendor.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "embedding-3",
        batch_size: int = 16,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        #: 向量是从哪儿来的。状态页要区分「云端端点」和「内置模型」，因为两者
        #: 出问题时的修法完全不同：一个是改配置，一个是重下模型。
        self.provider = "cloud"

    async def embed_query(self, text: str) -> list[float]:
        """查询向量。单列一个方法是为了让内置模型能加自己的查询前缀，
        而调用方（`vector.search`）不必知道用的是哪一家。"""
        vectors = await self.embed([text])
        return vectors[0] if vectors else []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed every text, preserving order.

        Order preservation is load-bearing: the store zips chunks to vectors, and a
        reordered batch would attach the wrong vector to the wrong passage — a
        corruption that would never be noticed until a citation looked wrong.
        """
        if not texts:
            return []

        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors = await self._embed_batch(batch)
            if len(vectors) != len(batch):
                raise EmbeddingError(
                    f"向量数量不匹配：请求 {len(batch)} 条，返回 {len(vectors)} 条"
                )
            out.extend(vectors)
        return out

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/embeddings",
                        json={"model": self.model, "input": batch},
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}",
                        },
                    )
                if response.status_code == 200:
                    body = response.json()
                    data = sorted(
                        body.get("data") or [], key=lambda d: int(d.get("index", 0))
                    )
                    return [d["embedding"] for d in data]
                if response.status_code == 429:
                    # Rate limit: back off, then retry. The wait grows so a burst
                    # does not keep hammering the same limit.
                    await asyncio.sleep(2 ** attempt)
                    last_error = EmbeddingError(
                        f"限流（429）：{response.text[:120]}"
                    )
                    continue
                raise EmbeddingError(
                    f"HTTP {response.status_code}：{response.text[:160]}"
                )
            except EmbeddingError:
                raise
            except Exception as exc:  # noqa: BLE001 - transport errors are retryable
                last_error = exc
                await asyncio.sleep(1 + attempt)

        raise EmbeddingError(f"向量化失败：{last_error}")

    @classmethod
    def from_settings(cls, model_settings: Any = None) -> Any:
        """挑一个能用的向量化来源，按「显式 > 内置 > 没有」的顺序。

        **这里曾经有一个静默的错。** 旧版实现遍历所有供应商，只要它有 key 和
        base_url 就返回一个 embedder，模型名回落成 `embedding-3`（这是智谱的模型
        名）。于是「配了一个只做对话的网关」这种最常见的情况，会得到一个指向
        `{base_url}/embeddings` 的客户端——那个路由在网关上根本不存在，每次检索
        都是 `HTTP 404 {"detail":"Not Found"}`。更糟的是健康检查因此报告
        `configured=True`：界面上看起来「已配置向量模型」，实际上一调用就死。

        现在只有两种情况算「配了云端向量」：环境变量显式给了地址与模型名。其余
        一律交给内置模型（进程内 ONNX，见 `local_embedder`），它不需要任何配置
        就能工作——「用户上传一本书，打开就能搜」是这个库的前提，不能靠用户先去
        找一个 embedding 供应商。
        """
        cloud = cls._from_env()
        if cloud is not None:
            return cloud
        local = cls._local()
        if local is not None:
            return local
        return None

    @classmethod
    def _from_env(cls) -> "Embedder | None":
        """显式配置的云端向量端点。

        用环境变量而不是 providers.json：那份文件是「对话用哪个模型」的设置，
        往里塞第四个模型名（还只对某些供应商有效）会让设置页和实际行为脱节。
        向量服务是一个独立的、可选的基础设施，环境变量正是表达这件事的粒度。
        """
        import os

        base_url = (os.environ.get("STUDIO_EMBED_BASE_URL") or "").strip()
        model = (os.environ.get("STUDIO_EMBED_MODEL") or "").strip()
        if not base_url or not model:
            return None
        return cls(
            base_url=base_url,
            api_key=(os.environ.get("STUDIO_EMBED_API_KEY") or "").strip(),
            model=model,
        )

    @classmethod
    def _local(cls) -> Any:
        """内置模型。没有文件就是没有——绝不回落到「猜一个模型名发出去」。"""
        try:
            from .local_embedder import LocalEmbedder

            embedder = LocalEmbedder()
            return embedder if embedder.available() else None
        except Exception:  # noqa: BLE001 - 缺依赖就等于没装，不该拖垮检索
            return None
