"""进程内向量化：不依赖任何外部服务，把一段中文切成可检索的向量。

为什么要内置
------------
这个项目的知识库只有两层召回：字面（主题检索）和语义（向量）。字面层是一段
纯本地扫描，永远能用；语义层过去要求一个 OpenAI 兼容的 `/embeddings` 端点，
于是「用户上传一本书、打开就能搜」这件事实际上不成立——没配模型时它静默退化，
配错了时它每次检索都报 404，而**书的正文块恰恰只在建索引时写库**，索引一失败
整本书在检索体系里就等于不存在。

所以这一层要能自带：一个 24 MB 的中文向量模型（bge-small-zh-v1.5，512 维）
放在程序目录里，用 onnxruntime 在进程内跑。没有 key、没有网络、没有第二个进程。

为什么是这个小模型
------------------
中文资料检索要的是「意思相近」而不是「模型很大」。bge-small-zh 在中文短查询
上明显强于同类 small 模型，512 维在一本 400 块的资料上做全表余弦足够快（毫秒
级）。整数量化版 24 MB，和 fp32 版（95 MB）在一个六万字书库上的 top-1 结果
一致，但安装包小四分之三——安装包体积是这个项目真实存在的约束。

输出维度和云端模型不同，这没关系：`studio_kb_chunks` 存了 `model` 与 `dim`，
换模型会自动触发全量重嵌，不会让两套向量混在一张表里比分数。
"""

from __future__ import annotations

import asyncio
import threading
import urllib.request
from pathlib import Path
from typing import Any

#: 存进 `studio_kb_chunks.model` 的名字。改了它 = 换模型 = 全量重嵌，这是对的。
MODEL_ID = "local:bge-small-zh-v1.5"

#: bge 中文模型的查询前缀。文档说得很明确：**查询**要加，**段落**不加。加了前缀
#: 的查询和不加前缀的正文在同一空间里比较，短查询的召回明显更稳（「建党一百年」
#: 这种说法本来就不在正文里逐字出现）。
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

#: 模型最大序列长度。超过的部分截断——一本书的段落目标长度是 320 字，
#: 真正会被截断的只有目录块。
MAX_TOKENS = 512

#: 一次前向的条数。太小则 Python 侧开销占主导，太大则峰值内存翻倍；
#: 16 条 × 512 token 在 CPU 上约 0.3 秒，是这台机器上量出来的。
BATCH = 16

#: 从哪儿取模型。ModelScope 放前面：国内直连，另外两个在这台机器上经常连不通。
_SOURCES = (
    "https://www.modelscope.cn/models/Xenova/bge-small-zh-v1.5/resolve/master/{path}",
    "https://hf-mirror.com/Xenova/bge-small-zh-v1.5/resolve/main/{path}",
    "https://huggingface.co/Xenova/bge-small-zh-v1.5/resolve/main/{path}",
)

#: 需要落地的两个文件：仓库里的路径 → 本地文件名。
_FILES = {
    "onnx/model_int8.onnx": "model.onnx",
    "tokenizer.json": "tokenizer.json",
}

#: 下载完整性下限。半截文件比没有文件更坏：onnxruntime 会在加载时报一句
#: protobuf 错误，而那句话不会让人想到「下载断了」。
_MIN_BYTES = {"model.onnx": 1_000_000, "tokenizer.json": 10_000}


def model_dir() -> Path:
    """模型放在程序目录里，不放 userdata。

    `userdata` 是用户的工作成果，备份、迁移、清理都绕着它走；模型是可重建的
    运行依赖，混进去会让「备份我的资料」变成「备份 24 MB 的二进制」。
    """
    import os

    override = os.environ.get("STUDIO_EMBED_MODEL_DIR", "").strip()
    if override:
        return Path(override)
    from ...config import BASE_DIR

    return Path(BASE_DIR) / "models" / "bge-small-zh-v1.5-onnx"


def files_ready(directory: Path | None = None) -> tuple[bool, str]:
    """模型文件齐不齐。只查文件，不加载——健康检查会高频调用它。"""
    root = directory or model_dir()
    for name, floor in _MIN_BYTES.items():
        path = root / name
        if not path.exists():
            return False, f"缺少模型文件 {name}"
        size = path.stat().st_size
        if size < floor:
            return False, f"模型文件 {name} 不完整（{size} 字节）"
    return True, ""


def _download_one(name: str, dest: Path) -> None:
    """按来源顺序试，第一个成功的落盘。临时文件 + 替换，避免半截文件。"""
    last_error = ""
    for template in _SOURCES:
        url = template.format(path=_source_path(name))
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=60) as response:
                if getattr(response, "status", 200) != 200:
                    last_error = f"{url} → HTTP {response.status}"
                    continue
                with open(tmp, "wb") as fh:
                    while True:
                        block = response.read(1 << 20)
                        if not block:
                            break
                        fh.write(block)
            if tmp.stat().st_size < _MIN_BYTES.get(name, 1):
                last_error = f"{url} → 下载不完整（{tmp.stat().st_size} 字节）"
                tmp.unlink(missing_ok=True)
                continue
            tmp.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001 - 换下一个源
            last_error = f"{url} → {type(exc).__name__}: {exc}"
            dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)
    raise RuntimeError(f"模型 {name} 下载失败：{last_error}")


def _source_path(name: str) -> str:
    for remote, local in _FILES.items():
        if local == name:
            return remote
    return name


def ensure_model(download: bool = True) -> tuple[Path | None, str]:
    """确保模型在本地，返回 `(目录, 失败原因)`。

    `download=False` 时只检查不联网——状态接口用它，避免一次页面刷新就发起
    24 MB 的下载。
    """
    root = model_dir()
    ok, why = files_ready(root)
    if ok:
        return root, ""
    if not download:
        return None, f"{why}，首次使用会自动下载"
    try:
        root.mkdir(parents=True, exist_ok=True)
        for local in _FILES.values():
            dest = root / local
            if dest.exists() and dest.stat().st_size >= _MIN_BYTES.get(local, 1):
                continue
            _download_one(local, dest)
    except Exception as exc:  # noqa: BLE001 - 下载失败只降级，不该让检索崩掉
        return None, f"{why}；自动下载失败：{exc}"
    ok, why = files_ready(root)
    return (root, "") if ok else (None, why)


class _Engine:
    """加载一次的推理引擎。进程内单例：模型加载约 1 秒，不能每次检索都付。"""

    _lock = threading.Lock()
    _session: Any = None
    _tokenizer: Any = None
    _dim = 0
    _error = ""

    @classmethod
    def get(cls) -> tuple[Any, Any, int, str]:
        with cls._lock:
            if cls._session is not None:
                return cls._session, cls._tokenizer, cls._dim, ""
            if cls._error:
                # 上次失败过，但文件可能已经补上了（下载完成后重试、换了目录）。
                # 无条件缓存错误会让「先失败、后修好」的进程永远回不来，只有重启
                # 才能恢复——而重启正是用户最不愿意做的那件事。
                if not files_ready()[0]:
                    return None, None, 0, cls._error
                cls._error = ""
            root, why = ensure_model()
            if root is None:
                cls._error = why
                return None, None, 0, why
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer

                tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
                tokenizer.enable_truncation(max_length=MAX_TOKENS)
                tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
                session = ort.InferenceSession(
                    str(root / "model.onnx"),
                    providers=["CPUExecutionProvider"],
                )
                shape = session.get_outputs()[0].shape
                dim = int(shape[-1]) if isinstance(shape[-1], int) else 0
                cls._session, cls._tokenizer, cls._dim = session, tokenizer, dim
                return session, tokenizer, dim, ""
            except Exception as exc:  # noqa: BLE001 - 缺依赖 / 文件坏了都走这里
                cls._error = f"本地向量模型加载失败：{type(exc).__name__}: {exc}"
                return None, None, 0, cls._error

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._session = None
            cls._tokenizer = None
            cls._dim = 0
            cls._error = ""


def _normalise(row: list[float]) -> list[float]:
    total = sum(value * value for value in row) ** 0.5
    if total <= 0:
        return row
    return [value / total for value in row]


class LocalEmbedder:
    """与 `Embedder` 同形：`model` / `base_url` / `embed()` / `embed_query()`。

    不继承 `Embedder` 是有意的：那个类的构造函数是「地址 + key + 模型名」，
    而这里没有地址也没有 key，套上去只会让人以为可以传参数调它。
    """

    def __init__(self) -> None:
        # 构造函数**不加载模型**：`health()` 每次刷新都会建一个，加载 24 MB 的
        # 会话要一秒，把它放进构造函数等于让状态接口变慢。
        self.model = MODEL_ID
        self.base_url = "local://onnx"
        self.provider = "local"

    def available(self) -> bool:
        """文件在不在。真加载一次留给第一次 `embed()`，那时才值得付这个成本。"""
        return files_ready()[0]

    @property
    def dim(self) -> int:
        return _Engine.get()[2]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_sync, list(texts))

    async def embed_query(self, text: str) -> list[float]:
        """查询要加 bge 的前缀，正文不加。见 `QUERY_PREFIX`。"""
        vectors = await self.embed([f"{QUERY_PREFIX}{text}"])
        return vectors[0] if vectors else []

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        session, tokenizer, dim, why = _Engine.get()
        if session is None:
            # 和云端嵌入失败同一种错误类型，上层只需要处理一种失败。
            from .embedder import EmbeddingError

            raise EmbeddingError(why or "本地向量模型不可用")

        import numpy as np

        names = {item.name for item in session.get_inputs()}
        out: list[list[float]] = []
        for start in range(0, len(texts), BATCH):
            batch = texts[start : start + BATCH]
            encoded = tokenizer.encode_batch(batch)
            ids = np.array([item.ids for item in encoded], dtype=np.int64)
            mask = np.array([item.attention_mask for item in encoded], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = session.run(None, feeds)[0]
            # bge 用 CLS 池化（第一 token），不是平均池化。用错池化不会报错，
            # 只会让相似度整体偏低，然后表现为「怎么搜都不太准」。
            out.extend(_normalise([float(v) for v in row]) for row in hidden[:, 0])
        if dim and any(len(row) != dim for row in out):
            raise RuntimeError("本地模型输出维度不一致")
        return out


def status() -> dict[str, Any]:
    """状态页要的东西：能不能用、用的什么模型、文件在哪、多大。"""
    root = model_dir()
    ok, why = files_ready(root)
    size = 0
    if ok:
        size = sum((root / name).stat().st_size for name in _MIN_BYTES)
    return {
        "available": ok,
        "model": MODEL_ID,
        "dir": str(root),
        "bytes": size,
        "reason": "" if ok else why,
    }
