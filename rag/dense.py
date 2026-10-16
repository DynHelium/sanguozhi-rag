"""Dense retrieval: bge-m3 embeds each chunk once (cached as an .npy matrix under data/index/), queries are embedded at
search time, and the top k chunks come from one matrix-vector product over L2-normalised rows (cosine = dot product).
Complements BM25: it matches meaning rather than shared characters, e.g. a modern "去世" (died) against a classical "薨".

向量检索: bge-m3 把文字编码成向量, numpy 算余弦相似度取前 k.

    uv run python -m tests.test_dense          # numpy 部分不需要模型; 模型部分没装好时跳过
    uv run python -m rag.dense                 # 原文切块编码, 缓存到 data/index/
    uv run python -m rag.dense --trans         # 白话切块编码 (缓存名 "trans"; 先用 ingest.trans_chunks 生成 chunks-trans.jsonl)
    uv run python -m eval.run_retrieval -r dense

模块分两部分: 前半是模型与 numpy 的基本操作 (load_model 加载模型并选设备, encode 把一批文字编码成每行
长度为 1 的向量矩阵, l2_normalize, cosine_top_k 取最相似的前 k 行, save_matrix / load_matrix 存读矩阵);
后半 "文本准备, 缓存, 接线" 用它们给一组 chunk 建向量索引, 缓存到 data/index/, 并包成检索器 (dense_retriever).

数据怎么流动 (以 3 个 chunk, 向量只有 4 维为例; bge-m3 实际是 1024 维):

    texts = ["孙策字伯符", "策为许贡客所杀", "孙策，孙坚子"]     list[str], 每个 chunk 的 search 繁转简
        │ encode(model, texts)
        ▼
    matrix = np.ndarray, shape (3, 4), dtype float32, 每行长度 (L2 范数) = 1
        [[0.50, 0.50, 0.50, 0.50],      <- 第 0 行就是 texts[0] 的向量, 行号 = chunk 在列表里的序号
         [0.00, 0.60, 0.80, 0.00],
         [0.70, 0.10, 0.70, 0.10]]
        │ save_matrix -> data/index/dense-BAAI__bge-m3.npy   (下次直接 load_matrix, 不用重新编码)

    query = "孙策是怎么死的"
        │ encode(model, [query])[0]
        ▼
    q = np.ndarray, shape (4,), 长度 = 1
        │ cosine_top_k(matrix, q, k=2)
        ▼
    [(1, 0.83), (2, 0.61)]           list[tuple[int, float]], (行号, 余弦相似度), 从高到低 (数字只是示意)

为什么每行先归一成长度 1: 余弦相似度 = 点积 / (两个长度相乘). 长度都是 1 时余弦就等于点积,
全书一次 matrix @ q 就算完, 不用每次除长度.

和 BM25 的区别: BM25 比 "有没有相同的字"; 向量比 "意思像不像", 所以 "去世" 和 "薨" 有机会对上.
文本用文言原文 (繁转简后的 search, 与 BM25 同一份, 遵守 "入库和查询走同一套归一化"). 查询保留整句,
不去问句词 (strip_question_words 是 BM25 专用的; 向量模型要整句的语义). 白话译文另建一套向量 (name="trans").

库用 sentence-transformers, 只取 dense 向量: 接口简单, 换 embedding 模型不改代码; FlagEmbedding 多出的
sparse / colbert 向量用不到. 依赖在 pyproject.toml 的 index extra (安装见 README); numpy 在基础依赖里, 所以没装
index extra 时评测脚本也能 import 本模块. 模型第一次加载时从 Hugging Face 下载 (bge-m3 约 2.3GB), 之后离线可用.
设备自动选, 优先级 cuda > mps > cpu; Apple Silicon (mps) 上编码全书 4365 块约 5 分钟. bge-m3 最长支持 8192 token,
每批按最长的一条补齐: 原文切块最长 511 字, 用默认截断长度; 白话切块更长, 截到 TRANS_MAX_SEQ (按缓存名查 MAX_SEQ), 理由见
TRANS_MAX_SEQ 的注释.
"""
from __future__ import annotations
import hashlib
import json
import os
import sys
import time
import numpy.typing as npt
import numpy as np
from pathlib import Path

from rag import config
from rag.normalize import to_simplified

from typing import TYPE_CHECKING, TypedDict

from rag.schema import ChunkDict, Search, Where

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# ---------- 模型与 numpy ----------


def load_model(name: str = config.EMBED_MODEL, device: str | None = None) -> SentenceTransformer:
    """加载 embedding 模型 (sentence-transformers). 本地有缓存就离线加载, 没有才联网下载.

    device 为 None 时自动选: 有 cuda 用 cuda, 有 mps 用 mps, 否则 cpu. 加载后打印一行用了哪个设备.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    try:
        model = SentenceTransformer(name, device=device, local_files_only=True)
    except OSError:  # 本地缓存没有这个模型 (实测 local_files_only 时抛 OSError), 联网下载
        model = SentenceTransformer(name, device=device)
    print(f"Using Device {device} to load model")
    return model


def encode(model: SentenceTransformer, texts: list[str], batch_size: int = 32) -> npt.NDArray[np.float32]:
    """把一批文字编码成向量矩阵.

    返回 np.ndarray, shape (len(texts), 维数), dtype float32, 第 i 行对应 texts[i], 每行 L2 范数为 1
    (用库自带的 normalize_embeddings). 按 batch_size 分批喂给模型, 因为全书几千块一次喂进去显存放不下.
    """
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, precision="float32")


def l2_normalize(matrix: npt.NDArray) -> npt.NDArray:
    """把二维矩阵的每一行除以它自己的 L2 范数 (sqrt(各元素平方和)), 返回新矩阵, 不改原矩阵.

    例: [[3, 4], [0, 2]] -> [[0.6, 0.8], [0, 1]]
    全 0 的行保持全 0 (分母按 1 算), 不会产生 nan.
    """
    l2 = np.sum(matrix * matrix, axis = 1, keepdims = True) ** (1 / 2)
    return matrix / np.where(l2 == 0, 1, l2)


def cosine_top_k(matrix: npt.NDArray, query_vec: npt.NDArray, k: int) -> list[tuple[int, float]]:
    """matrix 每行和 query_vec 都已归一. 返回相似度最高的 k 行: [(行号, 相似度), ...], 从高到低.

    相似度就是点积, 一次矩阵乘法算完全部行. 同分时行号小的在前; k 大于行数时返回全部行. 返回的是 Python 的
    int 和 float 而不是 numpy 类型, 方便把结果存成 json. 全排序 4000 行只要几毫秒, 所以直接全排序, 不用只排前 k 的写法.
    """
    res = np.matmul(matrix, query_vec).tolist()
    with_row_index = sorted([(i, score) for i, score in enumerate(res)], key = lambda x: (-x[1], x[0]))
    return with_row_index[:k]



def save_matrix(path: Path | str, matrix: npt.NDArray) -> None:
    """把矩阵存成 .npy 文件. path 的父目录不存在时先建好.
    先写同目录的临时文件再 os.replace 换上: 别的进程正 mmap 着旧缓存时 (见 load_matrix), 旧文件不会被截断, 它继续读旧内容."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, matrix)
    os.replace(tmp, path)


def load_matrix(path: Path | str) -> npt.NDArray:
    """读回 save_matrix 存的矩阵, dtype 和 shape 与存之前一样.
    macOS / Linux 上用 mmap 只读打开 (np.memmap): 不把整个文件复制进进程内存, 几个进程 (并行评测, 网页聊天) 同时用同一份缓存时
    共享操作系统的页缓存; 矩阵只读, 检索只做 matrix @ q; 重建缓存时 save_matrix 换上新文件, 不改旧文件.
    Windows 上整份读进内存: 那里被 mmap 打开的文件不能删除, 也不能被 os.replace 覆盖, 缓存重建和临时目录清理都会失败."""
    return np.load(path, mmap_mode=None if os.name == "nt" else "r")


# ---------- 文本准备, 缓存, 接线 ----------


def chunk_text(chunk: ChunkDict) -> str:
    """给 embedding 用的文字: search 字段繁转简 (与 BM25 同一份文本)."""
    return to_simplified(chunk["search"])


def cache_paths(model_name: str = config.EMBED_MODEL, name: str = "") -> tuple[Path, Path]:
    """缓存文件: 向量 .npy + 说明 .json. 模型名里的 "/" 换成 "__" 当文件名.
    name 区分不同的文档集合, 免得互相覆盖: 原文切块 "" (dense-BAAI__bge-m3.npy), 白话切块 "trans"
    (dense-trans-BAAI__bge-m3.npy)."""
    stem = config.INDEX_DIR / ("dense-" + (f"{name}-" if name else "") + model_name.replace('/', '__'))
    return stem.with_suffix(".npy"), stem.with_suffix(".json")


class CacheKey(TypedDict):
    """缓存说明文件 (.json) 的内容, 见 cache_key."""
    version: int
    model: str
    max_seq: int
    n: int
    sha1: str


CACHE_VERSION = 1  # 缓存格式版本: 编码方式变了 (如换归一化) 而文字没变时, 加一让旧缓存失效
DEFAULT_MAX_SEQ = 8192  # bge-m3 的默认截断长度 (token)
TRANS_MAX_SEQ = 1024  # 白话切块的截断长度 (token). 约 1500 汉字, 比绝大多数白话 chunk 长, 几乎不丢内容;
                      # 用默认 8192 时每批按最长样本补齐, batch 32 要 29GB 内存
MAX_SEQ = {"trans": TRANS_MAX_SEQ}  # 缓存名 -> 截断长度; 不在表里的用 DEFAULT_MAX_SEQ


def cache_key(chunks: list[ChunkDict], model_name: str, max_seq: int = DEFAULT_MAX_SEQ) -> CacheKey:
    """判断缓存还能不能用: 格式版本, 模型名, 截断长度, chunk 数, 全部 chunk 的 id 和文字的指纹.
    任何一项变了就重新编码 (重新切块, 改归一化, 换模型, 改截断长度都会触发)."""
    h = hashlib.sha1()
    for c in chunks:
        h.update(c["id"].encode("utf-8"))
        h.update(chunk_text(c).encode("utf-8"))
    return {"version": CACHE_VERSION, "model": model_name, "max_seq": max_seq, "n": len(chunks), "sha1": h.hexdigest()}


def build_or_load(chunks: list[ChunkDict], model: SentenceTransformer | None = None, model_name: str = config.EMBED_MODEL,
                  name: str = "") -> tuple[npt.NDArray, SentenceTransformer | None]:
    """缓存有效就读缓存, 否则加载模型, 编码全部 chunk, 写缓存. 返回 (matrix, model 或 None).
    截断长度按 name 查 MAX_SEQ (不在表里用 DEFAULT_MAX_SEQ); 传入的 model 的 max_seq_length 也会被设成这个长度."""
    npy, meta = cache_paths(model_name, name)
    max_seq = MAX_SEQ.get(name, DEFAULT_MAX_SEQ)
    key = cache_key(chunks, model_name, max_seq)
    if npy.exists() and meta.exists() and json.loads(meta.read_text(encoding="utf-8")) == key:
        return load_matrix(npy), model
    model = model or load_model(model_name)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = max_seq
    t = time.time()
    matrix = encode(model, [chunk_text(c) for c in chunks])
    print(f"编码 {len(chunks)} 个 chunk 用时 {time.time() - t:.0f} 秒", file=sys.stderr)
    save_matrix(npy, matrix)
    meta.write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    return matrix, model


def cache_ok(chunks: list[ChunkDict], name: str = "", model_name: str = config.EMBED_MODEL) -> bool:
    """缓存存在且和这批 chunk 对得上 (不加载模型)."""
    npy, meta = cache_paths(model_name, name)
    key = cache_key(chunks, model_name, MAX_SEQ.get(name, DEFAULT_MAX_SEQ))
    return npy.exists() and meta.exists() and json.loads(meta.read_text(encoding="utf-8")) == key


def dense_retriever(chunks: list[ChunkDict], name: str = "") -> Search:
    """返回 search(query, k, where=None) -> list[chunk dict], 按相似度从高到低; 给了 where 时只在 where 为 True 的块里排.
    查询只做繁转简, 整句编码. 模型在第一次查询时才加载 (缓存命中时建索引不需要模型).
    name 见 cache_paths: 给全书原文以外的文档集合 (如译文) 建索引时要传, 否则会覆盖原文的缓存."""
    matrix, model = build_or_load(chunks, name=name)
    state: dict[str, SentenceTransformer | None] = {"model": model}

    def search(query: str, k: int, where: Where | None = None) -> list[ChunkDict]:
        if state["model"] is None:
            state["model"] = load_model()
        q = encode(state["model"], [to_simplified(query)])[0]
        if not where:
            return [chunks[i] for i, _ in cosine_top_k(matrix, q, k)]
        sims = np.matmul(matrix, q).tolist()  # 全部行算一次 (毫秒级), 再只在范围内的行里排, 不复制子矩阵
        rows = sorted((i for i, c in enumerate(chunks) if where(c)), key=lambda i: (-sims[i], i))
        return [chunks[i] for i in rows[:k]]
    return search


def main() -> None:
    """uv run python -m rag.dense            原文切块 chunks.jsonl -> 缓存名 ""
       uv run python -m rag.dense --trans    白话切块 chunks-trans.jsonl -> 缓存名 "trans" (截到 TRANS_MAX_SEQ)"""
    trans = "--trans" in sys.argv[1:]
    path = config.DATA_DIR / "processed" / ("chunks-trans.jsonl" if trans else "chunks.jsonl")
    with open(path, encoding="utf-8") as f:
        chunks: list[ChunkDict] = [json.loads(line) for line in f]
    name = "trans" if trans else ""
    matrix, _ = build_or_load(chunks, name=name)
    npy, _ = cache_paths(name=name)
    print(f"{len(chunks)} 个 chunk -> {npy} shape {matrix.shape}")


if __name__ == "__main__":
    main()
