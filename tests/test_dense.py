"""rag/dense.py 的测试: 归一化, 余弦 top-k, 向量缓存读写, 模型编码.

    uv run python -m tests.test_dense

不依赖 pytest. numpy 部分和缓存部分 (用假的 encode) 不需要模型; 模型部分需要 `uv sync --extra index`
和本地的 bge-m3 缓存 (先跑一次 rag.dense, 或设 ALLOW_MODEL_DOWNLOAD=1 让测试去下载), 不满足时跳过, 不算失败.
"""
import importlib.util
import os
import tempfile
from pathlib import Path

from tests._run import Skip
from rag import dense

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    import numpy as np


def _need_numpy():
    if not HAS_NUMPY:
        raise RuntimeError("没有 numpy: 先 uv sync")


# ---------- l2_normalize ----------

def test_l2_normalize_rows():
    _need_numpy()
    m = np.array([[3, 4], [0, 2]], dtype=np.float32)
    out = dense.l2_normalize(m)
    assert np.allclose(out, [[0.6, 0.8], [0, 1]])
    assert np.allclose(m, [[3, 4], [0, 2]]), "不要改原矩阵"


def test_l2_normalize_zero_row():
    _need_numpy()
    out = dense.l2_normalize(np.array([[0, 0], [1, 1]], dtype=np.float32))
    assert not np.isnan(out).any(), "全 0 行不能变成 nan"
    assert np.allclose(out[0], [0, 0])
    assert np.isclose(np.linalg.norm(out[1]), 1.0)


# ---------- cosine_top_k ----------

M = None
if HAS_NUMPY:
    M = np.array([[1, 0, 0], [0, 1, 0], [0.6, 0.8, 0], [0, 0, 1]], dtype=np.float32)


def test_cosine_top_k_order():
    _need_numpy()
    q = np.array([0, 1, 0], dtype=np.float32)
    top = dense.cosine_top_k(M, q, 2)
    assert [i for i, _ in top] == [1, 2]
    assert abs(top[0][1] - 1.0) < 1e-6 and abs(top[1][1] - 0.8) < 1e-6


def test_cosine_top_k_ties_and_k():
    _need_numpy()
    q = np.array([0, 0, 0], dtype=np.float32)
    assert [i for i, _ in dense.cosine_top_k(M, q, 3)] == [0, 1, 2], "同分时行号小的在前"
    assert len(dense.cosine_top_k(M, np.array([1, 0, 0], dtype=np.float32), 100)) == 4, "k 超过行数时返回全部"


def test_cosine_top_k_python_types():
    _need_numpy()
    i, s = dense.cosine_top_k(M, np.array([1, 0, 0], dtype=np.float32), 1)[0]
    assert type(i) is int and type(s) is float, f"要 Python int/float, 实际 {type(i).__name__}/{type(s).__name__}"


# ---------- save_matrix / load_matrix ----------

def test_save_load_roundtrip():
    _need_numpy()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sub" / "m.npy"
        dense.save_matrix(path, M)
        back = dense.load_matrix(path)
        assert back.dtype == M.dtype and back.shape == M.shape and np.array_equal(back, M)


# ---------- 缓存 (用假 encode) ----------

def test_cache_rebuilds_when_chunks_change():
    _need_numpy()
    calls = []

    def fake_encode(model, texts, batch_size=32):
        calls.append(len(texts))
        return dense.l2_normalize(np.ones((len(texts), 3), dtype=np.float32))

    chunks = [{"id": "01-0000", "search": "太祖"}, {"id": "01-0001", "search": "孫策"}]
    old = (dense.encode, dense.config.INDEX_DIR)
    with tempfile.TemporaryDirectory() as d:
        dense.encode, dense.config.INDEX_DIR = fake_encode, Path(d)
        try:
            dense.build_or_load(chunks, model="fake")
            m, _ = dense.build_or_load(chunks, model="fake")
            assert calls == [2] and m.shape == (2, 3), "第二次应读缓存, 不重新编码"
            dense.build_or_load(chunks + [{"id": "01-0002", "search": "孫權"}], model="fake")
            assert calls == [2, 3], "chunk 变了要重新编码"
        finally:
            dense.encode, dense.config.INDEX_DIR = old


# ---------- 模型 (需要部署) ----------

def test_model_encode():
    """真实 bge-m3 编码: 形状, dtype, 每行单位长度, 同一段文字结果相同.
    没装 index extra 时跳过; 本地没有模型缓存时也跳过 (免得跑测试时下载 2.3GB), 设 ALLOW_MODEL_DOWNLOAD=1 才下载."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError as e:
        raise Skip(f"没装 index extra, {e}")
    if try_to_load_from_cache(dense.config.EMBED_MODEL, "config.json") is None and not os.environ.get("ALLOW_MODEL_DOWNLOAD"):
        raise Skip(f"本地没有 {dense.config.EMBED_MODEL} 的缓存 (设 ALLOW_MODEL_DOWNLOAD=1 才下载)")
    try:
        model = dense.load_model()
    except (ImportError, ModuleNotFoundError) as e:
        raise Skip(f"还没部署, {e}")
    m = dense.encode(model, ["孙策字伯符", "孙策字伯符", "今日天气晴朗"])
    assert m.shape[0] == 3 and m.ndim == 2 and m.dtype == np.float32, f"shape {m.shape} dtype {m.dtype}"
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-3), "每行长度应为 1"
    assert np.allclose(m[0], m[1], atol=1e-4), "同一段文字编码结果应相同"



def test_cache_key_includes_max_seq():
    """截断长度不同, 缓存键不同: 改了截断长度不会误用旧向量."""
    chunks = [{"id": "01-0000", "search": "太祖"}]
    assert dense.cache_key(chunks, "m", 1024) != dense.cache_key(chunks, "m", 8192)
    assert dense.MAX_SEQ.get("trans") == dense.TRANS_MAX_SEQ

if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
