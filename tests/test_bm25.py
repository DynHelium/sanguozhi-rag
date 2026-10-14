"""rag/bm25.py 的测试: 分词, 建索引, idf, 打分, 去问句词.

    uv run python -m tests.test_bm25

不依赖 pytest. 期望分数是手算的, 算式写在各测试的注释里. 真实数据用例需要 data/processed/chunks.jsonl
(先跑 rag.chunk), 没有则跳过.
"""
import json
import math

from tests._run import Skip
from rag import bm25, config

DOCS = [
    ["孙策", "策字", "字伯", "伯符"],
    ["策为", "为许", "许贡", "贡客", "客所", "所杀"],
    ["孙策", "孙坚", "坚子"],
]


def _close(a, b):
    return abs(a - b) < 1e-9


# ---------- tokenize ----------

def test_tokenize_bigrams():
    assert bm25.tokenize("孙策字伯符") == ["孙策", "策字", "字伯", "伯符"]
    assert bm25.tokenize("孙策，孙坚子") == ["孙策", "孙坚", "坚子"], "标点断开, 不能产生 '策孙'"


def test_tokenize_single_char_and_non_han():
    assert bm25.tokenize("曰：策") == ["曰", "策"], "单字段输出单字"
    assert bm25.tokenize("兴平元年 194年") == ["兴平", "平元", "元年", "年"], "数字和空格丢掉"
    assert bm25.tokenize("") == []
    assert bm25.tokenize("，。") == []


def test_tokenize_keeps_duplicates():
    assert bm25.tokenize("曹操曹操") == ["曹操", "操曹", "曹操"]


# ---------- build_index ----------

def test_build_index_shape():
    idx = bm25.build_index(DOCS)
    assert idx["n_docs"] == 3
    assert idx["doc_len"] == [4, 6, 3]
    assert _close(idx["avgdl"], 13 / 3)
    assert idx["postings"]["孙策"] == {0: 1, 2: 1}
    assert idx["postings"]["许贡"] == {1: 1}
    assert "曹操" not in idx["postings"]


def test_build_index_counts_repeats():
    idx = bm25.build_index([["曹操", "操曹", "曹操"], ["曹操"]])
    assert idx["postings"]["曹操"] == {0: 2, 1: 1}
    assert idx["postings"]["操曹"] == {0: 1}, "不出现的文档不存 0"


def test_build_index_empty():
    idx = bm25.build_index([])
    assert idx["n_docs"] == 0 and idx["doc_len"] == [] and idx["avgdl"] == 0.0 and idx["postings"] == {}


# ---------- idf ----------

def test_idf():
    assert _close(bm25.idf(3, 2), math.log(1.6))          # ln(1 + 1.5/2.5)
    assert _close(bm25.idf(3, 1), math.log(1 + 2.5 / 1.5))
    assert bm25.idf(1000, 1000) > 0, "Lucene 版本永远为正"
    assert bm25.idf(1000, 1) > bm25.idf(1000, 10) > bm25.idf(1000, 500), "越稀有越大"


# ---------- score_all ----------

def _expected(tf, dl, avgdl, idf_value, k1=1.2, b=0.75):
    return idf_value * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))


def test_score_all_values():
    idx = bm25.build_index(DOCS)
    s = bm25.score_all(idx, ["孙策"])
    assert set(s) == {0, 2}, "只返回命中的文档"
    assert _close(s[0], _expected(1, 4, 13 / 3, math.log(1.6)))   # 约 0.485
    assert _close(s[2], _expected(1, 3, 13 / 3, math.log(1.6)))   # 约 0.538
    assert s[2] > s[0], "同样 tf, 短文档分高"


def test_score_all_sums_terms():
    idx = bm25.build_index(DOCS)
    s = bm25.score_all(idx, ["孙策", "许贡"])
    assert set(s) == {0, 1, 2}
    assert _close(s[1], _expected(1, 6, 13 / 3, math.log(1 + 2.5 / 1.5)))
    double = bm25.score_all(idx, ["孙策", "孙策"])
    assert _close(double[0], 2 * s[0]), "重复的查询词按次数累加"


def test_score_all_params_and_unknown():
    idx = bm25.build_index(DOCS)
    assert bm25.score_all(idx, ["曹操"]) == {}, "全书没有的词跳过"
    assert bm25.score_all(idx, []) == {}
    s = bm25.score_all(idx, ["孙策"], k1=1.2, b=0.0)
    assert _close(s[0], s[2]), "b=0 时不看文档长度"


# ---------- strip_question_words ----------

def test_strip_question_words_breaks_bigrams():
    q = bm25.strip_question_words("陈宫是谁？")
    assert bm25.tokenize(q) == ["陈宫"], "问句词换成断点, 不留 '宫是'"
    assert bm25.tokenize(bm25.strip_question_words("曹丕的庙号是什么？")) == ["曹丕", "庙号"]


# ---------- 真实数据 ----------

def test_real_chunks_smoke():
    path = config.DATA_DIR / "processed" / "chunks.jsonl"
    if not path.exists():
        raise Skip("没有 chunks.jsonl")
    with open(path, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    search = bm25.bm25_retriever(chunks)
    top = search("毌丘俭字仲恭", 3)
    assert top and "28-0005" in top[0]["sources"], f"毌丘儉传首段应排第一, 实际 {[c['id'] for c in top]}"



def test_strip_keeps_names_with_question_chars():
    """姓名里含问句字的人 (是仪) 不被拆开; 其他问句词照删."""
    old = bm25._protected_names
    bm25._protected_names = lambda: ("是仪",)
    try:
        assert bm25.strip_question_words("是仪是谁").split() == ["是仪"]
        assert bm25.strip_question_words("陈宫是谁").split() == ["陈宫"]
    finally:
        bm25._protected_names = old


def test_retriever_where_filters_before_top_k():
    """给了 where 时先筛选再取前 k: 范围外分数最高的块不占名额, 范围内排第二的块排到第一."""
    chunks = [{"id": "a", "tier": 1, "search": "孙策孙策孙策"}, {"id": "b", "tier": 2, "search": "孙策和周瑜"}]
    search = bm25.bm25_retriever(chunks)  # ty: ignore[invalid-argument-type]
    assert [c["id"] for c in search("孙策", 1)] == ["a"]
    assert [c["id"] for c in search("孙策", 1, lambda c: c["tier"] == 2)] == ["b"]


if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
