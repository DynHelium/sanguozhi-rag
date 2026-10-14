"""BM25 检索: 字 bigram 倒排索引, 只用标准库实现.

    uv run python -m tests.test_bm25
    uv run python -m eval.run_retrieval -r bm25

输入是 chunk 列表, 对每块的 search 字段繁转简后建索引; 输出 search(query, k, where=None) -> 前 k 个 chunk (rag/schema.Search).
索引是普通 dict (BM25Index), 由 tokenize -> build_index 得到, 查询时 score_all 只遍历查询词的倒排表.
公式, 参数 (K1, B) 和 idf 的取法见 docs/DESIGN.md "BM25".

为什么用字 bigram: 文言没有现成的好分词器 (jieba 按现代汉语训练); 单字太泛, 两字一组能覆盖大多数人名,
地名, 官名 ("孙策" "官渡" "太守"). 跨词的 bigram ("策字") 是噪声, 由 idf 压低.

已知局限 (eval/retrieval.jsonl 里有对应的题):
    - 异体字 ("呉郡" 对 "吴郡") 不处理.
    - 别名 ("魏武帝") 不认, 由 rag/alias.expand_query 在查询侧扩展.
    - 白话对文言 ("去世" 对 "薨") 无能为力, 靠 dense 和白话索引.
"""
from rag.normalize import to_simplified
from collections import defaultdict
from functools import lru_cache
import math
from typing import TypedDict

from rag.schema import ChunkDict, Search, Where

K1 = 1.2
B = 0.75


class BM25Index(TypedDict):
    """build_index 的返回值, 字段含义见 build_index."""
    n_docs: int
    doc_len: list[int]
    avgdl: float
    postings: dict[str, dict[int, int]]


# 查询端去掉的白话问句词和虚词 (按检索评测挑的). 只作用于查询, 不碰文档.
# 长的在前, 免得 "什么" 先被 "么" 之类拆坏. 见 strip_question_words.
QUESTION_WORDS = ["为什么", "什么", "如何", "哪", "那", "谁", "怎", "么", "是", "的", "有", "了", "几", "吗", "呢", "过"]


def is_han(ch: str) -> bool:
    """单个字符是不是汉字 (CJK 统一汉字基本区 + 扩展 A 区, 如 "㶏"). 给 tokenize 用."""
    return "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿"


def tokenize(text: str) -> list[str]:
    """把一段 (已繁转简的) 文字切成字 bigram 列表, 按出现顺序, 重复的保留.

    只有连续的汉字 (is_han) 组成 bigram; 其他字符是断点, 本身丢掉. 只有一个字的汉字段输出单字,
    以免它完全搜不到.
        "孙策，孙坚子"     -> ["孙策", "孙坚", "坚子"]
        "兴平元年 194年"   -> ["兴平", "平元", "元年", "年"]
    """
    res = []
    prev = 0
    l = len(text)
    for i, ch in enumerate(text):
        if not is_han(ch):
            if i - prev > 1:
                res.extend([a+b for a, b in zip(text[prev:i-1], text[prev+1:i])])
            elif i > prev:
                res.append(text[prev])
            prev = i + 1
    if l - prev > 1:
        res.extend([a+b for a, b in zip(text[prev:-1], text[prev+1:])])
    elif l > prev:
        res.append(text[prev])
    return res


def build_index(docs: list[list[str]]) -> BM25Index:
    """由所有文档的词列表建索引. docs[i] 是第 i 个文档 tokenize 后的结果.

        build_index([["孙策", "策字"], ["孙策", "孙坚"]])
        -> {"n_docs": 2,
            "doc_len": [2, 2],                               每个文档的词数
            "avgdl": 2.0,                                    docs 为空时为 0.0
            "postings": {"孙策": {0: 1, 1: 1}, ...}}         词 -> {文档序号: 次数}, 不存 0
    """
    doc_len = [len(s) for s in docs]
    l = len(docs)
    postings: dict[str, dict[int, int]] = dict()
    for i, doc in enumerate(docs):
        for word in doc:
            if word not in postings:
                postings[word] = defaultdict(int)
            postings[word][i] += 1
    return BM25Index(
        n_docs = l,
        doc_len = doc_len,
        avgdl = sum(doc_len) / l if l else 0.0,
        postings = postings,
    )


def idf(n_docs: int, df: int) -> float:
    """词的稀有度 ln(1 + (n_docs - df + 0.5) / (df + 0.5)), 永远为正 (Lucene 的取法).

    n_docs 是文档总数, df 是含这个词的文档数 (1 <= df <= n_docs). 例: idf(3, 2) = ln(1.6) = 0.470.
    """
    return math.log1p((n_docs - df + 0.5) / (df + 0.5))


def score_all(index: BM25Index, query_tokens: list[str], k1: float = K1, b: float = B) -> dict[int, float]:
    """对查询算每个文档的 BM25 分数 (公式见 docs/DESIGN.md "BM25").

    返回 {文档序号: 分数}, 只含至少命中一个查询词的文档; query_tokens 为空时返回 {}.
    索引里没有的查询词跳过; 重复的查询词按次数累加 (与 rank_bm25 等常见实现一致).
        score_all(build_index([["孙策", "策字"], ["许贡"]]), ["孙策"])  -> {0: <正数>}
    """
    res: defaultdict[int, float] = defaultdict(float)
    for token in query_tokens:
        if token in index["postings"]:
            doc_index_to_count = index["postings"][token]
            df = len(doc_index_to_count)
            idf_score = idf(index["n_docs"], df)
            for doc_index, tf in doc_index_to_count.items():
                score = idf_score * tf * (k1 + 1)
                res[doc_index] += score / (tf + k1 * (1 - b + b * index["doc_len"][doc_index] / index["avgdl"]))
    return res


def strip_question_words(query: str) -> str:
    """把查询里的 QUESTION_WORDS 换成空格, 在 tokenize 之前调用.

    白话问句词在文言里很稀有, idf 反而很高, 会把无关的短文档顶上来.
    换成空格而不是删掉, 是为了让它成为断点, 不和两边的字拼出跨词 bigram ("宫是").
    别名表里姓名含这些字的人名 (如 "是仪") 原样保留.
        "陈宫是谁？"  -> "陈宫  ？"
        "是仪是谁"    -> "是仪  "
    """
    names = _protected_names()
    held = [n for n in names if n in query]
    for i, n in enumerate(held):
        query = query.replace(n, f"\0{i}\0")
    for w in QUESTION_WORDS:
        query = query.replace(w, " ")
    for i, n in enumerate(held):
        query = query.replace(f"\0{i}\0", n)
    return query


@lru_cache
def _protected_names() -> tuple[str, ...]:
    """别名表里姓名含 QUESTION_WORDS 单字的人名 (简体), 长的在前. 别名表不存在时为空."""
    from rag.alias import load_aliases  # 延迟导入: 只在查询时读别名表
    chars = {w for w in QUESTION_WORDS if len(w) == 1}
    names = {to_simplified(p["name"]) for p in load_aliases().get("people", [])}
    return tuple(sorted((n for n in names if len(n) >= 2 and chars & set(n)), key=len, reverse=True))


def bm25_retriever(chunks: list[ChunkDict]) -> Search:
    """对每个 chunk 的 search 字段繁转简后建索引, 返回检索函数.

    返回 search(query, k, where=None) -> list[chunk dict]: 按分数从高到低, 同分按 chunk 原顺序, 命中不足 k 个时有几个返回几个;
    给了 where 时先去掉 where 为 False 的块再排名 (rag/schema.Search). 查询依次经过繁转简, strip_question_words, tokenize.
    """
    index = build_index([tokenize(to_simplified(c["search"])) for c in chunks])

    def search(query: str, k: int, where: Where | None = None) -> list[ChunkDict]:
        scores = score_all(index, tokenize(strip_question_words(to_simplified(query))))
        if where:
            scores = {i: s for i, s in scores.items() if where(chunks[i])}
        top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [chunks[i] for i, _ in top]
    return search
