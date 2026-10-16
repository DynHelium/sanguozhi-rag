"""Reciprocal Rank Fusion of several ranked lists (BM25, dense, alias-expanded queries): score = sum of 1 / (RRF_K + rank).
Only ranks are used, so BM25 scores (unbounded) and cosine similarities (0.3-0.8) never need to be put on one scale;
RRF_K = 60 follows Cormack et al. (2009), and each list contributes its top DEPTH results.

多路检索融合: 用 RRF 合并 BM25 和 dense 的排名. BM25 擅长人名地名这类精确的词, dense 擅长意思相近的说法, 两者互补.

    uv run python -m eval.run_retrieval -r hybrid --save --compare data/processed/eval_runs/dense.json

RRF (Reciprocal Rank Fusion, 倒数排名融合): 只看每一路给的名次, 不看分数.

    score(chunk) = sum over 每一路 r:  1 / (RRF_K + rank_r(chunk))       rank 从 1 开始; 某一路没排到它, 这一路贡献 0

为什么不直接把分数加起来: BM25 分数没有上限 (常见 5 ~ 30), 余弦相似度在 0.3 ~ 0.8 之间, 量纲不同,
直接相加等于只看 BM25. RRF 只用名次, 不用调权重, 也不用归一化分数.
RRF_K = 60 来自原论文 (Cormack et al. 2009), 是常用默认值: 越大, 第 1 名和第 10 名的差距越小.

例 (RRF_K = 60):

    bm25  排名: ["A", "B", "C"]
    dense 排名: ["C", "A", "D"]

    A: 1/61 + 1/62 = 0.03252      两路都靠前 -> 第一
    C: 1/63 + 1/61 = 0.03227
    B: 1/62        = 0.01613      只有一路排到
    D: 1/63        = 0.01587
    -> [("A", 0.03252), ("C", 0.03227), ("B", 0.01613), ("D", 0.01587)]

每一路取前 DEPTH 名参与融合. 只取前 10 的话, 一路排第 11 的块就完全拿不到分.
"""
from rag import bm25, dense
from collections import defaultdict
from collections.abc import Callable

from rag.schema import ChunkDict, Search, Where

RRF_K = 60
DEPTH = 100


def rrf(rankings: list[list[str]], rrf_k: int = RRF_K) -> list[tuple[str, float]]:
    """rankings[r] 是第 r 路的 chunk id 列表, 从好到差. 返回 [(chunk id, 融合分), ...], 从高到低.
    同分按第一次出现的先后排 (先看第 0 路), 保证结果确定: 分数按第一次出现的顺序累加, 排序是稳定的."""
    chunk_to_score   = defaultdict[str, float](float)
    for ranking in rankings:
        for i, chunk in enumerate(ranking):
            chunk_to_score[chunk] += 1 / (1 + i + rrf_k)
    return sorted(chunk_to_score.items(), key = lambda x: x[1], reverse=True)


def hybrid_retriever(chunks: list[ChunkDict], name: str = "", expand: Callable[[str], str] | None = None,
                     expand_dense: bool = True, extra_route: bool = False) -> Search:
    """BM25 和 dense 各取前 DEPTH 名, 用 RRF 融合. 返回 search(query, k, where=None) -> list[chunk dict];
    where 传给每一路, 每一路在范围内取前 DEPTH 名再融合 (rag/schema.Search).

    name:         dense 缓存名 (见 rag/dense.cache_paths): 原文 "", 白话 "trans".
    expand:       查询扩展函数 (如 rag.alias.expand_query). 给了就用扩展后的查询检索 BM25;
                  expand_dense=True 时 dense 也用扩展后的查询.
    extra_route:  True 时不替换原查询: 原查询照常走 BM25 + dense, 扩展后的查询另作一路 (expand_dense 时两路) 一起融合.
                  直接替换会丢掉原查询的检索结果 (实测原来排第一的块常被挤下去); 另加一路让原查询的结果照样参与融合, 最终排名仍可能变化.
    扩展后的查询和原查询相同 (查询里没认出人) 时不另加路."""
    searches: list[Search] = [bm25.bm25_retriever(chunks), dense.dense_retriever(chunks, name=name)]
    by_id = {c["id"]: c for c in chunks}

    def search(query: str, k: int, where: Where | None = None) -> list[ChunkDict]:
        q2 = expand(query) if expand else query
        if extra_route:
            plan = [(searches[0], query), (searches[1], query)]
            if q2 != query:
                plan += [(searches[0], q2)] + ([(searches[1], q2)] if expand_dense else [])
        else:
            plan = [(searches[0], q2), (searches[1], q2 if expand_dense else query)]
        rankings = [[c["id"] for c in s(qq, DEPTH, where)] for s, qq in plan]
        return [by_id[cid] for cid, _ in rrf(rankings)[:k]]
    return search
