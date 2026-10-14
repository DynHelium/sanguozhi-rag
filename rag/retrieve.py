"""检索的组装: 读切块 (load_chunks), 拼出默认的检索 (make_search). 检索评测和问答共用这一处.

    chunks = load_chunks()                                 # 原文切块, 4365 块 (rag/schema.ChunkDict)
    search = make_search(chunks, alias="route2")
    search("曹操因为什么逃离洛阳？", 10)                     # -> 前 10 块 (ChunkDict), 从好到差

    uv run python -m rag.retrieve "曹操因为什么逃离洛阳？" --bm25   # 命令行: 打印前 5 块的出处和原文 (--bm25 不需要向量模型)

两套索引, 名字在命令行和代码里不同:
    命令行 --index   代码里的 index   切块文件                            search 字段
    orig             ""              data/processed/chunks.jsonl         原文 (加年号的公元年)
    trans            "trans"         data/processed/chunks-trans.jsonl   全书白话译文
index 也是 dense 向量缓存的名字 (rag/dense.py). 仓库附带全书译文 (data/processed/translations/);
白话切块由 `uv run python -m ingest.trans_chunks` 生成, 生成之前只能用原文索引.

默认检索: BM25 和 dense 各取前 rag/fusion.DEPTH 名, 用 RRF 融合 (rag/fusion.py);
给了 alias 时再用别名表扩展查询 (rag/alias.py).
"""
import json

from rag import config
from rag.schema import ChunkDict, Search


INDEXES = {"": "chunks.jsonl", "trans": "chunks-trans.jsonl"}  # 索引名 (也是 dense 缓存名) -> 切块文件


def load_chunks(index: str = "") -> list[ChunkDict]:
    """读一套切块. index "" = 原文切块 chunks.jsonl (search 是原文); "trans" = 白话切块 chunks-trans.jsonl
    (同样的 id 和 text, 多一个 trans 字段, search 换成白话; 由 ingest.trans_chunks 生成, 见模块 docstring).
    两者的 text 都是原文: 展示和引文核对只认原文. 命令行上的 "orig" 在这里写 "" ("orig" 会 KeyError)."""
    with open(config.DATA_DIR / "processed" / INDEXES[index], encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_search(chunks: list[ChunkDict], index: str = "", alias: str = "") -> Search:
    """返回 search(query, k, where=None): BM25 + dense 用 RRF 融合 (rag/fusion.py). index 同 load_chunks;
    where 限定检索范围 (rag/schema.Search).

    alias (查询扩展用 rag/alias.expand_query):
        ""        不扩展
        "bm25"    BM25 用扩展后的查询, dense 用原查询
        "both"    两路都用扩展后的查询
        "route"   原查询照常走两路, 扩展后的查询另加一路 BM25, 三路一起融合
        "route2"  同 route, 再加一路 dense (四路)
    白话索引的向量缓存需要先用 `uv run python -m rag.dense --trans` 建好 (截断长度见 rag/dense.MAX_SEQ).
    缓存缺失或和切块对不上时直接退出, 不在这里自动重建: 编码全书白话切块要几分钟, 不该在查询时悄悄发生."""
    from rag import dense, fusion
    from rag.alias import expand_query
    if index and not dense.cache_ok(chunks, name=index):
        raise SystemExit(f"{index} 索引的向量缓存缺失或过期, 先跑: uv run python -m rag.dense --{index}")
    return fusion.hybrid_retriever(chunks, name=index, expand=expand_query if alias else None,
                                   expand_dense=alias in ("both", "route2"), extra_route=alias.startswith("route"))


def main() -> None:
    """命令行查一个问题, 打印前 k 块: 编号, 正文还是裴注, 卷名和小标题, chunk id, 原文 (过长截断)."""
    import argparse
    ap = argparse.ArgumentParser(description="检索一个问题, 看找到的原文和出处")
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=5, help="显示前几块")
    ap.add_argument("--bm25", action="store_true", help="只用 BM25 (不需要向量模型和 index extra)")
    ap.add_argument("--alias", default="route2", help='别名扩展方式, 见 make_search; "" = 不扩展 (--bm25 时不用)')
    args = ap.parse_args()
    chunks = load_chunks()
    if args.bm25:
        from rag.bm25 import bm25_retriever
        search = bm25_retriever(chunks)
    else:
        search = make_search(chunks, alias=args.alias)
    for i, c in enumerate(search(args.question, args.k), start=1):
        where = c["juan_title"] + "".join("·" + h for h in c.get("headings") or [])
        text = c["text"] if len(c["text"]) <= 120 else c["text"][:120] + "……"
        print(f"[{i}] {where} {'正文' if c['tier'] == 1 else '裴注'} ({c['id']})\n    {text}\n")


if __name__ == "__main__":
    main()
