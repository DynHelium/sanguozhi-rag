"""检索评测: 跑 eval/retrieval.jsonl 的题, 报 hit@1/5/10 与 MRR, 按题型分组. 不经过 LLM.

    uv run python -m eval.run_retrieval -r bm25                    # 检索器: random (占位基线) / bm25 / dense / hybrid
    uv run python -m eval.run_retrieval -r hybrid --alias route2   # 加别名扩展 (见 rag/retrieve.make_search)
    uv run python -m eval.run_retrieval -r hybrid --strict         # 严格口径 (见下)
    uv run python -m eval.run_retrieval -r hybrid --save           # 存结果, 之后用 --compare 对比
    uv run python -m eval.run_retrieval -r hybrid --compare data/processed/eval_runs/hybrid.json

dense / hybrid 需要 `uv sync --extra index` 并先跑 `uv run python -m rag.dense` 建向量缓存.
--index 默认 orig (原文切块); --index trans 用白话切块, 需要先用 ingest.trans_chunks 生成 chunks-trans.jsonl (见 rag/retrieve.py).

题目文件 eval/retrieval.jsonl, 每行一题 (人工出题和标注, rag/schema.Question):

    {"id": "r-02", "type": "named", "q": "刘备是如何投奔刘表的？", "gold": ["32-0008", "38-0008"],
     "equiv": {"38-0008": ["38-0006"]}}

    q       用户会怎么问就怎么写 (简体白话), 不照抄原文
    gold    答案所在的源 id (段落 id 或裴注 id, 用 `uv run python -m eval.find 词` 查), 不是 chunk id
    equiv   可选: gold 源 id -> 同样能回答这题的其他源 id (同一件事记在别处), 命中也算
    note    可选: 出题备注
    type    题型, 报表按它分组:
              named     带专名 (人名/地名/年号), BM25 应该擅长
              event     只问事件, 不带或少带专名 (如 "谁投降了"), 测白话对文言的语义匹配
              alias     用别名提问 (如 "魏武"), 测查询侧的别名扩展
              variant   非标准写法 (异体字, 常见误写), 如 "毋丘俭" (原文 "毌丘")
              aggregate 答案分散在多处 (如 "有几个马忠" "吴国有几个皇帝")
              none      资料里没有答案 (如 "曹丕四友"), gold 写 []. 不计入 hit@k / MRR, 只列出前 3 名

出题约定
    - gold 填 "读了它就能回答" 的最小源 id 集合; 答案分散在几处时都填上, 命中任意一个就算 (见 eval/metrics.py).
    - 答案只在裴注里时, gold 只填裴注 id: 这类题正好检验裴注块能不能被找到.
    - 同一类题至少 5 道, 否则分组的数字没有意义.

流程: 读切块 -> 检查每个 gold 都是存在的源 id -> 每题取前 10 -> 用 eval/metrics.py 算 hit@1/5/10 与 reciprocal rank
-> 按题型汇总, 列出前 10 都没命中的题. --save 把每题结果写到 data/processed/eval_runs/<结果名>.json,
改动检索后用 --compare 看前后对比.

检索器接口: RETRIEVERS[name](chunks, index) 返回 rag/schema.Search, 即 search(query, k, where=None) -> 前 k 块, 从好到差; 评测不传 where (搜全书).
"""
import argparse
import json
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypedDict

from eval import metrics
from rag import bm25, config, dense, fusion, retrieve
from rag.schema import ChunkDict, Question, Search, Where

QUESTIONS_PATH = Path(__file__).parent / "retrieval.jsonl"
CHUNKS_PATH = config.DATA_DIR / "processed" / "chunks.jsonl"
RUNS_DIR = config.DATA_DIR / "processed" / "eval_runs"
KS = (1, 5, 10)  # 改 KS 要同步改 Metric, Row, Summary 和 evaluate, summarize 里的键

Metric = Literal["hit@1", "hit@5", "hit@10", "rr"]
COLS: tuple[Metric, ...] = ("hit@1", "hit@5", "hit@10", "rr")

# 键名带 "@", 只能用函数式写法.
Row = TypedDict("Row", {"id": str, "type": str, "q": str, "gold": list[str], "top": list[str], "rr": float,
                        "hit@1": float, "hit@5": float, "hit@10": float})
"""evaluate 的每题结果. top 是前 5 块的 id, rr 是 reciprocal rank, hit@k 是 0.0 或 1.0."""

Summary = TypedDict("Summary", {"n": int, "hit@1": float, "hit@5": float, "hit@10": float, "rr": float})
"""summarize 的一组 (全部或一个题型) 平均值; n 是题数, rr 的平均就是 MRR."""


def random_retriever(chunks: list[ChunkDict]) -> Search:
    """占位基线: 每题按题目文本做种子随机抽 k 个 (给了 where 时在范围内抽, 范围不够 k 个就全给). 任何真检索器都应该远好于它."""
    def search(query: str, k: int, where: Where | None = None) -> list[ChunkDict]:
        pool = [c for c in chunks if where(c)] if where else chunks
        return random.Random(query).sample(pool, min(k, len(pool)))
    return search


RETRIEVERS: dict[str, Callable[[list[ChunkDict], str], Search]] = {
    "random": lambda chunks, index: random_retriever(chunks),
    "bm25": lambda chunks, index: bm25.bm25_retriever(chunks),
    "dense": lambda chunks, index: dense.dense_retriever(chunks, name=index),
    "hybrid": lambda chunks, index: fusion.hybrid_retriever(chunks, name=index)}


def strict_is_hit(chunk: ChunkDict, gold: list[str]) -> bool:
    """--strict 时代替 metrics.is_hit: 只认规则 (a) (sources 在 gold 里), 不认 parent.
    gold 是正文段落时, 只找到挂在它上面的裴注块不算: 裴注说的不一定是题目问的那件事."""
    return any(s in gold for s in chunk["sources"])


def gold_ids(q: Question) -> list[str]:
    """gold 加上它们的等价源 id (可选字段 equiv). 检索评测是 "命中任意一个", 所以直接并起来."""
    return list(dict.fromkeys(q["gold"] + [e for g in q["gold"] for e in q.get("equiv", {}).get(g, [])]))


def load_jsonl(path: Path) -> list[Any]:
    """读 JSONL, 跳过空行. 每行是什么由调用方决定 (检索题是 rag/schema.Question)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_gold(questions: list[Question], chunks: list[ChunkDict]) -> list[str]:
    """返回错误列表: gold 为空 (none 题除外), none 题却填了 gold, gold 不是存在的源 id
    (常见原因: 填了 chunk id 如 "01-0046.0"), 题目 id 重复."""
    known = {s for c in chunks for s in c["sources"]}
    errors: list[str] = []
    seen: set[str] = set()
    for q in questions:
        if q["id"] in seen:
            errors.append(f"{q['id']}: 题目 id 重复")
        seen.add(q["id"])
        if q.get("type") == "none":
            if q.get("gold"):
                errors.append(f"{q['id']}: none 题的 gold 应为 []")
        elif not q.get("gold"):
            errors.append(f"{q['id']}: 没有 gold (资料里没有答案的题, type 写 none)")
        for g in q.get("gold", []):
            if g not in known:
                errors.append(f"{q['id']}: gold {g!r} 不是存在的源 id")
    return errors


def evaluate(questions: list[Question], search: Search, k_max: int) -> list[Row]:
    """每题取前 k_max 块, 算一行结果: {id, type, q, gold, top (前 5 块的 id), rr, hit@1, hit@5, hit@10}."""
    rows: list[Row] = []
    for q in questions:
        ranked, gold = search(q["q"], k_max), gold_ids(q)
        rows.append({"id": q["id"], "type": q.get("type", "?"), "q": q["q"], "gold": q["gold"],
                     "top": [c["id"] for c in ranked[:5]], "rr": metrics.reciprocal_rank(ranked, gold),
                     "hit@1": metrics.hit_at_k(ranked, gold, 1), "hit@5": metrics.hit_at_k(ranked, gold, 5),
                     "hit@10": metrics.hit_at_k(ranked, gold, 10)})
    return rows


def summarize(rows: list[Row]) -> dict[str, Summary]:
    """按题型求平均: {"全部" 或题型: {"n": 题数, "hit@1": ..., "hit@5": ..., "hit@10": ..., "rr": MRR}}."""
    groups: dict[str, list[Row]] = {"全部": rows}
    for r in rows:
        groups.setdefault(r["type"], []).append(r)

    def mean(rs: list[Row], c: Metric) -> float:
        return sum(r[c] for r in rs) / len(rs)
    return {name: {"n": len(rs), "hit@1": mean(rs, "hit@1"), "hit@5": mean(rs, "hit@5"), "hit@10": mean(rs, "hit@10"),
                   "rr": mean(rs, "rr")} for name, rs in groups.items()}


def print_summary(summary: dict[str, Summary], before: dict[str, Summary] | None) -> None:
    """打印汇总表; 给了 before (之前 --save 的 summary) 时每格后面附上变化量."""
    cols = COLS
    print(f"\n{'题型':<10}{'题数':>5}" + "".join(f"{('MRR' if c == 'rr' else c):>16}" for c in cols))
    for name, s in summary.items():
        cells = []
        for c in cols:
            cell = f"{s[c]:.2f}"
            if before and name in before:
                d = s[c] - before[name][c]
                cell += f" ({d:+.2f})"
            cells.append(f"{cell:>16}")
        print(f"{name:<10}{s['n']:>5}" + "".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser(description="检索评测: hit@k 与 MRR")
    ap.add_argument("-r", "--retriever", default="random", choices=sorted(RETRIEVERS))
    ap.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    ap.add_argument("--index", default="orig", choices=["orig", "trans"],
                    help="检索哪套切块: orig = 原文 chunks.jsonl, trans = 白话 chunks-trans.jsonl (按白话检索)")
    ap.add_argument("--strict", action="store_true", help="严格口径: 挂在 gold 正文上的裴注块不算命中")
    ap.add_argument("--alias", default="", choices=["", "bm25", "both", "route", "route2"],
                    help="hybrid 用别名扩展查询 (rag/alias.expand_query): bm25 = 只扩 BM25, both = 两路都扩, "
                         "route / route2 = 扩展查询另加一路 (BM25 / BM25 + dense) 一起融合, 见 rag/retrieve.make_search")
    ap.add_argument("--save", action="store_true", help=f"结果存到 {RUNS_DIR}/<结果名>.json (结果名由检索器, 索引, 别名扩展, strict 拼成, 如 hybrid-trans-aliasroute2)")
    ap.add_argument("--compare", type=Path, help="与之前 --save 的结果对比")
    args = ap.parse_args()

    if not args.questions.exists():
        sys.exit(f"还没有题目文件 {args.questions}, 格式见本文件 docstring")
    questions: list[Question] = load_jsonl(args.questions)
    index = "" if args.index == "orig" else args.index
    chunks = retrieve.load_chunks(index)
    if args.strict:
        metrics.is_hit = strict_is_hit  # hit_at_k / reciprocal_rank 通过模块全局名调用 is_hit
    errors = check_gold(questions, chunks)
    if errors:
        sys.exit("题目文件有问题:\n  " + "\n  ".join(errors))

    run_name = (args.retriever + (f"-{index}" if index else "") + (f"-alias{args.alias}" if args.alias else "")
                + ("-strict" if args.strict else ""))
    if index and args.retriever in ("dense", "hybrid") and not dense.cache_ok(chunks, name=index):
        sys.exit(f"{index} 索引的向量缓存缺失或过期, 先跑: uv run python -m rag.dense --{index}")
    search = (retrieve.make_search(chunks, index, args.alias) if args.alias and args.retriever == "hybrid"
              else RETRIEVERS[args.retriever](chunks, index))
    nones = [q for q in questions if q.get("type") == "none"]
    rows = evaluate([q for q in questions if q.get("type") != "none"], search, max(KS))
    summary = summarize(rows)
    before = json.loads(args.compare.read_text(encoding="utf-8"))["summary"] if args.compare else None
    print(f"检索器 {run_name}, {len(rows)} 题 (另有 none 题 {len(nones)} 道), {len(chunks)} 个 chunk")
    print_summary(summary, before)

    misses = [r for r in rows if not r["hit@10"]]
    if misses:
        print(f"\n前 {max(KS)} 名都没命中的 {len(misses)} 题:")
        for r in misses:
            print(f"  {r['id']:<8} [{r['type']}] {r['q']}\n           gold {r['gold']}  实际前 5: {r['top']}")
    if nones:
        print("\nnone 题 (资料里没有答案) 的前 3 名 (看分数分布):")
        for q in nones:
            print(f"  {q['id']:<8} {q['q']}  ->  {[c['id'] for c in search(q['q'], 3)]}")
    if args.save:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out = RUNS_DIR / f"{run_name}.json"
        out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已保存 {out}")


if __name__ == "__main__":
    main()
