"""用现成库 bm25s 交叉核对 rag/bm25.py 在真实 chunks 上的打分. bm25s 不进依赖, 临时装:

    uv run --with bm25s python -m tests.xcheck_bm25

对照 bm25s 的 method="lucene" (idf 公式相同). 它省掉了分子里的 (k1 + 1) (Lucene 8 以后的写法), 所有分数差常数倍 2.2,
排序不变, 这里先除掉这个常数再比. 分词和词表都用 rag/bm25.py 自己的, 所以只核对 build_index / idf / score_all;
tokenize 由 tests.test_bm25 检查.
"""
import json

import bm25s

from rag import bm25, config
from rag.normalize import to_simplified

QUERIES = ["孙策是被哪个吴郡太守的门客杀的？", "曹丕的庙号是什么？", "毌丘俭字仲恭", "官渡之战", "荀彧怎么死的？"]


def main() -> None:
    with open(config.DATA_DIR / "processed" / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    docs = [bm25.tokenize(to_simplified(c["search"])) for c in chunks]
    index = bm25.build_index(docs)
    vocab = {t: i for i, t in enumerate(index["postings"])}
    lib = bm25s.BM25(method="lucene", k1=bm25.K1, b=bm25.B)
    lib.index(bm25s.tokenization.Tokenized(ids=[[vocab[t] for t in d] for d in docs], vocab=vocab), show_progress=False)

    worst = 0.0
    for q in QUERIES:
        qt = [t for t in bm25.tokenize(to_simplified(q)) if t in vocab]
        mine = bm25.score_all(index, qt)
        ref = lib.get_scores([vocab[t] for t in qt])
        diff = max(abs(ref[i] - mine.get(i, 0.0) / (bm25.K1 + 1)) for i in range(len(docs)))
        worst = max(worst, diff)
        print(f"{diff:.2e}  {q}")
    print("一致" if worst < 1e-4 else "不一致: 检查 build_index / idf / score_all", f"(最大差 {worst:.2e})")


if __name__ == "__main__":
    main()
