"""检索评测指标: 命中判断, hit@k, reciprocal rank.

    uv run python -m tests.test_metrics

三个函数:
    1. is_hit           单个 chunk 是否命中 gold (规则见下面 "命中规则")
    2. hit_at_k         前 k 个结果里有没有命中
    3. reciprocal_rank  第一个命中排第几
三个函数都只用到 chunk 的 "sources" 和 "parent" 两个字段, 其他字段不看.

输入长什么样 (eval/run_retrieval.py 每题调用一次; chunk 是 chunks.jsonl 读出的一行, 见 rag/schema.ChunkDict):

    ranked = [                                   list, 检索结果, 排第 1 的在 ranked[0]
        {"id": "01-0046.0",                      dict, 一个 chunk (从 data/processed/chunks.jsonl 读出的一行)
         "sources": ["01-0046"],                 <- 这块文字来自哪些段落/裴注
         "parent": None,                         <- 正文块是 None
         "text": "二月，紹遣郭圖…", ...},
        {"id": "01-0009",
         "sources": ["01-0009", "01-0010", ..., "01-0014"],   <- 几个短段落合成的一块, sources 有多个
         "parent": None, ...},
        {"id": "01-0000n1.0",
         "sources": ["01-0000n1"],               <- 裴注块: sources 是裴注自己的 id
         "parent": "01-0000", ...},              <- 它注的是哪个正文段落
    ]

    gold = ["01-0046"]                           list[str], 这题的标准答案, 人工标注 (eval/retrieval.jsonl)

两种 id, 只有 "sources", "parent", gold 里的才参与比较:
    "01-0046"     正文段落 id = 卷号-段落序号 (定义见 rag/schema.py). 不随切块变化.
    "01-0000n1"   裴注 id = 所注段落 id + "n" + 第几条注 (从 0 数). 不随切块变化.
    "01-0046.0"   chunk 的 "id", 切块时才生成 (长段落切开后加 ".0" ".1"). 命中判断不用它: gold 写的是源 id.

命中规则, 一个 chunk 满足任意一条就算命中:
    (a) 它的 sources 里至少有一个在 gold 里.
    (b) 它的 parent 在 gold 里. 即 gold 是正文段落时, 挂在这段上的裴注块也算找到了.
    反过来不算: gold 是裴注 "02-0000n0" 时, 检索到正文块 "02-0000" 不算命中, 因为注里的内容正文里没有.
    (a)(b) 两条自然满足这一点, 没有另写判断.

    例子                                                  gold             命中?
    {"sources": ["01-0046"],              "parent": None}       ["01-0046"]      是 (a)
    {"sources": ["01-0009", ..., "01-0014"], "parent": None}    ["01-0010"]      是 (a)
    {"sources": ["02-0000n0"],            "parent": "02-0000"}  ["02-0000"]      是 (b)
    {"sources": ["02-0000"],              "parent": None}       ["02-0000n0"]    否
    {"sources": ["01-0001"]}  (没有 "parent" 键)                ["01-0002"]      否 (不报错)
gold 有多个 id 时, 命中其中任意一个就算 (所以只判断 "在不在 gold 里").
"""
from rag.schema import ChunkDict



def is_hit(chunk: ChunkDict, gold: list[str]) -> bool:
    """单个 chunk 是否命中 gold, 规则 (a)(b) 和例子见模块 docstring.

    没有 "parent" 键的 chunk 当作 parent 为 None, 不抛 KeyError.
    """
    return any(s in gold for s in chunk["sources"]) or chunk.get('parent') in gold


def hit_at_k(ranked: list[ChunkDict], gold: list[str], k: int) -> float:
    """前 k 个结果里有没有命中: 有返回 1.0, 没有返回 0.0.

    例 (ranked 第 3 个才命中):  k=1 -> 0.0;  k=3 -> 1.0;  k=100 -> 1.0
    只看 ranked[:k] (不足 k 个时看全部, 为空返回 0.0), 每个 chunk 用 is_hit 判断. 返回 float 而不是 bool,
    这样所有题目取平均就是 hit@k 的比例 (如 0.6 = 60% 的题前 k 名里有答案).
    """
    for r in ranked[:k]:
        if is_hit(r, gold):
            return 1.0
    return 0.0


def reciprocal_rank(ranked: list[ChunkDict], gold: list[str]) -> float:
    """第一个命中排在第 r 位 (从 1 数, 不是从 0) 时返回 1/r; 一个都没命中返回 0.0.

    例: 第 1 个就命中 -> 1.0;  第 3 个才命中 -> 1/3;  全不中或 ranked 为空 -> 0.0.
    每个 chunk 用 is_hit 判断, 只看第一个命中, 后面再命中不加分. 所有题目取平均就是 MRR (mean reciprocal rank).
    hit@k 只回答 "在不在前 k", MRR 还反映 "排得多靠前", 两个一起看.
    """
    for i, r in enumerate(ranked):
        if is_hit(r, gold):
            return 1.0 / (i + 1)
    return 0.0
