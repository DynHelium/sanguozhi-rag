"""数据文件的记录格式, 多个模块共用. 都是 TypedDict: json.loads 读进来仍是普通 dict, 这里只给类型检查和阅读用.

    sanguozhi.jsonl (ingest/sanguozhi.py 写)    每行一个 Paragraph
    chunks.jsonl    (rag/chunk.py 写)           每行一个 ChunkDict
    chunks-trans.jsonl                          同上, 多一个 trans 字段
    eval/retrieval.jsonl                        每行一个 Question

只在一个模块里用的格式 (别名表, 年号表, BM25 索引) 定义在各自的模块里.

三种 id:
    "01-0046"     段落 id = f"{卷号:02d}-{卷内段落序号:04d}", 序号从 0 起, 卷内连续
    "01-0000n1"   裴注 id = 所注段落 id + "n" + 段内第几条注 (从 0 起)
    "01-0046.1"   chunk id (rag/chunk.py 生成) = 源 id, 被切开时加 ".片号"; 换切法就会变
前两种统称 "源 id": 由原文版本 (raw/manifest.json 的 revid) 和 patches.yaml 决定, 不随切块变.
评测的 gold, 别名表的出处 (src), chunk 的 sources 都写源 id, 不写 chunk id.
"""
from collections.abc import Callable
from typing import Literal, NotRequired, Protocol, TypeAlias, TypedDict


class Variant(TypedDict):
    """一条校勘记录: 这里采用了哪个读法, 没采用的是什么. 记录不带位置, 只挂在所在的段落或注文上.

        {"kind": "改", "adopted": "以", "other": "爾"}     `（爾）〔以〕`  点校本改字
        {"kind": "删", "adopted": "",   "other": "祀"}     `（祀）`        点校本删衍字
        {"kind": "补", "adopted": "歸降", "other": ""}     `〔歸降〕`      点校本补字 (半角 `[六]` 也算)
        {"kind": "校", "adopted": "鄱陽", "other": "潘陽"}  `{{校|潘陽|鄱陽}}`  维基编辑校正讹字
        {"kind": "另", "adopted": "子卿遠", "other": "子遠，卿"}  `{{另|底本|另本}}`  异文, 采用底本
    """
    kind: Literal["改", "删", "补", "校", "另"]
    adopted: str  # 文本里用的写法
    other: str    # 没采用的写法


class Note(TypedDict):
    """一条裴注, 挂在所注段落的 notes 里."""
    offset: int               # 挂在所注段落 text 的第几个字之后, 标点和换行也算字: 注挂在 text[:offset] 的末尾.
                              # 例: 01-0000 第 0 条注 offset=29, text[:29] = "太祖武皇帝，……字孟德，漢相國參之後。"
    text: str                 # 同 Paragraph.text
    search: str               # 同 Paragraph.search
    variants: list[Variant]
    cites: list[str]          # 所引书名, 如 ["曹瞞傳"]; 裴松之自己的按语记为 "裴松之". 启发式抽取, 只作展示和过滤


class Paragraph(TypedDict):
    """sanguozhi.jsonl 的一行: 一个正文段落, 裴注嵌在 notes 里.

        {"id": "01-0000", "juan": 1, "book": "魏書", "juan_title": "魏書•武帝紀", "headings": [],
         "text": "太祖武皇帝，沛國譙人也……", "search": "太祖武皇帝，沛國譙人也……", "variants": [],
         "notes": [{"offset": 29, "text": "《曹瞞傳》曰：太祖一名吉……", "cites": ["曹瞞傳", "魏書"], ...},
                   {"offset": 48, "text": "司馬彪《續漢書》曰：……", "cites": ["續漢書"], ...}]}
    """
    id: str                   # 段落 id, 如 "01-0000" (格式见模块 docstring)
    juan: int                 # 1-30 魏書, 31-45 蜀書, 46-65 吳書
    book: str                 # "魏書" / "蜀書" / "吳書"
    juan_title: str           # 页眉的卷名, 各卷写法不统一, 如 "魏書•武帝紀"
    headings: list[str]       # 所在的各级标题, 从 1 级到最深一级, 如 ["武帝諸子", "任城威王彰"]; 卷首为 []
    text: str                 # 展示文本: 改正后的读法, 年号不带公元年. 引文逐字核对用它
    search: str               # 检索文本: 只比 text 多年号后的 " NNN年", 如 "太和三年 229年"
    variants: list[Variant]
    notes: list[Note]


class ChunkDict(TypedDict):
    """chunks.jsonl 的一行, 即 asdict(rag.chunk.Chunk). 字段说明见 rag/chunk.Chunk."""
    id: str
    tier: Literal[1, 2]       # 1 正文, 2 裴注
    juan: int
    book: str
    juan_title: str
    headings: list[str]
    text: str
    search: str
    sources: list[str]        # 覆盖的源 id (段落 id 或裴注 id "01-0003n2"), 评测的 gold 标的是源 id
    parent: str | None        # 裴注块所注的段落 id; 正文块为 None
    offset: int | None        # 裴注块挂在 parent 段落 text 的第几个字之后; 正文块为 None
    cites: list[str]
    variants: list[Variant]
    work: str                 # "三国志"
    trans: NotRequired[str]   # 只有 chunks-trans.jsonl 有: 白话译文, 这时 search 也换成白话


class Question(TypedDict):
    """eval/retrieval.jsonl 的一道检索题.

        {"id": "r-02", "type": "named", "q": "刘备是如何投奔刘表的？", "gold": ["32-0008", "38-0008"],
         "equiv": {"38-0008": ["38-0006"]}}
    """
    id: str
    type: Literal["named", "alias", "variant", "event", "aggregate", "none"]
    q: str
    gold: list[str]                               # 答案所在的源 id; type "none" (资料里没有答案) 为 []
    equiv: NotRequired[dict[str, list[str]]]     # gold 源 id -> 同样算命中的其他源 id (同一件事记在别处)
    note: NotRequired[str]                        # 出题备注


Where: TypeAlias = Callable[[ChunkDict], bool]
"""检索范围: 对要保留的块返回 True, 如 lambda c: c["book"] == "吳書" and c["tier"] == 2 (吴书的裴注)."""


class Search(Protocol):
    """检索函数: search(query, k, where=None) -> 前 k 个块, 从好到差. bm25 / dense / hybrid 的 *_retriever(chunks) 都返回它.
    给了 where 时只在 where(块) 为 True 的块里排名: 先筛选再取前 k, 所以范围里排名靠后的块不会被范围外的块挤掉."""
    def __call__(self, query: str, k: int, where: Where | None = None) -> list[ChunkDict]: ...
