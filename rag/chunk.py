"""切块: sanguozhi.jsonl 的段落和裴注 -> chunks.jsonl 的检索块.

    uv run python -m rag.chunk [--min-len 50] [--max-len 400]
    uv run python -m tests.test_chunk
    uv run python -m eval.length_stats data/processed/chunks.jsonl   # 看切出来的长度分布

输入是嵌套的: 正文段落里挂着裴注 (rag/schema.Paragraph). 输出是平的, 按原书阅读顺序排列:

    paragraphs: list[Paragraph]                         2240 个正文段落
    └─ {"id": "01-0000", "text": "太祖武皇帝，沛國譙人也……", "notes": [...]}
       ├─ notes[0] = {"offset": 29, "text": "《曹瞞傳》曰：……"}     源 id "01-0000n0"
       └─ notes[1] = {"offset": 48, "text": "司馬彪《續漢書》曰：……"}  源 id "01-0000n1"

    chunks: list[Chunk]                                 4365 块, 写盘时每行是 asdict(Chunk) (rag/schema.ChunkDict)
    ├─ Chunk(id="01-0000",     tier=1, sources=["01-0000"])                                正文块
    ├─ Chunk(id="01-0000n0",   tier=2, sources=["01-0000n0"], parent="01-0000", offset=29)  裴注块
    ├─ Chunk(id="01-0000n1.0", tier=2, sources=["01-0000n1"], parent="01-0000", offset=48)  长注切成两片
    ├─ Chunk(id="01-0000n1.1", ...)
    └─ ...

两个词: 小节 = juan 和 headings 都相同的连续一串段落 (大致是一个人的传); 单元 (Unit) = 一个正文段落,
或它被 split_long 切开后的一片.

流程: to_units (长段按句切片) -> group_units (短片并进同一小节的邻居) -> make_chunks (正文块, 再加裴注块)
-> interleave (排成阅读顺序: 正文块后面紧跟它的裴注块).

设计
    - 裴注独立成块, 裴注之间也不合并, 用 parent + offset 指回所注段落. 长注很多 (最长 10209 字),
      并进正文会把块撑大, 也分不清哪句是陈寿, 哪句是裴注所引.
    - 只在句末 。！？ 处切. 全书最长一句 398 字, 所以 max_len >= 400 时不会有放不下的单句.
    - 合并不跨卷, 不跨小节 (juan + headings 相同才并), 否则会把两个人的事并进一块.
    - text 和 search 同步切: search 只比 text 多年号后的 " NNN年", 句末标点完全一致, 切出的句数相同.
    - text 里的换行 (来自多行的 quote 模板) 当普通字符.
    - max_len 只限制切片, 不限制合并后的块: 接近 max_len 的片并进短邻居后会超过
      (默认参数下 49 个正文块超过 400 字, 最长 511).
    - chunk id 随切法变, 源 id (格式见 rag/schema.py) 不变. 评测的 gold 标源 id, 通过 Chunk.sources 映射到块, 同一套题能比较不同切法.
    长度分布等数据事实见 docs/DESIGN.md "切块".
"""
import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config
from .schema import Paragraph, Variant

PARAS_PATH = config.DATA_DIR / "processed" / "sanguozhi.jsonl"
CHUNKS_PATH = config.DATA_DIR / "processed" / "chunks.jsonl"

SENTENCE_END = "。！？"
CLOSING = "」』”’）"  # 句末标点后紧跟的这些字符属于同一句, 如 `曰：「善。」`


@dataclass
class Unit:
    """切块的输入单元: 一个正文段落 (或它被 split_long 切开后的一片).

    para:      源段落 id, 如 "01-0003".
    piece:     切开后的片序号, 从 0 起; 没切开的段落为 0. 同一段的各片 piece 连续.
    n_pieces:  这段一共切成几片 (没切开为 1). 生成 chunk id 时用来判断要不要加片号.
    """
    para: str
    juan: int
    book: str
    juan_title: str
    headings: list[str]
    text: str
    search: str
    variants: list[Variant]
    piece: int = 0
    n_pieces: int = 1


@dataclass
class Chunk:
    """检索与回答的基本单位. 写进 chunks.jsonl 的每一行就是 asdict(Chunk).

    id:       chunk id: 唯一, 确定 (同输入同参数 -> 同 id), 以 sources[0] 开头. 片号只看第一个单元:

                  01-0046 (411 字) 被 split_long 切成两片:
                  ├─ Chunk(id="01-0046.0", sources=["01-0046"])              第 0 片单独一块
                  └─ Chunk(id="01-0046.1", sources=["01-0046", "01-0047"])   第 1 片后面并了短段 01-0047
                  没切开的段落不带片号: Chunk(id="01-0008", sources=["01-0008"])
                  裴注切开时同样加片号: "01-0000n1.0", "01-0000n1.1"
    work:     "三国志" (将来加辅助史料时区分).
    tier:     1 正文 (陈寿), 2 裴注.
    juan, book, juan_title, headings: 同段落记录; 裴注块取所注段落的值. 出处引用用.
    text:     展示与引文核对用的文本. 正文块由若干单元的 text 用换行符连接.
    search:   给 BM25 / 向量用的文本. 与 text 一一对应, 只多年号后的公元年 (如 "太和三年 229年").
    sources:  这个块覆盖的源 id, 按顺序去重. 正文块是段落 id 列表; 裴注块是 ["01-0003n2"].
              评测靠它把 gold 源 id 映射到 chunk.
    parent:   裴注块所注段落 id; 正文块为 None.
    offset:   裴注块挂在 parent 段落 text 的第几个字之后 (段落坐标, 不是 chunk 坐标); 正文块为 None.
    cites:    裴注块所引书名 (启发式, 见 ingest/sanguozhi.py); 正文块为 [].
    variants: 校勘记录. 记录本身不带位置, 段落或注文被切开时无法分到具体哪片,
              所以每片都带上来源段落/注文的全部记录; 合并多个段落时依次拼接.
    """
    id: str
    tier: int
    juan: int
    book: str
    juan_title: str
    headings: list[str]
    text: str
    search: str
    sources: list[str]
    parent: str | None = None
    offset: int | None = None
    cites: list[str] = field(default_factory=list)
    variants: list[Variant] = field(default_factory=list)
    work: str = "三国志"


# ---------- 切句与组块 ----------

def split_sentences(text: str) -> list[str]:
    """按句末标点 。！？ 切句, 紧跟在句末标点后的闭合引号/括号 (CLOSING) 归前一句.

    性质:
      - "".join(结果) == text, 不丢字, 不改字, 不去空白.
      - 除最后一句外, 每句都以 SENTENCE_END 中的一个字结尾, 后面可以再跟若干 CLOSING 字符.
      - 空字符串返回 []. 没有句末标点的文本整体是一句.
    例: `曰：「善。」遂行。` -> [`曰：「善。」`, `遂行。`].
    引号内部的句号也切: `曰：「善。吾從之。」` -> [`曰：「善。`, `吾從之。」`].
    """
    prev = curr = 0
    l = len(text)
    res = []
    while curr < l:
        ch = text[curr]
        if ch in SENTENCE_END:
            curr += 1
            while curr < l and text[curr] in CLOSING:
                curr += 1
            res.append(text[prev:curr])
            prev = curr
        else:
            curr += 1
    if prev < l:
        res.append(text[prev:])
    return res


def split_long(text: str, search: str, max_len: int) -> list[tuple[str, str]]:
    """把超过 max_len 的文本在句子边界切成若干片, text 与 search 同步切.

    返回 [(text 片, search 片), ...]. len(text) <= max_len 时返回 [(text, search)].
    性质:
      - "".join(text 片) == text, "".join(search 片) == search.
      - 第 k 片的 search 去掉 " NNN年" 后等于第 k 片的 text (两者用同一个句子边界切).
        search 只比 text 多年号后的公元年 (见模块 docstring), 所以分别 split_sentences 后句数相同, 可以逐句配对.
      - 只在句子边界切: 每片都由完整的句子组成, 不在句中切断.
      - 每片 text 长度 <= max_len, 除非这一片只有一句且这一句本身超过 max_len.
      - 长度按 text 算, 不按 search 算.
    打包方式是贪心装满: 一句一句往当前片里放, 放不下就另起一片.
    """
    res_text = split_sentences(text)
    res_search = split_sentences(search)
    res = []
    curr_t, curr_s = "", ""
    for t_, s_ in zip(res_text, res_search):
        if len(curr_t) + len(t_) <= max_len:
            curr_t += t_
            curr_s += s_
        else:
            if curr_t:
                res.append((curr_t, curr_s))
            curr_t, curr_s = t_, s_
    if curr_t:
        res.append((curr_t, curr_s))
    return res


def _get_key(unit: Unit) -> tuple[int, tuple[str, ...]]:
    return (unit.juan, tuple(unit.headings))

def group_units(units: list[Unit], min_len: int) -> list[list[Unit]]:
    """把短单元并进同一小节的邻居, 返回分组; 每组将来成为一个正文块.

    例 (min_len=50; "短" = text 不到 50 字; 小节 = (juan, headings) 相同的一串单元):

        units: list[Unit]                         小节 (juan, headings)   长度
        ├─ 16-0027  郑浑字文公，河南开封人也……     (16, ["鄭渾"])          48  短
        ├─ 16-0028  浑将泰小子袤避难淮南……         (16, ["鄭渾"])         159
        ├─ 16-0029  时梁兴等略吏民五千餘家……       (16, ["鄭渾"])         353
        ├─ 16-0030  太祖征汉中，以浑为京兆尹……     (16, ["鄭渾"])         323
        ├─ 16-0031  明帝闻之，下诏称述……           (16, ["鄭渾"])          45  短
        └─ 16-0032  仓慈字孝仁，淮南人也……         (16, ["倉慈"])         385

        返回 list[list[Unit]]: [[16-0027, 16-0028], [16-0029], [16-0030, 16-0031], [16-0032]]

    规则 (在每个小节里各做一遍):
      - 每遇到一个长单元, 就和它前面还没分组的短单元组成一组.  -> 开头的短单元往后并
      - 小节末尾剩下的短单元并进前一组.                       -> 末尾的短单元往前并
      - 整个小节都是短单元时, 整个小节一组.

    性质:
      - 各组依次连起来正好是原来的 units: 顺序不变, 不丢不重.
      - 一组不跨小节, 否则会把两个人的事并进一块.
      - 两个长单元不在同一组: 只为短单元合并.
      - 总长不到 min_len 的组一定是整个小节 (没有邻居可并).
    前提: 同一小节的单元在 units 里是连续的 (全书如此; 代码按小节分组, 不检查是否相邻).
    """
    res = []
    key_to_units = defaultdict(list)
    for unit in units:
        key_to_units[_get_key(unit)].append(unit)

    for units_ in key_to_units.values():
        if len(units_) == 1:
            res.append(units_)
        else:
            prev = 0
            for i, u_ in enumerate(units_):
                if len(u_.text) >= min_len:
                    res.append(units_[prev:i+1])
                    prev = i+1
            if prev < len(units_):
                if prev == 0:
                    res.append(units_)
                else:
                    res[-1].extend(units_[prev:])
    return res


def make_chunks(paragraphs: list[Paragraph], min_len: int, max_len: int) -> list[Chunk]:
    """从段落记录 (sanguozhi.jsonl 的行) 生成全部 chunk: 先全部正文块, 再全部裴注块 (阅读顺序由 interleave 排).

    正文和裴注在段落记录里本来就分开存, 各走一条路:

        Paragraph {"id": "01-0000", "text": "太祖武皇帝……", "notes": [...]}
        ├─ text/search  -> to_units 切长段 -> group_units 并短单元 -> 每组一个正文块 (tier 1)
        │                  Chunk(id="01-0000", tier=1, sources=["01-0000"])
        └─ notes[i]     -> split_long 切长注 -> 每片一个裴注块 (tier 2), 注与注不合并
                           Chunk(id="01-0000n0", tier=2, sources=["01-0000n0"], parent="01-0000", offset=29)
                           切成多片时 id 加片号: "01-0000n1.0", "01-0000n1.1"

    性质:
      - 先按原文顺序列出全部正文块, 再按原文顺序列出全部裴注块.
      - 正文块的 text 去掉换行符后依次连起来, 等于全部段落 text 去掉换行符后依次连起来; 裴注块同理.
      - id 唯一, 确定, 以 sources[0] 开头. 格式见 Chunk.
      - 裴注块: sources == [注的源 id], parent 是存在的段落 id, 0 <= offset <= len(parent 段落的 text).
      - 每个块 search 去掉 " NNN年" 后包含 text.
    """
    res = []
    units = to_units(paragraphs, max_len)
    grouped_units = group_units(units, min_len)
    for units in grouped_units:
        id = units[0].para if units[0].n_pieces == 1 else units[0].para + f".{units[0].piece}"
        text = "\n".join([u.text for u in units])
        search = "\n".join([u.search for u in units])
        sources = list(dict.fromkeys(u.para for u in units).keys())
        variants = []
        for u in units:
            variants.extend(u.variants)
        res.append(
            Chunk(
                id=id, tier=1,
                juan=units[0].juan,
                juan_title=units[0].juan_title,
                headings=units[0].headings,
                book=units[0].book,
                text=text,
                search=search,
                sources=sources,
                variants=variants,
            )
        )

    for para in paragraphs:
        for index, note in enumerate(para["notes"]):
            list_text_search = split_long(note["text"], note["search"], max_len)
            if len(list_text_search) == 1:
                res.append(
                    Chunk(
                        id = f"{para['id']}n{index}",
                        tier = 2,
                        juan = para['juan'],
                        book = para['book'],
                        juan_title = para['juan_title'],
                        headings = para['headings'],
                        text = list_text_search[0][0],
                        search = list_text_search[0][1],
                        sources = [f"{para['id']}n{index}"],
                        parent = para['id'],
                        offset = note['offset'],
                        cites = note['cites'],
                        variants = note['variants']
                    )
                )
            else:
                for i_, (t_, s_) in enumerate(list_text_search):
                    res.append(
                        Chunk(
                            id = f"{para['id']}n{index}.{i_}",
                            tier = 2,
                            juan = para['juan'],
                            book = para['book'],
                            juan_title = para['juan_title'],
                            headings = para['headings'],
                            text = t_,
                            search = s_,
                            sources = [f"{para['id']}n{index}"],
                            parent = para['id'],
                            offset = note['offset'],
                            cites = note['cites'],
                            variants = note['variants']
                        )
                    )
    return res


# ---------- 读写与 CLI ----------

def load_paragraphs(path: Path = PARAS_PATH) -> list[Paragraph]:
    """读 sanguozhi.jsonl (ingest/sanguozhi.py 生成)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_units(paragraphs: list[Paragraph], max_len: int) -> list[Unit]:
    """每个段落的正文 (text/search) 用 split_long 切成一片或多片, 转成 Unit. 切法见 split_long.

    只处理正文, 不碰段落里的 notes; 裴注由 make_chunks 另行处理 (见其 docstring).
    """
    units = []
    for p in paragraphs:
        pieces = split_long(p["text"], p["search"], max_len)
        for k, (text, search) in enumerate(pieces):
            units.append(Unit(para=p["id"], juan=p["juan"], book=p["book"], juan_title=p["juan_title"],
                              headings=p["headings"], text=text, search=search, variants=p["variants"],
                              piece=k, n_pieces=len(pieces)))
    return units


def note_id(para_id: str, i: int) -> str:
    """第 i 条裴注 (段内从 0 起) 的源 id."""
    return f"{para_id}n{i}"


def interleave(chunks: list[Chunk]) -> list[Chunk]:
    """make_chunks 的输出是 "先全部正文块, 再全部裴注块"; 写盘前改成原书的阅读顺序: 每个正文块后面紧跟
    挂在它的段落上的裴注块 (按注的原顺序). 一个段落被切成几个正文块时, 它的注都跟在最后一块后面.

    为什么: 按文件顺序读某块的前后几块来扩展上下文时, 需要正文和它的裴注相邻; 不交错的话,
    一段的注离它两千多行, 永远读不到.
    只改顺序, 块的内容和 id 不变. 例:

        输入: [01-0004, 01-0005, 01-0004n0, 01-0004n1, 01-0005n0]
        输出: [01-0004, 01-0004n0, 01-0004n1, 01-0005, 01-0005n0]
    """
    notes: dict[str | None, list[Chunk]] = {}  # 裴注块的 parent 不会是 None; 万一是, 放到最后 (见下)
    for c in chunks:
        if c.tier == 2:
            notes.setdefault(c.parent, []).append(c)
    last_of = {}  # 段落 id -> 含这个段落的最后一个正文块的位置
    body = [c for c in chunks if c.tier == 1]
    for i, c in enumerate(body):
        for s in c.sources:
            last_of[s] = i
    out = []
    for i, c in enumerate(body):
        out.append(c)
        for s in c.sources:
            if last_of[s] == i:
                out.extend(notes.pop(s, []))
    for rest in notes.values():  # 找不到正文块的注 (不应发生) 放在最后, 不丢
        out.extend(rest)
    return out


def write_chunks(chunks: list[Chunk], path: Path = CHUNKS_PATH) -> None:
    """把 chunks 写成 JSONL (每行一个 chunk 的 JSON, 中文不转义), 覆盖已有文件; 目录不存在时先创建."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="切块: sanguozhi.jsonl -> chunks.jsonl")
    ap.add_argument("--min-len", type=int, default=50)
    ap.add_argument("--max-len", type=int, default=400)
    args = ap.parse_args()
    chunks = interleave(make_chunks(load_paragraphs(), args.min_len, args.max_len))
    write_chunks(chunks)
    n1 = sum(c.tier == 1 for c in chunks)
    print(f"正文块 {n1}, 裴注块 {len(chunks) - n1} -> {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
