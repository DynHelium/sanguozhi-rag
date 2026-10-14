"""年号对照表 data/eras.json: 年号元年是公元哪一年. 从原文统计, 再用参考表补齐和交叉核对, 每条带证据.

    uv run python -m ingest.eras          # 生成 data/eras.json 并打印覆盖情况
    uv run python -m tests.test_eras

    era_to_ce("興平", 2) -> 195
    era_to_ce("建興", 2, regime="蜀汉") -> 224          同名年号要指定政权, 否则返回 None

两个来源
    1. 原文: 维基文库 wikitext 的 YL 模板, 编者在部分年号处标了公元年. 每处带公元年的标注推出一个元年候选:
           {{YL|興平元年|194年}}   -> 興平 元年 = 194
           {{YL|太和六年|232年}}   -> 太和 元年 = 232 - (6 - 1) = 227
           {{YL|建安三年}}         没标公元年: 只计入出现次数
    2. 参考表 data/eras_ref.json: 中文维基百科 "中国年号列表" 固定修订版 oldid=94366635
       (该页凡例称主要依据李崇智《中國歷代年號考》), 由大模型解析成结构化数据, 每条带 source 链接.
       只取与三国志相关的政权 (REGIMES).
           {"中平": [{"start": 184, "end": 189, "regime": "东汉", "emperor": "汉灵帝劉宏", "source": "https://...", ...}]}

data/eras.json 的结构 (dict, 年号 -> 一条记录, 按年号排序):

    {"興平": {"start": 194,                    元年的公元年; 不能确定时为 null
              "status": "ok",                  见下
              "uses": 9,                       全书 YL 模板里出现的次数 (带不带公元年都算)
              "books": ["吳書", "蜀書", "魏書"],
              "candidates": [                  原文推出的元年, 按支持次数从多到少, 每个带证据 (可回原文核对)
                  {"start": 194, "count": 7, "books": [...],
                   "evidence": [{"juan": 1, "book": "魏書", "yl": "興平元年|194年"}, ...]}],
              "ref": [参考表里相关政权的条目],
              "by_regime": {"东汉": 194},
              "verified": true},              只有 ok 才有: 参考表里有同一个元年
     "建興": {"start": null, "status": "多政权", "by_regime": {"蜀汉": 223, "孙吴": 252, "西晋": 313}, ...}, ...}

status
    "ok"       原文推出唯一元年, 且不和参考表矛盾
    "ref"      原文没标公元年, 参考表里相关政权只有一个 -> 取参考表的元年 (如 建安)
    "多政权"    同名年号属于不同政权 (如 建興), start 为 null, 按政权查 by_regime
    "mismatch" 原文推出的元年和参考表对不上, 不采用, 待核
    "待核"      两边都没有

设计
    - 年号名只取 2 个字, 排除 NOT_ERAS 里的误标 (如 `{{YL|皓以四年}}`).
    - 原文推出多个元年时不替它裁决: 可能是同名年号属于不同政权, 也可能是编者标错. 和参考表对得上才算 多政权.
    - 原文标注是维基编者手标的, 参考表是解析出来的, 两者都可能有错, 所以每条都留着证据和出处.
"""
import json
import re
from collections import Counter, defaultdict
from typing import Literal, NotRequired, TypedDict

from rag import config

RAW_WIKI = config.DATA_DIR / "sources" / "sanguozhi" / "raw"
NOT_ERAS = {"皓以", "客來"}  # YL 模板误标 (如 "{{YL|皓以四年}}"), 不是年号

OUT = config.DATA_DIR / "eras.json"
REF = config.DATA_DIR / "eras_ref.json"
REGIMES = {"西汉", "东汉", "曹魏", "蜀汉", "孫吳", "孙吴", "西晋", "东晋"}
_YL = re.compile(r"\{\{YL\|([^|}]+)(?:\|([^|}]*))?\}\}")
_ERA_YEAR = re.compile(r"^(.{2})(元|[一二三四五六七八九十]+)年$")
_DIGITS = dict(zip("一二三四五六七八九", range(1, 10)))


class Evidence(TypedDict):
    """一处带公元年的 YL 标注, 可以回原文核对."""
    juan: int
    book: str
    yl: str     # 模板的两个参数, 如 "興平元年|194年"


class Candidate(TypedDict):
    """原文推出的一个元年候选."""
    start: int
    count: int              # 支持这个元年的标注数
    books: list[str]
    evidence: list[Evidence]


class RefRow(TypedDict):
    """参考表 eras_ref.json 的一条: 一个政权的一个年号."""
    start: int | None       # 元年; 参考表里个别条目没有
    end: int | None
    years: str              # 如 "6年"
    regime: str             # 政权, 如 "东汉" "蜀汉" "孫吳"
    emperor: str
    period_raw: str         # 参考表原文的起止, 如 "184年十二月－189年三月"
    note: NotRequired[str]
    source: str             # 维基百科固定修订版的链接


Status = Literal["ok", "ref", "多政权", "mismatch", "待核", "conflict"]  # conflict 只在合并参考表之前出现


class Era(TypedDict):
    """eras.json 的一条 (年号 -> Era). 字段含义见模块 docstring."""
    start: int | None
    status: Status
    uses: int
    books: list[str]
    candidates: list[Candidate]
    ref: NotRequired[list[RefRow]]          # 有 eras_ref.json 时才有, 下同
    by_regime: NotRequired[dict[str, int]]  # 政权 (简体) -> 元年
    verified: NotRequired[bool]             # 只有 status "ok" 才有


def cn_number(s: str) -> int:
    """年数的汉字 -> int: 元 1, 九 9, 十 10, 十二 12, 二十 20, 二十四 24. 不认识的抛 ValueError."""
    if s == "元":
        return 1
    if s in _DIGITS:
        return _DIGITS[s]
    if "十" in s and s.count("十") == 1:
        tens, ones = s.split("十")
        if (tens and tens not in _DIGITS) or (ones and ones not in _DIGITS):
            raise ValueError(s)
        return (_DIGITS[tens] if tens else 1) * 10 + (_DIGITS[ones] if ones else 0)
    raise ValueError(s)


def parse_yl(first: str, second: str | None) -> tuple[str, int, int | None] | None:
    """一个 YL 模板 -> (年号, 第几年, 公元年或 None). 不是 "两字年号 + 数字 + 年" 的返回 None."""
    m = _ERA_YEAR.match(first.strip())
    if not m or m.group(1) in NOT_ERAS:
        return None
    try:
        n = cn_number(m.group(2))
    except ValueError:
        return None
    ce = None
    if second:
        c = re.fullmatch(r"\s*(\d{1,4})年?\s*", second)
        ce = int(c.group(1)) if c else None
    return m.group(1), n, ce


def juan_books() -> dict[int, str]:
    """卷号 -> 所属的书 ("魏書" 等), 取自 sanguozhi.jsonl."""
    books: dict[int, str] = {}
    with open(config.DATA_DIR / "processed" / "sanguozhi.jsonl", encoding="utf-8") as f:
        for p in map(json.loads, f):
            books.setdefault(p["juan"], p["book"])
    return books


def build() -> dict[str, Era]:
    """扫描全部 raw 文件的 YL 模板, 生成年号表 (格式见模块 docstring); 有 eras_ref.json 时合并参考表."""
    books = juan_books()
    uses: Counter[str] = Counter()
    where: dict[str, set[str]] = defaultdict(set)
    cands: dict[str, dict[int, list[Evidence]]] = defaultdict(lambda: defaultdict(list))
    for f in sorted(RAW_WIKI.glob("卷*.wiki")):
        m_juan = re.search(r"\d+", f.stem)
        assert m_juan, f  # 文件名都是 卷NN.wiki
        juan = int(m_juan.group())
        for m in _YL.finditer(f.read_text(encoding="utf-8")):
            parsed = parse_yl(m.group(1), m.group(2))
            if not parsed:
                continue
            era, n, ce = parsed
            uses[era] += 1
            where[era].add(books.get(juan, "?"))
            if ce is not None:
                cands[era][ce - (n - 1)].append({"juan": juan, "book": books.get(juan, "?"), "yl": m.group(1) + "|" + m.group(2)})
    table: dict[str, Era] = {}
    for era in sorted(uses):
        cs: list[Candidate] = sorted(({"start": s, "count": len(ev), "books": sorted({e["book"] for e in ev}), "evidence": ev}
                     for s, ev in cands[era].items()),
                    key=lambda c: (-c["count"], c["start"]))
        status: Status = "待核" if not cs else ("ok" if len(cs) == 1 else "conflict")
        table[era] = {"start": cs[0]["start"] if status == "ok" else None, "status": status, "uses": uses[era],
                      "books": sorted(where[era]), "candidates": cs}
    if REF.exists():
        merge_ref(table, json.loads(REF.read_text(encoding="utf-8")))
    return table


def merge_ref(table: dict[str, Era], ref: dict[str, list[RefRow]]) -> None:
    """把参考表 (eras_ref.json) 合进来, 规则见模块 docstring. 原地修改 table."""
    for era, e in table.items():
        rows = [r for r in ref.get(era, []) if r.get("regime") in REGIMES and r.get("start") is not None]
        by_regime = {r["regime"].replace("孫吳", "孙吴"): r["start"] for r in rows if r["start"] is not None}
        e["ref"], e["by_regime"] = rows, by_regime
        starts = set(by_regime.values())
        if e["status"] == "ok":
            if starts and e["start"] not in starts:
                e["status"], e["start"] = "mismatch", None
            else:
                e["verified"] = bool(starts)
        elif e["status"] == "conflict":
            mine = {c["start"] for c in e["candidates"]}
            e["status"] = "多政权" if mine <= starts and len(by_regime) > 1 else "mismatch"
        elif len(by_regime) == 1:
            e["status"], e["start"] = "ref", rows[0]["start"]
        elif len(by_regime) > 1:
            e["status"] = "多政权"


def load() -> dict[str, Era]:
    """读 data/eras.json."""
    return json.loads(OUT.read_text(encoding="utf-8"))


def era_to_ce(era: str, n: int, table: dict[str, Era] | None = None, regime: str | None = None) -> int | None:
    """年号第 n 年的公元年. status 为 ok / ref 时直接给; "多政权" 时要传 regime (如 "蜀汉"); 其他情况返回 None."""
    e = (table if table is not None else load()).get(era)
    if not e:
        return None
    if e["status"] in ("ok", "ref"):
        assert e["start"] is not None  # ok / ref 一定有元年
        return e["start"] + n - 1
    by_regime = e.get("by_regime", {})
    if e["status"] == "多政权" and regime in by_regime:
        return by_regime[regime] + n - 1
    return None


def main() -> None:
    table = build()
    OUT.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    by = Counter(e["status"] for e in table.values())
    covered_uses = sum(e["uses"] for e in table.values() if e["status"] in ("ok", "ref", "多政权"))
    print(f"{len(table)} 个年号 -> {OUT}")
    print("  " + "  ".join(f"{k} {v}" for k, v in by.most_common()))
    print(f"  原文标注与参考表一致 (verified): {sum(1 for e in table.values() if e.get('verified'))}")
    print(f"  按出现次数: {covered_uses}/{sum(e['uses'] for e in table.values())} 处的年号有确定的元年")
    for era, e in table.items():
        if e["status"] in ("conflict", "mismatch", "多政权"):
            print(f"  {e['status']} {era}: 原文 " + ", ".join(f"{c['start']}({c['count']}次, {'/'.join(c['books'])})" for c in e["candidates"])
                  + f" | 参考表 {e.get('by_regime')}")
    top = sorted((e["uses"], era) for era, e in table.items() if e["status"] == "待核")[::-1][:15]
    print("  待核 (出现最多的 15 个): " + " ".join(f"{era}{u}" for u, era in top))


if __name__ == "__main__":
    main()
