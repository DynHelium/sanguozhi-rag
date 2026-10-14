"""出题标注工具: 在全书正文和裴注里找词, 列出源 id, 用来给检索评测题标 gold.

    uv run python -m eval.find 官渡                 # 含 "官渡" 的段落和裴注
    uv run python -m eval.find 关羽 刘备 -n 50      # 多个词: 同时包含全部词的才列出
    uv run python -m eval.find 关羽 --main          # 只搜正文 (--notes 只搜裴注)
    uv run python -m eval.find --id 01-0046         # 看某个源 id 的全文 (段落或裴注, 如 01-0000n0)

输出每行一个命中:
    01-0046    正文  魏書•武帝紀            …攻東郡太守劉延於白馬，紹引兵至黎陽…
    01-0000n0  裴注  魏書•武帝紀  →01-0000  …《曹瞞傳》曰：太祖一名吉利…
左边就是源 id, 直接抄进题目的 gold. 裴注后面的 →01-0000 是它所注的段落.

匹配规则: 查询词和原文都先繁转简 (OpenCC t2s), 再把异体字 `呉` 当 `吴`, 所以用简体输入即可.
搜的是 search 字段. 维基文库只给全书 262 处年号标了公元年 (如 01-0022 `興平元年 194年`), 大部分年号没有,
所以按公元年 (如 "208年") 搜基本找不到, 要用年号 (如 "建安十三年") 搜.
这只是标注辅助, 不是检索系统: 它的匹配方式与 rag/ 的归一化无关, 不要拿它的结果当评测基线.
"""
import argparse
import json
from typing import Literal, TypedDict

from opencc import OpenCC

from rag import config
from rag.schema import Paragraph

PARAS_PATH = config.DATA_DIR / "processed" / "sanguozhi.jsonl"
_t2s = OpenCC("t2s")


def _norm(s: str) -> str:
    return _t2s.convert(s).replace("呉", "吴")


class Source(TypedDict):
    """一个段落或一条裴注, 摊平后的一项."""
    id: str                       # 段落 id "01-0000" 或裴注 id "01-0000n0"
    kind: Literal["正文", "裴注"]
    title: str                    # 卷名 + 各级标题, 如 "魏書•武帝紀"
    parent: str | None            # 裴注所注的段落 id; 正文为 None
    text: str
    search: str


def load_sources() -> list[Source]:
    """把段落和裴注摊平成一个列表, 按原书顺序 (每段正文后面跟它的裴注)."""
    out: list[Source] = []
    with open(PARAS_PATH, encoding="utf-8") as f:
        for line in f:
            p: Paragraph = json.loads(line)
            title = p["juan_title"] + ("·" + "·".join(p["headings"]) if p["headings"] else "")
            out.append({"id": p["id"], "kind": "正文", "title": title, "parent": None,
                        "text": p["text"], "search": p["search"]})
            for i, n in enumerate(p["notes"]):
                out.append({"id": f"{p['id']}n{i}", "kind": "裴注", "title": title, "parent": p["id"],
                            "text": n["text"], "search": n["search"]})
    return out


def snippet(text: str, term: str, width: int) -> str:
    """第一个命中位置前后各 width 字. 位置在归一化文本上算, t2s 逐字转换, 长度不变, 可直接截原文."""
    i = _norm(text).find(term)
    if i < 0:
        return text[:width * 2]
    lo, hi = max(0, i - width), min(len(text), i + len(term) + width)
    return ("…" if lo else "") + text[lo:hi] + ("…" if hi < len(text) else "")


def main() -> None:
    ap = argparse.ArgumentParser(description="在三国志正文和裴注里找词, 列出源 id")
    ap.add_argument("terms", nargs="*", help="查询词, 多个词表示同时包含")
    ap.add_argument("--id", help="显示某个源 id 的全文")
    ap.add_argument("--main", action="store_true", help="只搜正文")
    ap.add_argument("--notes", action="store_true", help="只搜裴注")
    ap.add_argument("-n", type=int, default=30, help="最多列出几条 (默认 30)")
    ap.add_argument("-w", type=int, default=25, help="命中处前后各显示几个字 (默认 25)")
    args = ap.parse_args()
    sources = load_sources()

    if args.id:
        hit = next((s for s in sources if s["id"] == args.id), None)
        if not hit:
            raise SystemExit(f"没有这个源 id: {args.id}")
        parent = f"  (注的是 {hit['parent']})" if hit["parent"] else ""
        print(f"{hit['id']}  {hit['kind']}  {hit['title']}{parent}\n\n{hit['text']}")
        return
    if not args.terms:
        ap.error("给出查询词, 或用 --id")

    terms = [_norm(t) for t in args.terms]
    kinds = {"正文"} if args.main else {"裴注"} if args.notes else {"正文", "裴注"}
    hits = [s for s in sources if s["kind"] in kinds and all(t in _norm(s["search"]) for t in terms)]
    for s in hits[:args.n]:
        parent = f"→{s['parent']}" if s["parent"] else ""
        print(f"{s['id']:<11}{s['kind']}  {s['title']:<14} {parent:<9} {snippet(s['text'], terms[0], args.w)}")
    more = f", 只显示前 {args.n} 条 (用 -n 调整)" if len(hits) > args.n else ""
    print(f"\n共 {len(hits)} 条 (正文 {sum(h['kind'] == '正文' for h in hits)}, 裴注 {sum(h['kind'] == '裴注' for h in hits)}){more}")


if __name__ == "__main__":
    main()
