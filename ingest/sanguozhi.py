"""解析三国志和裴注: raw wikitext -> data/processed/sanguozhi.jsonl, 一行一个正文段落, 裴注嵌在段落里.

    uv run python -m ingest.sanguozhi
    uv run python -m tests.test_sanguozhi

输入: data/sources/sanguozhi/raw/卷NN.wiki (先用 ingest.fetch_wikisource 下载) 和 patches.yaml.
输出: 每行一个 rag/schema.Paragraph. 仓库里只放去掉标点的白文 (ingest/baiwen.py), 这个文件由使用者自己生成.

    卷01.wiki: `太祖武皇帝，沛國譙人也，……漢相國參之後。{{*|〔《[[曹瞞傳]]》曰：〕太祖一名吉利，……}}桓帝世，……`
        |
        v
    {"id": "01-0000", "juan": 1, "book": "魏書", "juan_title": "魏書•武帝紀", "headings": [],
     "text": "太祖武皇帝，沛國譙人也……", "search": "太祖武皇帝，沛國譙人也……", "variants": [],
     "notes": [{"offset": 29, "text": "《曹瞞傳》曰：太祖一名吉……", "cites": ["曹瞞傳", "魏書"], ...}, ...]}

每卷的处理顺序 (parse_juan)
    1. apply_patches -> unwrap_links -> 删 <!-- 注释 --> (见 ingest/wikitext.py).
    2. separate_notes: {{*|...}} 和 {{*s}}...{{*e}} 两种写法都取出注文, 在正文原位置留一个私用区字符
       (U+E000 + 序号) 作标记. 之后正文无论怎么清洗, 都能从标记找回裴注挂在第几个字之后.
    3. split_lines: 只在模板外的换行处切行, quote 等多行模板保持完整.
       `== 标题 ==` 行更新标题路径; 空行是段落边界; 行首 `:` 缩进去掉.
    4. 每段 clean_fragment 清洗, 取出标记位置作为裴注的 offset.

设计
    - 段落 id 是 "卷号-段内序号", 只由 raw 的 revid 和 patches.yaml 决定, 切块, 评测, 别名表都靠它.
    - 单独成段 (前后都是空行) 的注文挂到上一段末尾 (offset = 上一段长度): 裴注总是注它前面的正文.
    - cites 是启发式抽取 (cites_of), 可能漏收或误收, 只用于展示和过滤, 不作史实依据.
    - 半角空格与制表符一律删掉 (古文不用空格); 只有检索文本里年号和公元年之间的空格保留.
"""
import json
import re
import sys
from pathlib import Path
from typing import TypedDict

import yaml

from rag import config
from rag.schema import Note, Paragraph, Variant
from ingest.wikitext import (Patch, Template, _COMMENT, apply_patches, clean_fragment, clean_markup, scan,
                             split_args, unwrap_links)

SGZ_DIR = config.DATA_DIR / "sources" / "sanguozhi"
RAW_DIR = SGZ_DIR / "raw"
OUT_PATH = config.DATA_DIR / "processed" / "sanguozhi.jsonl"

MARK_BASE = 0xE000  # 私用区, 原文里不会出现
_MARK = re.compile("[-]")
_HEADING = re.compile(r"(=+)\s*(.+?)\s*\1")
_CITE = re.compile(r"《([^《》]{1,20})》(?=[^《》，。]{0,3}?[曰云載载稱称言說说：])")
_LEAD_CITE = re.compile(r"^(?:案|按)?([^\s，。：、「」『』《》“”（）]{2,10}?)(?:曰|云|載|载|稱|称)")


def book_of(juan: int) -> str:
    """卷 1-30 魏書, 31-45 蜀書, 46-65 吳書."""
    return "魏書" if juan <= 30 else "蜀書" if juan <= 45 else "吳書"


def separate_notes(text: str) -> tuple[str, list[str]]:
    """把裴注从正文里取出. 返回 (带标记的正文 wikitext, 注文 wikitext 列表); 第 i 条注文的标记是 chr(MARK_BASE + i).

    {{*|注文}}: 注文是第一个 | 之后的全部内容.
    {{*s}} ... {{*e}}: 两个标记之间的所有内容 (可跨行, 可含模板) 是注文; 两个标记在同一层 (不在同一层时 raise ValueError).
    裴注也会嵌在别的模板里 (如 quote 引的书信中间夹注), 所以递归进模板内部处理;
    标记留在模板参数里, render_template 取参数时会把它原样带进正文.
    """
    notes: list[str | None] = []
    return _separate(text, notes), notes  # type: ignore[return-value]


def _separate(text: str, notes: list[str | None]) -> str:
    main: list[str] = []
    long_note: list[str] | None = None
    long_idx = -1
    for part in scan(text):
        target = main if long_note is None else long_note
        if isinstance(part, str):
            target.append(part)
            continue
        name = split_args(part.raw)[0].strip()
        if name == "*":
            if long_note is not None:
                raise ValueError("长裴注 {{*s}} 内不应再有 {{*|}}")
            main.append(chr(MARK_BASE + len(notes)))
            notes.append(part.raw[part.raw.index("|") + 1:])
        elif name == "*s":
            if long_note is not None:
                raise ValueError("{{*s}} 嵌套")
            long_idx = len(notes)
            notes.append(None)
            main.append(chr(MARK_BASE + long_idx))
            long_note = []
        elif name == "*e":
            if long_note is None:
                raise ValueError("{{*e}} 前没有同层的 {{*s}}")
            notes[long_idx] = "".join(long_note)
            long_note = None
        elif long_note is not None:
            target.append("{{" + part.raw + "}}")  # 长注文内部的模板留给 clean_note
        else:
            main.append("{{" + _separate(part.raw, notes) + "}}")
    if long_note is not None:
        raise ValueError("{{*s}} 没有同层的 {{*e}}")
    return "".join(main)


def split_lines(wikitext: str) -> list[str]:
    """只在模板外的换行处切行; 多行模板 (quote, header 等) 整个留在它开始的那一行."""
    lines = [""]
    for part in scan(wikitext):
        if isinstance(part, Template):
            lines[-1] += "{{" + part.raw + "}}"
            continue
        pieces = part.split("\n")
        lines[-1] += pieces[0]
        lines.extend(pieces[1:])
    return lines


def _squeeze(s: str, keep_year_space: bool = False) -> str:
    """删半角空格/制表符 (检索文本保留年号与公元年之间的空格), 合并多余换行."""
    s = re.sub(r"[ \t]+(?!\d+年)", "", s) if keep_year_space else re.sub(r"[ \t]+", "", s)
    return re.sub(r"\n{2,}", "\n", s)


def _strip_with_marks(s: str) -> tuple[str, list[tuple[int, int]]]:
    """去掉标记字符, 返回 (文本, [(注文序号, offset)]), 并去首尾空白 (offset 随之调整)."""
    out, marks = [], []
    for ch in s:
        if _MARK.match(ch):
            marks.append((ord(ch) - MARK_BASE, len(out)))
        else:
            out.append(ch)
    text = "".join(out)
    lead = len(text) - len(text.lstrip())
    text = text.strip()
    return text, [(i, min(max(o - lead, 0), len(text))) for i, o in marks]


def cites_of(note_wikitext: str, note_text: str) -> list[str]:
    """注文所引的书名 (或史评者), 去重, 按出现顺序. 启发式, 来源有四种:
    书名线模板 {{WavyBookMark|...}}; 紧跟 曰/云/載/稱/言/： (含简体) 的 《书名》; 注文开头不带书名号的
    "吳錄曰", "孫盛曰" (人名也算出处); 出现 "臣松之" 时记 "裴松之" (裴松之自己的按语).
    音注 (如 "橫音光") 没有出处, 返回 [].
    """
    found: list[str] = []

    def walk(wt: str) -> None:
        for part in scan(wt):
            if isinstance(part, Template):
                args = split_args(part.raw)
                if args[0].strip() == "WavyBookMark" and len(args) > 1:
                    found.append(clean_fragment(args[1]).display.strip())
                walk(part.raw)

    walk(note_wikitext)
    found += _CITE.findall(note_text)
    m = _LEAD_CITE.match(note_text)  # 注文开头不带书名号的 "吳錄曰", "孫盛曰"
    if m and m.group(1) not in found:
        found.append(m.group(1))
    if "臣松之" in note_text:
        found.append("裴松之")
    return list(dict.fromkeys(c for c in found if c))


class NoteText(TypedDict):
    """clean_note 的结果: rag/schema.Note 里除 offset 以外的字段, offset 由 parse_juan 按标记位置补上."""
    text: str
    search: str
    variants: list[Variant]
    cites: list[str]


def clean_note(note_wikitext: str) -> NoteText:
    """清洗一条注文的 wikitext, 返回 Note 里除 offset 以外的字段 (offset 在 parse_juan 里按标记位置补上)."""
    wt = re.sub(r"(?m)^:+", "", note_wikitext)
    c = clean_fragment(wt)
    text = _squeeze(c.display).strip()
    return {"text": text, "search": _squeeze(c.search, keep_year_space=True).strip(),
            "variants": c.variants, "cites": cites_of(wt, text)}


def juan_title(text: str) -> str:
    """页眉模板的 section 参数, 如 "魏書•武帝紀"; 没有就是 "". 各卷写法不统一, 原样清洗后保存."""
    m = re.search(r"\|\s*section\s*=\s*([^\n|]*)", text)
    return clean_markup(m.group(1)).strip() if m else ""


def parse_juan(juan: int, raw: str, patches: list[Patch]) -> list[Paragraph]:
    """解析一卷的 raw wikitext, 返回这一卷的段落记录 (格式见模块 docstring). 步骤见模块 docstring."""
    filename = f"卷{juan:02d}.wiki"
    text = _COMMENT.sub("", unwrap_links(apply_patches(filename, raw, patches)))
    title = juan_title(text)
    main, note_wts = separate_notes(text)
    notes = [clean_note(n) for n in note_wts]

    records: list[Paragraph] = []
    headings: dict[int, str] = {}
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        c = clean_fragment("".join(buf))
        buf.clear()
        disp, marks = _strip_with_marks(_squeeze(c.display))
        search, _ = _strip_with_marks(_squeeze(c.search, keep_year_space=True))
        attached: list[Note] = [{**notes[i], "offset": o} for i, o in marks]
        if not disp:
            if not attached:
                return
            if records:  # 单独成段的注文挂到上一段末尾
                prev = records[-1]
                moved: list[Note] = [{**n, "offset": len(prev["text"])} for n in attached]
                prev["notes"] += moved
                return
        records.append({
            "id": f"{juan:02d}-{len(records):04d}", "juan": juan, "book": book_of(juan),
            "juan_title": title, "headings": [headings[k] for k in sorted(headings)],
            "text": disp, "search": search, "variants": c.variants, "notes": attached,
        })

    for line in split_lines(main):
        stripped = line.strip()
        m = _HEADING.fullmatch(stripped)
        if m and "{{" not in stripped and not _MARK.search(stripped):
            flush()
            level = len(m.group(1))
            headings = {k: v for k, v in headings.items() if k < level}
            headings[level] = clean_fragment(m.group(2)).display.strip()
        elif not stripped or not clean_fragment(_MARK.sub("", stripped)).display.strip() and not _MARK.search(stripped):
            flush()  # 空行, 或只有页眉页脚之类清洗后为空的行
        else:
            buf.append(re.sub(r"^\s*:+", "", line))
    flush()
    return records


def main() -> None:
    """读 raw/ 下全部 卷NN.wiki, 按 patches.yaml 打补丁后逐卷解析, 每段一行写入 sanguozhi.jsonl (覆盖), 最后打印卷数, 段数, 正文与裴注字数.

    raw/ 下没有卷文件时退出并提示先运行 ingest.fetch_wikisource.
    """
    files = sorted(RAW_DIR.glob("卷*.wiki"))
    if not files:
        sys.exit(f"没有找到 {RAW_DIR}/卷*.wiki, 先运行 uv run python -m ingest.fetch_wikisource")
    with open(SGZ_DIR / "patches.yaml", encoding="utf-8") as f:
        patches: list[Patch] = yaml.safe_load(f) or []
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_para = n_note = n_char = n_note_char = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for f in files:
            juan = int(f.stem[1:])
            for r in parse_juan(juan, f.read_text(encoding="utf-8"), patches):
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_para += 1
                n_note += len(r["notes"])
                n_char += len(r["text"])
                n_note_char += sum(len(n["text"]) for n in r["notes"])
    print(f"{len(files)} 卷, {n_para} 段, 正文 {n_char} 字; 裴注 {n_note} 条, {n_note_char} 字 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
