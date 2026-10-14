"""ingest/sanguozhi.py 的测试: 片段清洗和校勘括号 (clean_fragment), 裴注分离, 按行切段, 段落和裴注的挂载.

    uv run python -m tests.test_sanguozhi

不依赖 pytest. 短用例覆盖短注 {{*|...}}, 长注 {{*s}}...{{*e}}, 嵌在 quote 里的注, 单独成段的注文.
真实文件用例对 65 卷整体跑一遍 (需先下载 raw, 没有则跳过): 注文条数 (1763 条短注 + 13 对长注),
清洗后无标记残留, offset 合法, id 唯一且按段落顺序.
"""
import re

import yaml

from tests._run import Skip
from ingest.sanguozhi import (MARK_BASE, RAW_DIR, SGZ_DIR, parse_juan, separate_notes, split_lines)
from ingest.wikitext import apply_brackets, clean_fragment

M0, M1 = chr(MARK_BASE), chr(MARK_BASE + 1)


def test_clean_fragment():
    c = clean_fragment("{{blue|{{-|臣}}吾（爾）〔以〕}}{{校|潘陽|鄱陽}}　{{YL|太和三年|229年}}")
    assert c.display == "臣吾以鄱陽太和三年"
    assert c.search == "臣吾以鄱陽太和三年 229年"
    assert [v["kind"] for v in c.variants] == ["校", "改"]


def test_brackets():
    assert apply_brackets("世（祀）之尊，稽顙〔歸降〕，建興[六]年") == ("世之尊，稽顙歸降，建興六年", [
        {"kind": "删", "adopted": "", "other": "祀"},
        {"kind": "补", "adopted": "歸降", "other": ""},
        {"kind": "补", "adopted": "六", "other": ""},
    ])


def test_separate_short_and_long_notes():
    main, notes = separate_notes("甲{{*|注一}}乙{{*s}}長注{{quote|書}}{{*e}}丙")
    assert main == f"甲{M0}乙{M1}丙"
    assert notes == ["注一", "長注{{quote|書}}"]


def test_separate_note_inside_template():
    main, notes = separate_notes("書曰：{{quote|臣聞{{*|注}}之}}")
    assert main == f"書曰：{{{{quote|臣聞{M0}之}}}}"
    assert notes == ["注"]


def test_separate_unbalanced_long_note_raises():
    for bad in ["甲{{*s}}乙", "甲{{*e}}乙"]:
        try:
            separate_notes(bad)
        except ValueError:
            continue
        raise AssertionError(f"应当报错却没有: {bad!r}")


def test_split_lines_keeps_multiline_template():
    assert split_lines("甲\n{{quote|一\n二}}乙\n丙") == ["甲", "{{quote|一\n二}}乙", "丙"]


def test_parse_juan_structure():
    raw = "\n".join([
        "{{header2\n| section = 魏書·武帝紀\n}}",
        "==張遼==",
        "　　張遼字文遠。{{*|注甲}}",
        "",
        "{{*|單獨成段的注}}",
        "",
        "===附傳===",
        ":評曰：善。",
        "[[category:三國志|1]]",
    ])
    recs = parse_juan(1, raw, [])
    assert [r["id"] for r in recs] == ["01-0000", "01-0001"]
    a, b = recs
    assert a["juan_title"] == "魏書·武帝紀" and a["book"] == "魏書"
    assert a["headings"] == ["張遼"] and a["text"] == "張遼字文遠。"
    assert [(n["text"], n["offset"]) for n in a["notes"]] == [("注甲", 6), ("單獨成段的注", 6)]
    assert b["headings"] == ["張遼", "附傳"] and b["text"] == "評曰：善。"


def test_real_files():
    files = sorted(RAW_DIR.glob("卷*.wiki"))
    if not files:
        raise Skip("没有下载 raw 文件")
    with open(SGZ_DIR / "patches.yaml", encoding="utf-8") as f:
        patches = yaml.safe_load(f) or []
    recs = []
    for f in files:
        recs += parse_juan(int(f.stem[1:]), f.read_text(encoding="utf-8"), patches)
    assert len({r["juan"] for r in recs}) == 65
    assert sum(len(r["notes"]) for r in recs) == 1763 + 13
    assert len({r["id"] for r in recs}) == len(recs)
    residue = re.compile(r"[A-Za-z0-9|{}\[\]<>=　-]|-\{|（|〔")
    for r in recs:
        assert r["text"] and r["juan_title"], r["id"]
        assert not residue.search(r["text"]), (r["id"], r["text"][:30])
        for n in r["notes"]:
            assert 0 <= n["offset"] <= len(r["text"]), r["id"]
            # 注文里允许出现 {罒良} 这类生僻字部件描述以外的残留就报错
            assert not residue.search(re.sub(r"\{[^{}]{2,4}\}", "", n["text"])), (r["id"], n["text"][:30])


if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
