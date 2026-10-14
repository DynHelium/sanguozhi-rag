"""rag/chunk.py 的测试: 切句, 长段切片, 短段合并, 生成块, 写盘顺序.

    uv run python -m tests.test_chunk

不依赖 pytest. 主要检查契约 (不丢字, 不跨小节, 只在句子边界切, id 唯一且确定, 裴注指回正文);
合并方向按 group_units 的规则检查 (小节开头的短单元往后并, 末尾的往前并). 句子怎么打包不约束. 真实数据用例需要 data/processed/sanguozhi.jsonl (先跑 ingest.sanguozhi), 没有则跳过.
"""
import re

from tests._run import Skip
from rag import chunk
from rag.chunk import Unit

_YEAR = re.compile(r" \d+年")
_END = re.compile(f"[{chunk.SENTENCE_END}][{chunk.CLOSING}]*$")


def _unit(para, text, headings=("甲傳",), juan=1, search=None):
    return Unit(para=para, juan=juan, book="魏書", juan_title="魏書•測試", headings=list(headings),
                text=text, search=search or text, variants=[])


# ---------- split_sentences ----------

def test_split_sentences_basic():
    assert chunk.split_sentences("甲！乙？丙。") == ["甲！", "乙？", "丙。"]
    assert chunk.split_sentences("甲，乙") == ["甲，乙"]
    assert chunk.split_sentences("") == []


def test_split_sentences_closing_quote_stays():
    assert chunk.split_sentences("曰：「善。」遂行。") == ["曰：「善。」", "遂行。"]
    assert chunk.split_sentences("書曰：「臣聞之。」』丁") == ["書曰：「臣聞之。」』", "丁"]


def test_split_sentences_invariants():
    for text in ["甲。乙曰：「丙。丁！」戊？\n己，庚", "曄議曰：", "。。」甲", "甲。\n乙。"]:
        sents = chunk.split_sentences(text)
        assert "".join(sents) == text, text
        assert all(s for s in sents), text
        for s in sents[:-1]:
            assert _END.search(s), (text, s)


# ---------- split_long ----------

def _check_split_long(text, search, max_len):
    pieces = chunk.split_long(text, search, max_len)
    assert "".join(t for t, _ in pieces) == text
    assert "".join(s for _, s in pieces) == search
    for t, s in pieces:
        assert _YEAR.sub("", s) == t, (t, s)
        if len(t) > max_len:
            assert len(chunk.split_sentences(t)) == 1, f"超长片只能是单句: {t!r}"
    for t, _ in pieces[:-1]:
        assert _END.search(t), f"只能在句子边界切: {t!r}"
    return pieces


def test_split_long_short_text_unchanged():
    assert chunk.split_long("甲。乙。", "甲。乙。", 10) == [("甲。乙。", "甲。乙。")]


def test_split_long_lockstep_with_years():
    text = "建安五年，戰於官渡。紹軍大潰。太祖還許。建安七年，紹薨。"
    search = "建安五年 200年，戰於官渡。紹軍大潰。太祖還許。建安七年 202年，紹薨。"
    pieces = _check_split_long(text, search, 12)
    assert len(pieces) > 1


def test_split_long_oversized_sentence_kept_whole():
    text = "甲" * 30 + "。乙。"
    pieces = _check_split_long(text, text, 10)
    assert pieces[0][0] == "甲" * 30 + "。"


# ---------- group_units ----------

def _check_groups(units, min_len):
    groups = chunk.group_units(units, min_len)
    assert [u for g in groups for u in g] == units, "分组连起来必须等于原单元序列"
    sections = {}
    for u in units:
        sections.setdefault((u.juan, tuple(u.headings)), []).append(u)
    for g in groups:
        assert g, "不应有空组"
        assert len({(u.juan, tuple(u.headings)) for u in g}) == 1, "一组不能跨卷或跨小节"
        if len(g) > 1:
            assert any(len(u.text) < min_len for u in g), "两个够长的单元不应合并"
        if sum(len(u.text) for u in g) < min_len:
            assert g == sections[(g[0].juan, tuple(g[0].headings))], "短组只能是整个小节"
    return groups


def test_group_merges_short_lead_in():
    lead, body = _unit("14-0043", "曄議曰："), _unit("14-0044", "聖帝孝孫之欲褒崇先祖，誠無量已。" * 4)
    assert _check_groups([lead, body], 20) == [[lead, body]]


def test_group_merges_short_reply_somewhere():
    a = _unit("22-0012", "故聖人制禮，或抑或致，以求厥中。" * 3)
    b = _unit("22-0013", "帝不聽。")
    c = _unit("22-0014", "青龍中，營治宮室，百姓失農時。" * 3)
    groups = _check_groups([a, b, c], 20)
    assert len(groups) == 2


def test_group_never_crosses_sections():
    a = _unit("01-0001", "甲" * 30, headings=["甲傳"])
    b = _unit("01-0002", "乙。", headings=["乙傳"])
    c = _unit("02-0000", "丙。", headings=["乙傳"], juan=2)
    assert _check_groups([a, b, c], 20) == [[a], [b], [c]]


def test_group_short_lead_before_next_section():
    """小节开头的短单元 (传的首段 "某字某, 某地人也") 往后并; 后面还有别的小节时也一样."""
    lead = _unit("11-0000", "袁渙字曜卿，陳郡扶樂人也。父滂，為漢司徒。", headings=["袁渙"])
    body = _unit("11-0001", "當時諸公子多越法度，而渙清靜，舉動必以禮。" * 2, headings=["袁渙"])
    nxt = _unit("11-0006", "張範，字公儀，河內脩武人也。祖父歆，為漢司徒。" * 2, headings=["張範"])
    assert _check_groups([lead, body, nxt], 30) == [[lead, body], [nxt]]


def test_group_short_tail_before_next_section():
    """小节末尾的短单元往前并; 后面有别的小节, 或者它在全部单元的最后, 都一样."""
    a = _unit("01-0001", "甲" * 30, headings=["甲傳"])
    b = _unit("01-0002", "乙。", headings=["甲傳"])
    c = _unit("01-0003", "丙" * 30, headings=["丙傳"])
    d = _unit("01-0004", "丁。", headings=["丙傳"])
    assert _check_groups([a, b, c, d], 20) == [[a, b], [c, d]]


def test_group_long_units_untouched():
    units = [_unit(f"01-{i:04d}", "甲" * 30) for i in range(3)]
    assert _check_groups(units, 20) == [[u] for u in units]


def test_group_runs_of_short_units():
    units = [_unit(f"01-{i:04d}", "短。") for i in range(5)] + [_unit("01-0005", "甲" * 30)]
    _check_groups(units, 20)


# ---------- make_chunks ----------

def _para(pid, text, notes=(), headings=("甲傳",), search=None):
    juan = int(pid[:2])
    return {"id": pid, "juan": juan, "book": "魏書", "juan_title": "魏書•測試", "headings": list(headings),
            "text": text, "search": search or text, "variants": [],
            "notes": [{"offset": o, "text": t, "search": t, "variants": [], "cites": []} for o, t in notes]}


def _check_chunks(paragraphs, min_len, max_len):
    chunks = chunk.make_chunks(paragraphs, min_len, max_len)
    tiers = [c.tier for c in chunks]
    assert tiers == sorted(tiers), "先列全部正文块, 再列全部裴注块"
    main = [c for c in chunks if c.tier == 1]
    notes = [c for c in chunks if c.tier == 2]
    flat = lambda xs: "".join(x.replace("\n", "") for x in xs)
    assert flat(c.text for c in main) == flat(p["text"] for p in paragraphs), "正文丢字或乱序"
    assert flat(c.text for c in notes) == flat(n["text"] for p in paragraphs for n in p["notes"]), "裴注丢字或乱序"

    ids = [c.id for c in chunks]
    assert len(set(ids)) == len(ids), "chunk id 重复"
    assert ids == [c.id for c in chunk.make_chunks(paragraphs, min_len, max_len)], "chunk id 不确定"

    by_id = {p["id"]: p for p in paragraphs}
    note_ids = {chunk.note_id(p["id"], i) for p in paragraphs for i in range(len(p["notes"]))}
    covered = set()
    for c in chunks:
        assert c.id.startswith(c.sources[0]), c.id
        assert c.work == "三国志"
        assert _YEAR.sub("", c.search).find(c.text) >= 0, f"search 去掉公元年后应包含 text: {c.id}"
        covered.update(c.sources)
        if c.tier == 1:
            assert c.parent is None and c.offset is None and all(s in by_id for s in c.sources), c.id
            assert len(set(c.sources)) == len(c.sources), c.id
        else:
            assert len(c.sources) == 1 and c.sources[0] in note_ids, c.id
            assert c.sources[0].startswith(f"{c.parent}n"), c.id
            assert 0 <= c.offset <= len(by_id[c.parent]["text"]), c.id
            assert (c.juan, c.headings) == (by_id[c.parent]["juan"], by_id[c.parent]["headings"]), c.id
    assert covered == set(by_id) | note_ids, "有源 id 没有被任何 chunk 覆盖"
    return chunks


def test_make_chunks_small():
    paras = [
        _para("01-0000", "太祖武皇帝，沛國譙人也。" * 5, notes=[(5, "《曹瞞傳》曰：太祖一名吉利。"), (60, "音義。")]),
        _para("01-0001", "曄議曰："),
        _para("01-0002", "建安五年，戰於官渡。" * 8, search="建安五年 200年，戰於官渡。" * 8,
              notes=[(10, "《魏書》曰：" + "紹軍大潰。" * 20)]),
        _para("01-0003", "帝不聽。", headings=["乙傳"]),
    ]
    chunks = _check_chunks(paras, 20, 40)
    note_chunks = [c for c in chunks if c.tier == 2]
    assert [c.parent for c in note_chunks[:2]] == ["01-0000", "01-0000"]
    assert note_chunks[0].offset == 5 and note_chunks[1].offset == 60
    assert len(note_chunks) > 3, "超过 max_len 的裴注应被切开"


def test_real_data():
    if not chunk.PARAS_PATH.exists():
        raise Skip("没有 data/processed/sanguozhi.jsonl")
    paras = chunk.load_paragraphs()
    _check_groups(chunk.to_units(paras, 400), 50)
    chunks = _check_chunks(paras, 50, 400)
    for c in chunks:
        if c.tier == 2 and len(c.text) > 400:
            assert len(chunk.split_sentences(c.text)) == 1, f"裴注块超长且不是单句: {c.id} {len(c.text)}"



def test_interleave_puts_notes_after_their_paragraph():
    """写盘顺序: 正文块后紧跟它段落的裴注; 段落切成几块时注跟在最后一块后面 (按文件顺序读相邻块时依赖这个顺序)."""
    from rag.chunk import Chunk, interleave
    def mk(id, tier, sources, parent=None):
        return Chunk(id=id, tier=tier, juan=1, juan_title="", headings=[], book="", text=id, search=id,
                     sources=sources, parent=parent)
    cs = [mk("01-0004.0", 1, ["01-0004"]), mk("01-0004.1", 1, ["01-0004"]), mk("01-0005", 1, ["01-0005"]),
          mk("01-0004n0", 2, ["01-0004n0"], "01-0004"), mk("01-0004n1", 2, ["01-0004n1"], "01-0004"),
          mk("01-0005n0", 2, ["01-0005n0"], "01-0005")]
    assert [c.id for c in interleave(cs)] == ["01-0004.0", "01-0004.1", "01-0004n0", "01-0004n1", "01-0005", "01-0005n0"]

if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
