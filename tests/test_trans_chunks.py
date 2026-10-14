"""ingest/trans_chunks.py 的测试 (用手写的小数据, 不读真实数据):

    uv run python -m tests.test_trans_chunks
"""
from ingest import trans_chunks


def test_clean_removes_translator_notes():
    """〔待核〕〔原文作…〕〔…提示…〕 这类译者批注被去掉."""
    t = "黄初二年〔年号提示为建安，但上文已改元黄初，按上下文补，待核〕四月，刘璋〔待核〕与〔原文作“林”〕张飞"
    assert trans_chunks.clean(t) == "黄初二年四月，刘璋与张飞"


def test_clean_keeps_collation_chars():
    """原文带过来的短校勘字 (如 〔颺〕) 保留."""
    assert trans_chunks.clean("单飏〔颺〕说") == "单飏〔颺〕说"


def test_attach_joins_sources_in_order_and_reports_missing():
    """多个源的译文按顺序拼接并替换 search; 缺译文的 chunk 保留原文 search 并报告缺的源 id; 不改输入."""
    chunks = [{"id": "01-0004", "sources": ["01-0004", "01-0005"], "search": "原文"},
              {"id": "01-0006", "sources": ["01-0006"], "search": "原文6"}]
    new, missing = trans_chunks.attach(chunks, {"01-0004": "甲", "01-0005": "乙〔待核〕"})
    assert new[0]["trans"] == "甲\n乙" and new[0]["search"] == "甲\n乙"
    assert new[1]["trans"] == "" and new[1]["search"] == "原文6", "缺译文时保留原文 search"
    assert missing == ["01-0006"]
    assert chunks[0]["search"] == "原文", "不改输入"


def test_slice_translation_by_proportion_snaps_to_sentence():
    """按原文区间的比例切译文, 切点落在句末标点后; 覆盖整段时不切."""
    tr = "第一句话。第二句话。第三句话。第四句话。"
    # 原文 100 字, 这块是后一半 [50, 100)
    assert trans_chunks.slice_translation(tr, 50, 50, 100) == "第三句话。第四句话。"
    assert trans_chunks.slice_translation(tr, 0, 50, 100) == "第一句话。第二句话。"
    assert trans_chunks.slice_translation(tr, 0, 100, 100) == tr, "整段不切"


def test_attach_slices_split_paragraph():
    """长段切出的两块各配一半译文: 起点用 str.find 在原文段里找, 不用 offset 字段 (那是裴注挂在正文里的位置, 两块相同)."""
    src = "甲" * 50 + "乙" * 50
    chunks = [{"id": "01-0076.0", "sources": ["01-0076"], "text": "甲" * 50, "offset": 7, "search": "原"},
              {"id": "01-0076.1", "sources": ["01-0076"], "text": "乙" * 50, "offset": 7, "search": "原"}]
    new, _ = trans_chunks.attach(chunks, {"01-0076": "甲甲甲。乙乙乙。"}, {"01-0076": src})
    assert new[0]["trans"] == "甲甲甲。" and new[1]["trans"] == "乙乙乙。"


def test_attach_short_tail_plus_next_paragraph():
    """"长段 A 的末尾 (不足 30 字) + 换行 + 下一段 B" 的块: 只配 A 末尾那一句的译文和 B 的整段译文, 不拿 A 的整段译文."""
    a = "甲" * 40 + "。" + "乙" * 40 + "。" + "未至，權退。"
    b = "十四年，" + "丙" * 30 + "。"
    chunk = {"id": "47-0007.1", "sources": ["A", "B"], "text": "未至，權退。\n" + b, "search": "原"}
    trans = {"A": "第一句。第二句。没到，孙权就撤退了。", "B": "建安十四年，丙。"}
    new, _ = trans_chunks.attach([chunk], trans, {"A": a, "B": b})
    assert new[0]["trans"] == "没到，孙权就撤退了。\n建安十四年，丙。"


def test_portion_tolerates_newline_inside_one_paragraph():
    """切块在同一段里插了换行 (原文段里没有) 时, 两行合起来仍算这一段的同一部分."""
    src = "甲甲。乙乙。丙丙。"
    assert trans_chunks._portion("乙乙。\n丙丙。", src) == (3, 6)
    assert trans_chunks._portion("甲甲。乙乙。\n丙丙。", src) == (0, 9)


def test_slice_translation_aligns_by_sentence_ordinal():
    """原文和译文句末数相等时按序号对齐; 按比例切会少一句. 句末数不等时退回按比例切."""
    tr = "一一一一一一一一。二。三。"
    assert trans_chunks.slice_translation(tr, 0, 4, 6, "甲。乙。丙。") == "一一一一一一一一。二。"
    assert trans_chunks.slice_translation(tr, 4, 2, 6, "甲。乙。丙。") == "三。"
    assert trans_chunks.slice_translation(tr, 0, 4, 6) == "一一一一一一一一。", "不给原文时按比例"
    assert trans_chunks.slice_translation(tr, 0, 3, 5, "甲乙。丙。") == "一一一一一一一一。", "句末数不等时按比例"


def test_closing_quote_stays_with_previous_chunk():
    """切点落在 "！”" 这类组合之后, 闭引号不落到下一块开头 (按序号和按比例两种切法都是)."""
    src = "甲曰：“善！”乙。"
    tr = "甲说：“好！”乙。"
    first = trans_chunks.slice_translation(tr, 0, 7, len(src), src)
    second = trans_chunks.slice_translation(tr, 7, 2, len(src), src)
    assert (first, second) == ("甲说：“好！”", "乙。")
    tr2 = "某某说：“太好了！”" + "乙" * 8 + "。"
    assert trans_chunks.slice_translation(tr2, 50, 50, 100) == "乙" * 8 + "。", "按比例切也跳过闭引号"
    assert trans_chunks._snap("说：“好！”乙。", 5) == 6
    assert trans_chunks._snap('说："好！"乙。', 5) == 6, "半角闭引号 (前面有奇数个 \") 也跳过"
    assert trans_chunks._snap('好。"乙"一语。', 2) == 2, "半角开引号留给下一块"


if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
