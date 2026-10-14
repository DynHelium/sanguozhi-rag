"""rag/quotes.py 的测试 (引文抽取与逐字核对). 不依赖 pytest, 不调 LLM:

    uv run python -m tests.test_quotes
"""
from rag import quotes

CHUNKS = [
    {"id": "01-0004", "tier": 1, "juan": 1, "juan_title": "魏書•武帝紀", "headings": [], "parent": None, "cites": [],
     "sources": ["01-0004"], "text": "卓表太祖爲驍騎校尉，欲與計事。太祖乃變易姓名，間行東歸。"},
    {"id": "01-0004n1", "tier": 2, "juan": 1, "juan_title": "魏書•武帝紀", "headings": [], "parent": "01-0004",
     "cites": ["魏書"], "sources": ["01-0004n1"], "text": "《魏書》曰：太祖以卓終必覆敗，遂不就拜，逃歸鄉里。"},
]


def test_extract_quotes_basic():
    reply = "原文：「太祖乃變易姓名，間行東歸」[1]，又「遂不就拜」[2]。"
    assert quotes.extract_quotes(reply) == ["太祖乃變易姓名，間行東歸", "遂不就拜"]


def test_extract_quotes_none_and_nested():
    assert quotes.extract_quotes("资料中没有记载。") == []
    assert quotes.extract_quotes("「公曰：『方今收英雄時也』，不可」") == ["公曰：『方今收英雄時也』，不可"], "只取最外层"


# ---------- verify_quotes ----------

def test_verify_quotes_found_and_missing():
    got = quotes.verify_quotes(["太祖乃變易姓名，間行東歸", "太祖大怒"], CHUNKS)
    assert got == [("太祖乃變易姓名，間行東歸", "01-0004"), ("太祖大怒", None)]


def test_verify_quotes_simplified_and_punct():
    got = quotes.verify_quotes(["太祖乃变易姓名间行东归", "遂不就拜,逃归乡里"], CHUNKS)
    assert got == [("太祖乃变易姓名间行东归", "01-0004"), ("遂不就拜,逃归乡里", "01-0004n1")], "繁简和标点差别要放过"


def test_verify_quotes_strict_and_empty():
    got = quotes.verify_quotes(["太祖乃改易姓名", ""], CHUNKS)
    assert got == [("太祖乃改易姓名", None), ("", None)], "改了一个字就算找不到; 空引文不算找到"


def test_extract_quotes_unclosed():
    """模型输出被截断时, 没闭合的「 丢掉."""
    assert quotes.extract_quotes("「遂不就拜」, 又「太祖乃變易") == ["遂不就拜"]


def test_verify_quotes_markdown_ellipsis_and_punct_only():
    got = quotes.verify_quotes(["**遂不就拜**", "卓表太祖……間行東歸", "**", "……"], CHUNKS)
    assert got[0] == ("**遂不就拜**", "01-0004n1"), "引文里的 markdown 加粗要放过"
    assert got[1][1] is None, "用省略号跳过中间一段不是逐字引用, 算找不到"
    assert got[2][1] is None and got[3][1] is None, "去掉标点后为空的不算找到"


def test_check_and_fix_cites():
    """「引文」[n] 的编号核对: 对 / 标错 / 找不到; 文字只在一块里时 fix_cites 改成那一块的编号."""
    from rag.quotes import check_cites, fix_cites
    chunks = [{"id": "a", "text": "太祖乃變易姓名，間行東歸。"}, {"id": "b", "text": "遂不就拜。"}]
    reply = "「太祖乃變易姓名，間行東歸」[1]，又「遂不就拜」[1]，「烏有」[2]"
    assert [c["status"] for c in check_cites(reply, chunks)] == ["ok", "wrong", "missing"]
    fixed, n = fix_cites(reply, chunks)
    assert n == 1 and "「遂不就拜」[2]" in fixed and "「太祖乃變易姓名，間行東歸」[1]" in fixed and "「烏有」[2]" in fixed
    assert [c["status"] for c in check_cites(fixed, chunks)] == ["ok", "ok", "missing"]


def test_renumber_cites():
    """回答里写成 chunk id 的出处 (各种括号, 一个或几个) 换成 [n]; 不认识的 id 和已经是 [n] 的不动."""
    chunks = [{"id": "41-0000"}, {"id": "41-0001"}, {"id": "01-0004n1.0"}]
    from rag.quotes import renumber_cites
    r = renumber_cites("「城不可得」[41-0000]; 又（41-0001）; 注 (01-0004n1.0); 两处 [41-0000, 41-0001]; 编的 [99-0001]; 已是 [2]",
                       chunks)
    assert r == "「城不可得」[1]; 又[2]; 注 [3]; 两处 [1][2]; 编的 [99-0001]; 已是 [2]"


if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
