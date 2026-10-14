"""rag/context.py 的测试 (format_context). 不依赖 pytest, 不调 LLM:

    uv run python -m tests.test_context

覆盖: 每块的 id 都在, 编号 [1] [2] 的顺序, 两个原文短语 (先统一转简体再查, 所以不验证是否保留繁体), 空输入, 调用前后 chunks 不变.
具体措辞不查.
"""
from rag import context

CHUNKS = [
    {"id": "01-0004", "tier": 1, "juan": 1, "juan_title": "魏書•武帝紀", "headings": [], "parent": None, "cites": [],
     "sources": ["01-0004"], "text": "卓表太祖爲驍騎校尉，欲與計事。太祖乃變易姓名，間行東歸。"},
    {"id": "01-0004n1", "tier": 2, "juan": 1, "juan_title": "魏書•武帝紀", "headings": [], "parent": "01-0004",
     "cites": ["魏書"], "sources": ["01-0004n1"], "text": "《魏書》曰：太祖以卓終必覆敗，遂不就拜，逃歸鄉里。"},
]


# ---------- format_context ----------

def test_format_context_contains_all():
    ctx = context.format_context(CHUNKS)
    assert isinstance(ctx, str)
    for c in CHUNKS:
        assert c["id"] in ctx, f"缺 chunk id {c['id']}"
    assert "[1]" in ctx and "[2]" in ctx, "要按顺序编号 [1] [2]"
    assert ctx.index("[1]") < ctx.index("[2]") < ctx.index("01-0004n1"), "按传入顺序: 第二块的 id 在 [2] 之后"


def test_format_context_text_present():
    """原文短语在输出里 (转简体后比较)."""
    from rag.normalize import to_simplified
    ctx = to_simplified(context.format_context(CHUNKS))
    assert "变易姓名" in ctx and "遂不就拜" in ctx


def test_format_context_empty_and_no_mutation():
    assert context.format_context([]) == ""
    before = [dict(c) for c in CHUNKS]
    context.format_context(CHUNKS)
    assert CHUNKS == before, "不要改 chunks"


if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
