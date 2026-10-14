"""rag/fusion.rrf 的测试: 融合分数, 同分时的顺序, rrf_k 参数和空输入.

    uv run python -m tests.test_fusion

不依赖 pytest, 不加载模型.
"""
from rag import fusion


def test_rrf_docstring_example():
    """fusion.py 模块 docstring 里的例子 (RRF_K = 60)."""
    got = fusion.rrf([["A", "B", "C"], ["C", "A", "D"]])
    assert [cid for cid, _ in got] == ["A", "C", "B", "D"]
    assert abs(got[0][1] - (1 / 61 + 1 / 62)) < 1e-12 and abs(got[3][1] - 1 / 63) < 1e-12


def test_rrf_ties_keep_first_seen_order():
    """同分时按第一次出现的先后 (先看第 0 路), 结果要确定."""
    assert [cid for cid, _ in fusion.rrf([["A"], ["B"]])] == ["A", "B"]
    assert [cid for cid, _ in fusion.rrf([["B"], ["A"]])] == ["B", "A"]


def test_rrf_k_and_empty():
    assert fusion.rrf([]) == [] and fusion.rrf([[], []]) == []
    got = dict(fusion.rrf([["A", "B"]], rrf_k=0))
    assert got == {"A": 1.0, "B": 0.5}


if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
