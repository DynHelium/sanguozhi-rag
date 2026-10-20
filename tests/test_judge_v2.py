"""eval/judge_v2.py 的测试: 统一格式, 英文题对应中文题, 按清单算判定. 不连网:

    uv run python -m tests.test_judge_v2
"""
from eval import judge_v2 as J

CK = {"must": [{"id": "m1", "text": "a", "core": True}, {"id": "m2", "text": "b", "core": False}],
      "must_not": [{"id": "x1", "text": "c"}]}


def test_normalize():
    assert J.normalize("原文「太祖乃變易姓名」[3]，又『曰』[1, 2]。") == "原文“太祖乃變易姓名”，又‘曰’。"
    assert J.normalize("建安七年 [3-5] 事") == "建安七年 事"


def test_base_id():
    assert J.base_id("e-h-01") == "h-01" and J.base_id("h-01") == "h-01" and J.base_id("e-hu-03") == "hu-03"


def test_verdict():
    v = lambda must, must_not=None: J.verdict({"must": must, "must_not": must_not or {"x1": "no"}}, CK)  # noqa: E731
    assert v({"m1": "yes", "m2": "yes"}) == "对"
    assert v({"m1": "yes", "m2": "no"}) == "部分"
    assert v({"m1": "yes", "m2": "wrong"}) == "部分"
    assert v({"m1": "yes"}) == "部分"                      # 缺标记当作 no
    for bad in ("no", "wrong", "hedged"):
        assert v({"m1": bad, "m2": "yes"}) == "错"
    assert v({"m1": "yes", "m2": "yes"}, {"x1": "yes"}) == "错"


def test_core_partial():
    ck = {"must": [{"id": "m1", "text": "a", "core": True, "core_partial": True}, {"id": "m2", "text": "b", "core": True}],
          "must_not": []}
    assert J.verdict({"must": {"m1": "no", "m2": "yes"}}, ck) == "部分"
    assert J.verdict({"must": {"m1": "hedged", "m2": "yes"}}, ck) == "部分"
    assert J.verdict({"must": {"m1": "wrong", "m2": "yes"}}, ck) == "错"   # 说错仍算错
    assert J.verdict({"must": {"m1": "yes", "m2": "no"}}, ck) == "错"      # 没标的 core 照旧


def test_asked_and_fabricated():
    ck = {"must": [{"id": "m1", "text": "a", "core": True, "asked": True}, {"id": "m2", "text": "b", "core": False, "asked": True},
                   {"id": "m3", "text": "c", "core": False, "asked": False}], "must_not": []}
    v = lambda must, **kw: J.verdict({"must": must, **kw}, ck)  # noqa: E731
    assert v({"m1": "yes", "m2": "yes", "m3": "no"}) == "对"            # 背景细节没写不扣分
    assert v({"m1": "yes", "m2": "yes", "m3": "wrong"}) == "部分"       # 背景细节写错了扣
    assert v({"m1": "yes", "m2": "no", "m3": "yes"}) == "部分"          # 题目要求的非 core 项漏了
    assert v({"m1": "yes", "m2": "yes"}, fabricated=["编的引文"]) == "部分"
    assert v({"m1": "yes", "m2": "yes"}, extra_errors=["年份错"]) == "部分"
    assert v({"m1": "yes", "m2": "yes"}, unsupported=["查不到"]) == "对"
    assert v({"m1": "no", "m2": "yes"}, fabricated=["x"]) == "错"


if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
