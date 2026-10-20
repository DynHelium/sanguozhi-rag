"""eval/run_rag.py 的测试: 各组的 system prompt, 只写功能的工具说明, oracle 取块, 截断时重答. 不连网, 不调模型:

    uv run python -m tests.test_run_rag
"""
from types import SimpleNamespace

from eval import run_rag as R
from rag import agent


def test_system_prompts():
    lang = R.P_BASE.rsplit("\n", 1)[1]
    for g in R.GROUPS:
        s = R.system_for(g)
        assert s.endswith(lang), g  # 作答语言那一行始终在最后
        assert (R.TOOL_LINE in s) == (g in ("agentic", "harness")), g
        assert (R.HARNESS in s) == (g == "harness"), g
    assert R.system_for("closed") == R.system_for("static") == R.system_for("oracle") == R.P_BASE
    assert R.system_for("harness").startswith(R.system_for("agentic").rsplit("\n", 1)[0])  # harness 只在 agentic 上追加
    assert "引" not in R.P_BASE and "[" not in R.P_BASE  # 不要求引原文和编号


def test_tools_plain():
    names = lambda ts: [(t["function"]["name"], sorted(t["function"]["parameters"]["properties"]),  # noqa: E731
                         t["function"]["parameters"]["required"]) for t in ts]
    assert names(R.TOOLS_PLAIN) == names(agent.TOOLS)  # 工具和参数不变, 只换说明
    text = str([t["function"] for t in R.TOOLS_PLAIN])
    for rule in ("分别搜", "再作答", "先 read", "直接 search", "一次只搜", "可以拿去 search", "分清"):
        assert rule not in text, rule
    assert "分别搜" in str(agent.TOOLS)  # 原来的说明没被改动 (深拷贝)


def test_oracle_chunks():
    chunks = [{"id": "01-0001", "sources": ["01-0001"]}, {"id": "01-0002.0", "sources": ["01-0002"]},
              {"id": "01-0002.1", "sources": ["01-0002", "01-0003"]}, {"id": "01-0002n0", "sources": ["01-0002n0"]}]
    assert [c["id"] for c in R.oracle_chunks(chunks, ["01-0002"])] == ["01-0002.0", "01-0002.1"]
    assert [c["id"] for c in R.oracle_chunks(chunks, ["01-0003", "01-0002n0"])] == ["01-0002.1", "01-0002n0"]
    assert R.oracle_chunks(chunks, []) == []


def fake(replies):
    """按顺序返回 (content, finish_reason) 的假 complete, 记下每次的 messages."""
    seen = []

    def complete(messages, model=None, **kw):
        seen.append(messages)
        content, finish = replies[len(seen) - 1]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)])
    return complete, seen


def test_one_call_retry():
    complete, seen = fake([("答案", "stop")])
    assert R.one_call(complete, "m", "sys", "问", {}) == ("答案", 1, False)
    complete, seen = fake([("**是董", "length"), ("完整的答案", "stop")])
    assert R.one_call(complete, "m", "sys", "问", {}) == ("完整的答案", 2, True)
    assert seen[0] == seen[1]  # 原样再请求一次
    complete, seen = fake([("", "stop"), ("", "stop")])
    assert R.one_call(complete, "m", "sys", "问", {}) == ("", 2, True)  # 只重答一次


def test_harness_switches():
    assert R.HARNESS_ON <= R.INTERVENTIONS
    assert "review" not in R.HARNESS_ON and "quote_fix" not in R.HARNESS_ON


if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
