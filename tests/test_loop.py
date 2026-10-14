"""rag/loop.py 的测试: agent_loop 回传的消息和轮数上限, 用假模型驱动. 不依赖 pytest, 不连网:

    uv run python -m tests.test_loop
"""
import copy
import json
from types import SimpleNamespace

from rag import loop

TOOLS = [{"type": "function", "function": {"name": "t"}}]


def tc(cid, name, **args):
    return SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)))


def says(content, calls=None):
    """模型的一条回复 (message): 有 content 和 tool_calls."""
    return SimpleNamespace(content=content, tool_calls=calls)


class Script:
    """假的 complete(messages, tools=None): 依次返回 replies, 记下每次收到的 messages 和 tools."""
    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def __call__(self, messages, tools=None):
        self.seen.append({"messages": copy.deepcopy(messages), "tools": tools})
        return self.replies.pop(0)


def _loop(script, max_rounds=loop.MAX_ROUNDS):
    """跑 agent_loop: 假模型 script, 假工具 (回 "结果:名字"). 返回 (回答, messages, 工具调用记录)."""
    called = []

    def run_tool(name, arguments):
        called.append((name, json.loads(arguments)))
        return f"结果:{name}"
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}]
    reply = loop.agent_loop(msgs, TOOLS, script, run_tool, max_rounds=max_rounds)
    return reply, msgs, called


def test_loop_tool_messages_match_calls():
    script = Script(says("", [tc("a", "search", query="x"), tc("b", "era", era="建安", year=7)]), says("答"))
    reply, msgs, called = _loop(script)
    assert reply == "答" and called == [("search", {"query": "x"}), ("era", {"era": "建安", "year": 7})]
    asst = next(m for m in msgs if m["role"] == "assistant")
    assert [c["id"] for c in asst["tool_calls"]] == ["a", "b"] and asst["tool_calls"][0]["function"]["name"] == "search"
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert tool_msgs == [{"role": "tool", "tool_call_id": "a", "content": "结果:search"},
                         {"role": "tool", "tool_call_id": "b", "content": "结果:era"}]
    assert script.seen[0]["tools"] == TOOLS and script.seen[1]["tools"] == TOOLS


def test_loop_answer_without_tools():
    script = Script(says("直接答"))
    reply, msgs, called = _loop(script)
    assert reply == "直接答" and called == [] and len(msgs) == 2, "没调工具就直接返回, messages 不追加"


def test_loop_cap_omits_tools_on_last_round():
    script = Script(*[says("", [tc(f"x{i}", "t")]) for i in range(3)], says("最后", [tc("y", "t")]))
    reply, msgs, called = _loop(script, max_rounds=3)
    assert reply == "最后" and len(called) == 3 and len(script.seen) == 4, "最后一次的 tool_calls 忽略, 不执行"
    assert script.seen[-1]["tools"] is None
    assert script.seen[-1]["messages"][-1] == {"role": "developer", "content": loop.FINAL_TURN}
    assert script.seen[-2]["messages"][-1] == {"role": "developer", "content": loop.LAST_TOOL_ROUND}, "倒数第二次请求前提醒"
    assert {"role": "developer", "content": loop.LAST_TOOL_ROUND} not in script.seen[-3]["messages"], "只在倒数第二次之前提醒"


def test_loop_zero_rounds_goes_straight_to_final():
    script = Script(says("只能答"))
    reply, msgs, _ = _loop(script, max_rounds=0)
    assert reply == "只能答" and script.seen[0]["tools"] is None and msgs[-1]["content"] == loop.FINAL_TURN


def test_loop_missing_tool_call_id_is_filled():
    script = Script(says("", [tc(None, "t")]), says("答"))
    _, msgs, _ = _loop(script)
    asst = next(m for m in msgs if m["role"] == "assistant")
    tool = next(m for m in msgs if m["role"] == "tool")
    assert asst["tool_calls"][0]["id"] and tool["tool_call_id"] == asst["tool_calls"][0]["id"]


def test_loop_none_content():
    """只调工具时 content 常为 None: 放回 messages 时写 "" (有的接口不收 null); 回答为 None 时返回 ""."""
    script = Script(says(None, [tc("a", "t")]), says(None))
    reply, msgs, _ = _loop(script, max_rounds=1)
    assert next(m for m in msgs if m["role"] == "assistant")["content"] == "" and reply == ""


if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
