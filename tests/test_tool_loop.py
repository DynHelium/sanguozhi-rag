"""eval/tool_calling.py 的测试: TOOLS (工具声明), check_call (检查一个调用), loop_one (多轮循环).
不依赖 pytest, 不连网, 不加载 bge-m3:

    uv run python -m tests.test_tool_loop

假模型: Script 临时替换 tool_calling.call, 依次返回 (预设的 message, "stop", 0.0), 退出时恢复; 每次收到的 messages
(深拷贝), tools 和其他参数都记下来. message 有 content 和 tool_calls; 每个调用有 id, type="function",
function.name, function.arguments (JSON 字符串), 和真实接口返回的对象同样形状, 用 SimpleNamespace 造.
"""
import copy
import json
from types import SimpleNamespace

from eval import tool_calling as tc
from rag import context

CHUNKS = [{"id": f"c{i}", "tier": 1, "juan": 1, "juan_title": "魏書•武帝紀", "headings": [], "parent": None, "cites": [],
           "sources": [f"s{i}"], "text": f"原文{i}"} for i in range(6)]
BY_ID = {c["id"]: c for c in CHUNKS}
TABLE = {"吕蒙": ["c1", "c2"], "白衣": ["c3"], "白衣渡江是谁的事迹？": ["c0"]}


def fake_search(query, k, where=None):
    return [BY_ID[x] for x in TABLE.get(query, [])][:k]


def tool_call(cid, query, name="search", raw=None):
    args = raw if raw is not None else json.dumps({"query": query}, ensure_ascii=False)
    return SimpleNamespace(id=cid, type="function", function=SimpleNamespace(name=name, arguments=args))


def says(text):
    return SimpleNamespace(content=text, tool_calls=None)


def calls(*tcs):
    return SimpleNamespace(content="", tool_calls=list(tcs))


class Script:
    """替换 tc.call: 依次返回 replies 里的消息. 记录每次收到的 messages (深拷贝) 和 tools."""

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def __enter__(self):
        self.saved = tc.call

        def fake(model, messages, tools=tc.TOOLS, **kw):
            self.seen.append({"messages": copy.deepcopy(messages), "tools": tools, "kw": kw})
            return self.replies.pop(0), "stop", 0.0
        tc.call = fake
        return self

    def __exit__(self, *exc):
        tc.call = self.saved


Q = {"id": "r-10", "q": "白衣渡江是谁的事迹？", "gold": ["s1"]}


# ---------- TOOLS ----------

def test_tools_schema():
    """OpenAI 标准格式: [{"type": "function", "function": {"name", "description", "parameters": JSON Schema}}]."""
    assert isinstance(tc.TOOLS, list) and tc.TOOLS
    search = [t for t in tc.TOOLS if t.get("type") == "function" and t["function"]["name"] == "search"]
    assert search, "要有一个名为 search 的 function"
    f = search[0]["function"]
    assert isinstance(f.get("description"), str) and f["description"], "description 是写给模型看的, 不能空"
    p = f["parameters"]
    assert p["type"] == "object" and p["properties"]["query"]["type"] == "string" and "query" in p["required"]


# ---------- check_call ----------

def test_check_call_valid():
    assert tc.check_call(tool_call("c1", "吕蒙")) == (True, [], "吕蒙")


def test_check_call_problems():
    for bad in [tool_call(None, "吕蒙"),                 # 没有 id: 回传结果时对不上
                tool_call("c1", "吕蒙", name="find"),    # 不是 search
                tool_call("c1", "", raw="{query: 吕蒙"),  # 不是 JSON
                tool_call("c1", "", raw='{"query": ""}'),   # 空检索词
                tool_call("c1", "", raw='{"q": "吕蒙"}')]:  # 缺 query
        ok, errors, query = tc.check_call(bad)
        assert ok is False and errors, f"要判为无效并说明原因: {bad}"


# ---------- loop_one ----------

def test_loop_search_then_answer():
    with Script(calls(tool_call("call_1", "吕蒙")), says("是吕蒙「原文1」")) as s:
        r = tc.loop_one("m", Q, fake_search, BY_ID)
    assert r["end"] == "answer" and r["answer"] == "是吕蒙「原文1」"
    assert r["rounds"][0]["queries"] == ["吕蒙"] and r["rounds"][0]["chunks"] == ["c1", "c2"]
    assert r["recall"] is True, "gold s1 在 c1 里"
    first, second = s.seen[0]["messages"], s.seen[1]["messages"]
    assert first[0]["role"] == "system" and first[-1] == {"role": "user", "content": Q["q"]}
    asst, tool = second[-2], second[-1]
    assert asst["role"] == "assistant" and asst["tool_calls"][0]["id"] == "call_1", "要把模型的调用原样放回 messages"
    assert asst["tool_calls"][0]["function"]["name"] == "search"
    assert tool["role"] == "tool" and tool["tool_call_id"] == "call_1", "tool 消息要带对应的 tool_call_id"
    assert tool["content"] == context.format_context(fake_search("吕蒙", tc.TOOL_K))


def test_loop_parallel_calls():
    """一条回复里有两个调用: 都被执行, 各回一条 tool 消息, 顺序和调用列表一致 (按顺序执行, 不是并发)."""
    with Script(calls(tool_call("a", "吕蒙"), tool_call("b", "白衣")), says("好")) as s:
        r = tc.loop_one("m", Q, fake_search, BY_ID)
    assert r["rounds"][0]["queries"] == ["吕蒙", "白衣"] and r["n_search"] == 2
    tools = [m for m in s.seen[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tools] == ["a", "b"]


def test_loop_no_call():
    with Script(says("我凭记忆答")):
        r = tc.loop_one("m", Q, fake_search, BY_ID)
    assert r["end"] == "no_call" and r["rounds"] == [] and r["recall"] is False


def test_loop_malformed_stops():
    with Script(calls(tool_call("x", "", raw="不是 JSON"))) as s:
        r = tc.loop_one("m", Q, fake_search, BY_ID)
    assert r["end"] == "malformed" and len(s.seen) == 1, "格式不对就停, 不再请求"


def test_loop_cap_by_omitting_tools():
    """模型一直想搜: 搜满 MAX_ROUNDS 轮后, 最后一次请求不传 tools (不靠 tool_choice="none")."""
    replies = [calls(tool_call(f"c{i}", "吕蒙")) for i in range(tc.MAX_ROUNDS)] + [says("最后的回答")]
    with Script(*replies) as s:
        r = tc.loop_one("m", Q, fake_search, BY_ID)
    assert r["n_search"] == tc.MAX_ROUNDS and r["end"] == "answer"
    assert len(s.seen) == tc.MAX_ROUNDS + 1
    assert all(x["tools"] for x in s.seen[:-1]), "上限之前每次都要传 tools"
    assert not s.seen[-1]["tools"], "最后一次不传 tools"
    assert s.seen[-1]["kw"].get("tool_choice") is None, "不要用 tool_choice 控制上限"
    assert r["repeat_query"] is True


if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
