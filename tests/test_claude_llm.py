"""rag/claude_llm.py 的测试: 用假的 Anthropic 客户端, 不连网, 不需要 key, 也不需要装 anthropic 包.

    uv run python -m tests.test_claude_llm
"""
from types import SimpleNamespace

from rag import claude_llm, config, llm


class Block(SimpleNamespace):
    """假的内容块: 有 type 等属性, to_dict() 返回原样的 dict (SDK 的块也有 to_dict)."""
    def to_dict(self) -> dict:
        return dict(vars(self))


class FakeClient:
    """messages.create(**kw) 依次返回给定的回复, 并记下每次的参数."""
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return self.replies.pop(0)


def reply(*blocks, stop="end_turn", usage=None):
    u = usage or SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(content=list(blocks), stop_reason=stop, usage=u, model="claude-haiku-5-5")


TOOLS = [{"type": "function", "function": {"name": "search", "description": "检索", "parameters": {"type": "object"}}}]


def use(fake: FakeClient) -> None:
    claude_llm._client = fake
    claude_llm._RAW.clear()
    claude_llm._TOOLS.clear()


def test_messages_tools_and_raw_replay():
    """system 进顶层, developer 变成对话中途的 system 消息, 连续的 tool 结果合成一条 user;
    带 tool_calls 的 assistant 发回当初的原始内容块 (含 thinking); effort 进 output_config; 返回 OpenAI 形状."""
    think = Block(type="thinking", thinking="", signature="sig")
    fake = FakeClient(reply(think, Block(type="tool_use", id="t1", name="search", input={"query": "浚儀"}), stop="tool_use"),
                      reply(Block(type="text", text="答案")))
    use(fake)
    msgs = [{"role": "system", "content": "你是老师"}, {"role": "user", "content": "问题"}, {"role": "developer", "content": "提醒"}]
    r = claude_llm.complete(msgs, model="claude-haiku-5-5", tools=TOOLS, reasoning_effort="low")
    first = fake.calls[0]
    assert first["system"] == "你是老师" and first["messages"][1] == {"role": "system", "content": "提醒"}
    assert first["output_config"] == {"effort": "low"} and first["tools"][0]["name"] == "search"
    tc = r.choices[0].message.tool_calls[0]
    assert r.choices[0].finish_reason == "tool_calls" and tc.function.arguments == '{"query": "浚儀"}'
    msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function",
              "function": {"name": "search", "arguments": tc.function.arguments}}]},
             {"role": "tool", "tool_call_id": "t1", "content": "资料"}]
    r2 = claude_llm.complete(msgs, model="claude-haiku-5-5", tools=TOOLS)
    sent = fake.calls[1]["messages"]
    assert sent[2]["content"][0] == think.to_dict(), "思考块原样发回"
    assert sent[3] == {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "资料"}]}
    assert r2.choices[0].message.content == "答案" and r2.choices[0].finish_reason == "stop"
    assert abs(r2.usage.cost - (100 * 0.10 + 20 * 0.50) / 1e6) < 1e-12


def test_last_round_reuses_this_conversations_tools():
    """最后一次不带 tools 时, 照发这段对话里最后一个工具调用当时的 tools, 并加 tool_choice none; 两段对话不串."""
    other = [{"type": "function", "function": {"name": "start_game", "description": "", "parameters": {"type": "object"}}}]
    fake = FakeClient(reply(Block(type="tool_use", id="a1", name="search", input={}), stop="tool_use"),
                      reply(Block(type="tool_use", id="b1", name="start_game", input={}), stop="tool_use"),
                      reply(Block(type="text", text="完")))
    use(fake)
    claude_llm.complete([{"role": "user", "content": "q1"}], tools=TOOLS)
    claude_llm.complete([{"role": "user", "content": "q2"}], tools=other)
    conv = [{"role": "user", "content": "q1"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "a1", "content": "x"}]
    claude_llm.complete(conv)
    last = fake.calls[-1]
    assert last["tool_choice"] == {"type": "none"} and [t["name"] for t in last["tools"]] == ["search"]


def test_llm_complete_dispatches_on_backend():
    """LLM_BACKEND=anthropic 时 rag.llm.complete 转给 claude_llm.complete, 并带上 LLM_REASONING_EFFORT 的默认值."""
    fake = FakeClient(reply(Block(type="text", text="好")))
    use(fake)
    saved = (config.LLM_BACKEND, config.LLM_MODEL, config.LLM_REASONING_EFFORT)
    config.LLM_BACKEND, config.LLM_MODEL, config.LLM_REASONING_EFFORT = "anthropic", "claude-haiku-5-5", "low"
    try:
        assert llm.chat([{"role": "user", "content": "你好"}]) == "好"
        assert fake.calls[0]["model"] == "claude-haiku-5-5" and fake.calls[0]["output_config"] == {"effort": "low"}
    finally:
        config.LLM_BACKEND, config.LLM_MODEL, config.LLM_REASONING_EFFORT = saved


def test_retry_without_thinking_when_signature_rejected():
    """接口说思考块签名对不上时, 去掉历史里的 thinking 块重发一次."""
    class Picky(FakeClient):
        def _create(self, **kw):
            self.calls.append(kw)
            if any(isinstance(m["content"], list) and any(b.get("type") == "thinking" for b in m["content"]) for m in kw["messages"]):
                raise RuntimeError("Invalid `signature` in `thinking` block")
            return self.replies.pop(0)
    fake = Picky(reply(Block(type="text", text="好")))
    use(fake)
    claude_llm._RAW["tool:t1"] = [{"type": "thinking", "thinking": "", "signature": "s"},
                                  {"type": "tool_use", "id": "t1", "name": "search", "input": {}}]
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "x"}]
    assert claude_llm.complete(msgs, tools=TOOLS).choices[0].message.content == "好" and len(fake.calls) == 2


def test_adjacent_system_merged_old_thinking_dropped_empty_skipped():
    """相邻的 developer 合成一条; 上一轮 (最后一条 user 之前) 的回复不回放思考块; 空的 assistant 消息跳过;
    最后一次不带 tools 而这次对话的工具声明查不到时, 不发 tools 也不发 tool_choice."""
    fake = FakeClient(reply(Block(type="text", text="好")))
    use(fake)
    claude_llm._RAW["text:旧答"] = [{"type": "thinking", "thinking": "", "signature": "s"}, {"type": "text", "text": "旧答"}]
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "q1"}, {"role": "assistant", "content": "旧答"},
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": ""},
            {"role": "developer", "content": "再查"}, {"role": "developer", "content": "最后一轮"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "zz", "type": "function", "function": {"name": "search", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "zz", "content": "x"}]
    claude_llm.complete(msgs)
    sent = fake.calls[0]
    assert sent["messages"][1] == {"role": "assistant", "content": [{"type": "text", "text": "旧答"}]}
    assert sent["messages"][3] == {"role": "system", "content": "再查\n\n最后一轮"}, sent["messages"]
    assert [m["role"] for m in sent["messages"]] == ["user", "assistant", "user", "system", "assistant", "user"]
    assert "tools" not in sent and "tool_choice" not in sent


if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
