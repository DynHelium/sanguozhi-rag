"""rag/llm.py 的测试 (client / chat / complete). 不依赖 pytest, 不连网:

    uv run python -m tests.test_llm

测 chat / complete 时把 llm.client 换成返回假客户端的函数, 记录每次 chat.completions.create 收到的参数,
返回预先写好的响应. 测 client() 本身时先清掉它的缓存 (cache_clear). 只检查传了什么参数, 返回什么.
"""
from types import SimpleNamespace

from rag import config, llm


class FakeCompletions:
    def __init__(self, content="好", tool_calls=None):
        self.calls, self.content, self.tool_calls = [], content, tool_calls

    def create(self, **kwargs):
        self.calls.append(kwargs)
        msg = SimpleNamespace(content=self.content, tool_calls=self.tool_calls, role="assistant")
        usage = SimpleNamespace(prompt_tokens=12, completion_tokens=3)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage)


class RaisingCompletions:
    """create 一被调用就抛错, 并记下被调了几次 (检查 complete 不吞异常, 不自己重试)."""
    def __init__(self):
        self.n = 0

    def create(self, **kwargs):
        self.n += 1
        raise RuntimeError("Error code: 429 - rate limited")


class FakeClient:
    def __init__(self, **kw):
        self.chat = SimpleNamespace(completions=FakeCompletions(**kw))


_REAL_CLIENT = llm.client


def _reset_client_cache():
    """清掉 client() 的 functools.cache, 下次调用时按当前 config 重建."""
    _REAL_CLIENT.cache_clear()


class _Env:
    """临时替换 config; client 给了假客户端时把 llm.client 换成返回它的函数, 为 None 时用真的 client() 并清缓存.
    退出时全部还原."""

    def __init__(self, model="fake-model", effort="", client=None):
        self.model, self.effort, self.client = model, effort, client

    def __enter__(self):
        self.saved = config.LLM_MODEL, config.LLM_REASONING_EFFORT, llm.client
        config.LLM_MODEL, config.LLM_REASONING_EFFORT = self.model, self.effort
        if self.client is None:
            llm.client = _REAL_CLIENT
            _reset_client_cache()
        else:
            llm.client = lambda: self.client
        return self.client

    def __exit__(self, *exc):
        config.LLM_MODEL, config.LLM_REASONING_EFFORT, llm.client = self.saved
        _reset_client_cache()


MSGS = [{"role": "user", "content": "你好"}]


# ---------- client ----------

def test_client_requires_model():
    with _Env(model="", client=None):
        try:
            llm.client()
        except RuntimeError:
            return
        raise AssertionError("LLM_MODEL 为空时 client() 要抛 RuntimeError, 提示去配 .env")


def test_client_cached_and_configured():
    """同一进程只建一个客户端; base_url 和 api_key 取自 config (这里不发请求, 不会连网)."""
    with _Env(client=None):
        a, b = llm.client(), llm.client()
        assert a is b, "client() 要缓存, 不要每次新建"
        assert str(a.base_url).rstrip("/") == config.LLM_BASE_URL.rstrip("/")


def test_client_reads_config_at_first_call():
    """client() 在第一次调用时才读 config: 调用方可能在 import 之后改 config.LLM_BASE_URL."""
    saved = config.LLM_BASE_URL
    try:
        with _Env(client=None):
            config.LLM_BASE_URL = "http://localhost:11434/v1"
            assert str(llm.client().base_url).rstrip("/") == "http://localhost:11434/v1"
    finally:
        config.LLM_BASE_URL = saved


def test_client_empty_api_key_ok():
    """api_key 为空时也能建客户端: SDK 不接受空 key, 换成占位串."""
    saved = config.LLM_API_KEY
    try:
        with _Env(client=None):
            config.LLM_API_KEY = ""
            llm.client()
    finally:
        config.LLM_API_KEY = saved


# ---------- chat ----------

def test_chat_returns_text_and_passes_model():
    fake = FakeClient(content="陈寿")
    with _Env(client=fake):
        assert llm.chat(MSGS) == "陈寿"
        sent = fake.chat.completions.calls[-1]
        assert sent["model"] == "fake-model" and sent["messages"] == MSGS


def test_chat_none_content_is_empty_string():
    """只返回 tool_calls 时 content 是 None, chat 要返回 "" 而不是 None."""
    with _Env(client=FakeClient(content=None)):
        assert llm.chat(MSGS) == ""


def test_chat_passes_kwargs():
    fake = FakeClient()
    with _Env(client=fake):
        llm.chat(MSGS, temperature=0, max_tokens=5)
        sent = fake.chat.completions.calls[-1]
        assert sent["temperature"] == 0 and sent["max_tokens"] == 5


# ---------- reasoning_effort ----------

def test_reasoning_effort_from_config():
    fake = FakeClient()
    with _Env(effort="none", client=fake):
        llm.chat(MSGS)
        assert fake.chat.completions.calls[-1].get("reasoning_effort") == "none"


def test_reasoning_effort_not_sent_when_unset():
    """.env 没设时不传这个参数: 不支持它的接口会报错."""
    fake = FakeClient()
    with _Env(effort="", client=fake):
        llm.chat(MSGS)
        assert "reasoning_effort" not in fake.chat.completions.calls[-1]


def test_reasoning_effort_caller_wins():
    fake = FakeClient()
    with _Env(effort="none", client=fake):
        llm.chat(MSGS, reasoning_effort="low")
        assert fake.chat.completions.calls[-1]["reasoning_effort"] == "low"


# ---------- complete ----------

def test_complete_returns_full_response_with_tool_calls():
    tc = [SimpleNamespace(id="c1", function=SimpleNamespace(name="search", arguments='{"query": "吕蒙"}'))]
    fake = FakeClient(content=None, tool_calls=tc)
    tools = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
    with _Env(client=fake):
        resp = llm.complete(MSGS, tools=tools)
        assert resp.choices[0].message.tool_calls == tc, "complete 返回完整响应, 调用方要看 tool_calls"
        assert fake.chat.completions.calls[-1]["tools"] == tools


def test_complete_returns_whole_response():
    """返回整个响应, 不是 message: 调用方还要看 finish_reason 和 usage."""
    with _Env(client=FakeClient()):
        resp = llm.complete(MSGS)
        assert resp.choices[0].finish_reason == "stop"
        assert resp.usage.prompt_tokens == 12


def test_complete_explicit_none_model():
    """显式传 model=None 也要回退到 config.LLM_MODEL."""
    fake = FakeClient()
    with _Env(client=fake):
        llm.complete(MSGS, model=None)
        assert fake.chat.completions.calls[-1]["model"] == "fake-model"


def test_complete_reasoning_effort():
    """reasoning_effort 的规则在 complete 里 (调用方可能直接调 complete, 不经过 chat)."""
    fake = FakeClient()
    with _Env(effort="none", client=fake):
        llm.complete(MSGS)
        assert fake.chat.completions.calls[-1].get("reasoning_effort") == "none"
    with _Env(effort="", client=fake):
        llm.complete(MSGS)
        assert "reasoning_effort" not in fake.chat.completions.calls[-1]


def test_complete_passes_timeout_and_extra_body():
    fake = FakeClient()
    body = {"usage": {"include": True}}
    with _Env(client=fake):
        llm.complete(MSGS, timeout=300, extra_body=body)
        sent = fake.chat.completions.calls[-1]
        assert sent["timeout"] == 300 and sent["extra_body"] == body


def test_complete_raises_without_retry():
    """异常原样抛出, 只请求一次: 要不要重试由调用方按异常信息 (如含 "429") 决定."""
    rc = RaisingCompletions()
    with _Env(client=SimpleNamespace(chat=SimpleNamespace(completions=rc))):
        try:
            llm.complete(MSGS)
        except RuntimeError as e:
            assert "429" in str(e)
        else:
            raise AssertionError("complete 不能吞掉异常")
    assert rc.n == 1, f"complete 不要自己重试 (请求了 {rc.n} 次)"


def test_complete_model_override():
    fake = FakeClient()
    with _Env(client=fake):
        llm.complete(MSGS, model="deepseek/deepseek-v4.1-flash")
        assert fake.chat.completions.calls[-1]["model"] == "deepseek/deepseek-v4.1-flash"
        llm.complete(MSGS)
        assert fake.chat.completions.calls[-1]["model"] == "fake-model", "不给 model 就用 config.LLM_MODEL"


if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
