"""rag/calls.py 的测试 (make_complete 和启动检查 setup). 不连网, 不真的等待:

    uv run python -m tests.test_calls

假的 llm.complete 按脚本依次抛异常或返回响应; time.sleep 换成记录等待秒数.
"""
import threading
from types import SimpleNamespace

from rag import calls as R, config

MODEL = "deepseek/deepseek-v4.1-flash"


def ok(prompt=100, completion=20, cost=0.001, cached=0):
    u = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, cost=cost,
                        prompt_tokens_details=SimpleNamespace(cached_tokens=cached))
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="答", tool_calls=None))], usage=u)


class RateLimitError(Exception):
    pass


def patched(script):
    """换掉 llm.complete 和 time.sleep; 返回 (调用记录, 等待记录, 恢复函数)."""
    calls, sleeps = [], []
    orig_complete, orig_sleep = R.llm.complete, R.time.sleep

    def fake(messages, model=None, **kw):
        calls.append({"model": model, **kw})
        x = script.pop(0)
        if isinstance(x, Exception):
            raise x
        return x
    R.llm.complete, R.time.sleep = fake, sleeps.append

    def restore():
        R.llm.complete, R.time.sleep = orig_complete, orig_sleep
    return calls, sleeps, restore


class backend:
    """with backend("anthropic"): 临时改 config.LLM_BACKEND (和 key), 出来时恢复."""
    def __init__(self, name, api_key="sk-or-test", anthropic_key="sk-ant-test"):
        self.new = (name, api_key, anthropic_key)

    def __enter__(self):
        import os
        self.saved = (config.LLM_BACKEND, config.LLM_API_KEY, os.environ.get("ANTHROPIC_API_KEY"))
        config.LLM_BACKEND, config.LLM_API_KEY = self.new[0], self.new[1]
        if self.new[2] is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = self.new[2]

    def __exit__(self, *_):
        import os
        config.LLM_BACKEND, config.LLM_API_KEY, key = self.saved
        if key is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = key


def exits(fn, *a) -> str:
    """fn(*a) 应该 SystemExit; 返回提示文字."""
    try:
        fn(*a)
    except SystemExit as e:
        return str(e)
    raise AssertionError(f"{fn.__name__}{a} 应该退出")


def test_setup_checks_backend_model_and_key():
    """setup: openai 后端补全短名; 模型和后端对不上, 缺 key, 后端名不认识时退出并提示."""
    with backend("openai"):
        assert R.setup("deepseek-v4.1-flash") == MODEL
        assert "LLM_BACKEND=anthropic" in exits(R.setup, "claude-haiku-5-5")
    with backend("openai", api_key=""):
        assert "LLM_API_KEY" in exits(R.setup, "deepseek-v4.1-flash")
    with backend("anthropic"):
        assert R.setup("claude-haiku-5-5") == "claude-haiku-5-5"
        assert "只能用 claude-* 模型" in exits(R.setup, "deepseek-v4.1-flash")
    with backend("anthropic", anthropic_key=None):
        assert "ANTHROPIC_API_KEY" in exits(R.setup, "claude-haiku-5-5")
    with backend("ollama"):
        assert "openai 或 anthropic" in exits(R.setup, "qwen3.5:9b")
    with backend("openai"):
        assert "deepseek/deepseek-v4.1-flash" in exits(R.setup, "qwen3.5:4b"), "本地模型名 (没有 提供方/) 退出并提示"


def test_setup_model_name_forms():
    """anthropic/ 前缀: OpenRouter 后端照收 (经 OpenRouter 调 claude-* 模型), Claude 原生接口去掉前缀; 别的提供方前缀退出.
    大小写和首尾空白不影响; 不在价目表里的 claude-* 模型提醒花费记 0; .env 的 LLM_MODEL 为空时填上 (rag/llm.client 要它)."""
    import contextlib, io
    with backend("openai"):
        assert R.setup("anthropic/claude-haiku-5-5") == "anthropic/claude-haiku-5-5"
        assert R.setup("  DeepSeek-V4.1-Flash ") == MODEL
    with backend("anthropic"):
        assert R.setup("anthropic/claude-haiku-5-5") == "claude-haiku-5-5"
        assert R.setup("CLAUDE-HAIKU-5-5") == "claude-haiku-5-5"
        assert "只能用 claude-* 模型" in exits(R.setup, "deepseek/deepseek-v4.1-flash")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            assert R.setup("claude-sonnet-5-5") == "claude-sonnet-5-5"
        assert "花费会记成 0" in err.getvalue()
    saved = config.LLM_MODEL
    try:
        config.LLM_MODEL = ""
        with backend("openai"):
            R.setup("deepseek-v4.1-flash")
        assert config.LLM_MODEL == MODEL
    finally:
        config.LLM_MODEL = saved


def test_setup_warns_about_leftover_base_url():
    """.env 里还留着连本地模型的 LLM_BASE_URL 时提醒一句 (它已经不起作用), 不退出."""
    import contextlib, io, os
    saved = os.environ.get("LLM_BASE_URL")
    os.environ["LLM_BASE_URL"] = "http://localhost:11434/v1"
    err = io.StringIO()
    try:
        with backend("openai"), contextlib.redirect_stderr(err):
            assert R.setup("deepseek-v4.1-flash") == MODEL
    finally:
        os.environ.pop("LLM_BASE_URL") if saved is None else os.environ.__setitem__("LLM_BASE_URL", saved)
    assert "不再起作用" in err.getvalue()


def test_caller_timeout_overrides_default():
    """调用方传的 timeout / max_tokens 覆盖默认的 300 秒 / MAX_TOKENS (游戏的 REQUEST_TIMEOUT=60 靠这个)."""
    calls, _, restore = patched([ok(), ok()])
    try:
        c = R.make_complete(MODEL, threading.Semaphore(1))
        c([], model=MODEL)
        c([], model=MODEL, timeout=60, max_tokens=100)
        assert (calls[0]["timeout"], calls[0]["max_tokens"]) == (300, R.MAX_TOKENS)
        assert (calls[1]["timeout"], calls[1]["max_tokens"]) == (60, 100)
    finally:
        restore()


def test_spend_cap_429_is_not_retried():
    """花费到顶的 429 (Claude 原生接口的 enforced_spend_limit_reached) 重试没用, 直接抛出."""
    for msg in ["429 {'error_code': 'enforced_spend_limit_reached'}", "429 You have reached your API usage limits"]:
        calls, sleeps, restore = patched([RateLimitError(msg), ok()])
        try:
            c = R.make_complete(MODEL, threading.Semaphore(1))
            try:
                c([], model=MODEL)
                raise AssertionError("花费到顶时不重试")
            except RateLimitError:
                assert len(calls) == 1 and sleeps == []
        finally:
            restore()


def test_retries_429_with_growing_delay():
    calls, sleeps, restore = patched([RateLimitError("429 Too Many Requests"), RateLimitError("429"), ok()])
    try:
        c = R.make_complete(MODEL, threading.Semaphore(1))
        r = c([{"role": "user", "content": "q"}], model=MODEL)
        assert r.choices[0].message.content == "答" and len(calls) == 3
        assert len(sleeps) == 2 and sleeps[1] > sleeps[0], "退避时间要变长"
    finally:
        restore()


def test_other_errors_raise_immediately():
    calls, _, restore = patched([ValueError("bad request")])
    try:
        c = R.make_complete(MODEL, threading.Semaphore(1))
        try:
            c([], model=MODEL)
        except ValueError:
            assert len(calls) == 1
            return
        raise AssertionError("非 429 的错误不重试, 直接抛出")
    finally:
        restore()


def test_gives_up_after_retries():
    _, _, restore = patched([RateLimitError("429")] * 10)
    try:
        c = R.make_complete(MODEL, threading.Semaphore(1))
        try:
            c([], model=MODEL)
        except RateLimitError:
            return
        raise AssertionError("一直 429 时最终要抛出, 不能无限重试")
    finally:
        restore()


def test_openrouter_pins_provider_and_asks_usage():
    calls, _, restore = patched([ok(), ok()])
    try:
        with backend("openai"):
            R.make_complete(MODEL, threading.Semaphore(1))([], model=MODEL, tools=["t"])
        body = calls[0]["extra_body"]
        assert body["provider"]["only"] == R.PROVIDERS[MODEL] and body["usage"] == {"include": True}
        assert calls[0]["tools"] == ["t"], "其他参数原样透传"
        with backend("anthropic"):
            R.make_complete("claude-haiku-5-5", threading.Semaphore(1))([], model="claude-haiku-5-5")
        assert "extra_body" not in calls[1], "Claude 后端不加 (没有它 OpenRouter 不返回花费, 所以 openai 后端一定要加)"
    finally:
        restore()


def test_usage_is_summed_per_question():
    _, _, restore = patched([ok(100, 10, 0.001, 50), ok(200, 20, 0.002, 0)])
    try:
        c = R.make_complete(MODEL, threading.Semaphore(1))
        c([], model=MODEL)
        c([], model=MODEL)
        assert c.stats["prompt"] == 300 and c.stats["completion"] == 30 and c.stats["cached"] == 50
        assert abs(c.stats["cost"] - 0.003) < 1e-9
    finally:
        restore()


def test_usage_missing_fields_count_as_zero():
    """usage 里可能缺 cost / prompt_tokens_details, 也可能整个是 None: 都当 0."""
    bare = ok()
    bare.usage = SimpleNamespace(prompt_tokens=50, completion_tokens=5)
    none = ok()
    none.usage = None
    _, _, restore = patched([bare, none])
    try:
        c = R.make_complete(MODEL, threading.Semaphore(1))
        c([], model=MODEL)
        c([], model=MODEL)
        assert c.stats == {"prompt": 50, "completion": 5, "cached": 0, "cost": 0.0}
    finally:
        restore()


def test_question_budget():
    _, _, restore = patched([ok()])
    orig = R.QUESTION_BUDGET
    R.QUESTION_BUDGET = -1
    try:
        c = R.make_complete(MODEL, threading.Semaphore(1))
        try:
            c([], model=MODEL)
        except TimeoutError:
            return
        raise AssertionError("超过每题总时长就不再发请求")
    finally:
        R.QUESTION_BUDGET = orig
        restore()


if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
