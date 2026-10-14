"""rag/openrouter.make_complete 的测试. 不连网, 不真的等待:

    uv run python -m tests.test_make_complete

假的 llm.complete 按脚本依次抛异常或返回响应; time.sleep 换成记录等待秒数.
"""
import threading
from types import SimpleNamespace

from rag import openrouter as R

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


def test_retries_429_with_growing_delay():
    calls, sleeps, restore = patched([RateLimitError("429 Too Many Requests"), RateLimitError("429"), ok()])
    try:
        c = R.make_complete(MODEL, True, threading.Semaphore(1))
        r = c([{"role": "user", "content": "q"}], model=MODEL)
        assert r.choices[0].message.content == "答" and len(calls) == 3
        assert len(sleeps) == 2 and sleeps[1] > sleeps[0], "退避时间要变长"
    finally:
        restore()


def test_other_errors_raise_immediately():
    calls, _, restore = patched([ValueError("bad request")])
    try:
        c = R.make_complete(MODEL, True, threading.Semaphore(1))
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
        c = R.make_complete(MODEL, True, threading.Semaphore(1))
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
        R.make_complete(MODEL, True, threading.Semaphore(1))([], model=MODEL, tools=["t"])
        body = calls[0]["extra_body"]
        assert body["provider"]["only"] == R.PROVIDERS[MODEL] and body["usage"] == {"include": True}
        assert calls[0]["tools"] == ["t"], "其他参数原样透传"
        R.make_complete(MODEL, False, threading.Semaphore(1))([], model=MODEL)
        assert "extra_body" not in calls[1], "不走 OpenRouter 时不加"
    finally:
        restore()


def test_usage_is_summed_per_question():
    _, _, restore = patched([ok(100, 10, 0.001, 50), ok(200, 20, 0.002, 0)])
    try:
        c = R.make_complete(MODEL, True, threading.Semaphore(1))
        c([], model=MODEL)
        c([], model=MODEL)
        assert c.stats["prompt"] == 300 and c.stats["completion"] == 30 and c.stats["cached"] == 50
        assert abs(c.stats["cost"] - 0.003) < 1e-9
    finally:
        restore()


def test_usage_missing_fields_count_as_zero():
    """本地接口的 usage 没有 cost / prompt_tokens_details, 也可能整个是 None: 都当 0."""
    bare = ok()
    bare.usage = SimpleNamespace(prompt_tokens=50, completion_tokens=5)
    none = ok()
    none.usage = None
    _, _, restore = patched([bare, none])
    try:
        c = R.make_complete("qwen3.5:9b", False, threading.Semaphore(1))
        c([], model="qwen3.5:9b")
        c([], model="qwen3.5:9b")
        assert c.stats == {"prompt": 50, "completion": 5, "cached": 0, "cost": 0.0}
    finally:
        restore()


def test_question_budget():
    _, _, restore = patched([ok()])
    orig = R.QUESTION_BUDGET
    R.QUESTION_BUDGET = -1
    try:
        c = R.make_complete(MODEL, True, threading.Semaphore(1))
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
