"""经 OpenRouter 调用大模型时的一层包装: 选服务商, 限流重试, 每题超时, token 和花费记账.
评测脚本和网页聊天共用; 真正发请求的是 rag/llm.complete.
"""
import os
import random
import threading
import time
from typing import Any, Callable

from rag import config, llm

OPENROUTER_URL = "https://openrouter.ai/api/v1"
PROVIDERS = {"deepseek/deepseek-v4.1-flash": config.LLM_PROVIDERS}  # 模型全名 -> 只用这几家服务商 (按顺序优先; 空列表 = 不限定)
PER_MODEL = 4  # 同一个模型同时在途的请求数上限: 调用方建 threading.Semaphore(PER_MODEL) 传给 make_complete 的 sem
MAX_TOKENS = 8192  # 每次请求的输出上限: 不设时 OpenRouter 按模型的最大输出预留额度, 余额低时会被拒 (402)
QUESTION_BUDGET = 1200  # 每题总时长上限 (秒), 防服务商卡住; 正常一题几十秒到几分钟


def use_openrouter() -> None:
    """把 rag.llm 的接口切到 OpenRouter (改 config.LLM_BASE_URL / LLM_API_KEY, 下一次建客户端时生效).
    key 取环境变量 (.env) 的 OPENROUTER_API_KEY, 没有就用 LLM_API_KEY (.env.example 默认就填 OpenRouter 的 key);
    两个都没有时退出并提示."""
    key = os.environ.get("OPENROUTER_API_KEY") or config.LLM_API_KEY
    if not key:
        raise SystemExit("没有 OpenRouter 的 key: 在 .env 里填 LLM_API_KEY (或 OPENROUTER_API_KEY)")
    config.LLM_BASE_URL, config.LLM_API_KEY = OPENROUTER_URL, key


def full_name(m: str) -> str:
    """短名 (deepseek-v4.1-flash) 补成 OpenRouter 的模型全名 (deepseek/deepseek-v4.1-flash); 不认识的原样返回."""
    return next((k for k in PROVIDERS if k.split("/")[-1] == m), m)


def make_complete(model: str, openrouter: bool, sem: threading.Semaphore) -> Callable[..., Any]:
    """返回包了一层的 complete(messages, model=None, **kw) -> ChatCompletion, 签名同 rag.llm.complete,
    作为 agent 循环发请求用的 complete. 每题 (网页里每次提问) 新建一个: 计时和记账按题.

        model       模型全名, 用来查 PROVIDERS (请求里的模型名是调用时传的 model=)
        openrouter  走 OpenRouter 时为 True: 请求加 extra_body = {"usage": {"include": True}, "provider": {...}},
                    让 OpenRouter 返回花费, 并只用 PROVIDERS[model] 里的服务商 (按顺序, 第一家失败时换下一家);
                    model 不在 PROVIDERS 里或列表为空时不加 "provider", 由 OpenRouter 自选
        sem         同一模型共用的信号量, 限制同时在途的请求数

    每次发请求前检查: 自创建起已超过 QUESTION_BUDGET 秒就抛 TimeoutError (在途的请求不中断, 一题实际耗时可略超);
    单次请求另有 300 秒超时, 输出上限 MAX_TOKENS (调用方传了 max_tokens 时以调用方为准). 被限流 (错误信息含 "429", 或异常类型名含 "RateLimit") 时等 10, 20, 40, 80, 160 秒
    (各加 0~5 秒随机) 再试, 最多试 6 次; 其他错误直接抛出. 等待时不占 sem. 其他参数 (tools, reasoning_effort) 原样透传.

    返回的函数带 .stats, 累加每次响应的 usage (缺的字段当 0; 本地接口没有 cost):
        {"prompt": 41510, "completion": 1261, "cached": 17536, "cost": 0.001}
         输入 token        输出 token           命中缓存的输入 token  OpenRouter 返回的花费 (美元)
    """
    extra = {}
    if openrouter:
        body = {"usage": {"include": True}}
        if PROVIDERS.get(model):  # 空列表 = 不限定服务商
            body["provider"] = {"order": PROVIDERS[model], "only": PROVIDERS[model], "allow_fallbacks": True}
        extra = {"extra_body": body}

    start = time.time()
    stats = {"prompt": 0, "completion": 0, "cached": 0, "cost": 0.0}

    def tally(resp: Any) -> Any:
        u = getattr(resp, "usage", None)
        if u is not None:
            stats["prompt"] += getattr(u, "prompt_tokens", 0) or 0
            stats["completion"] += getattr(u, "completion_tokens", 0) or 0
            stats["cached"] += getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            stats["cost"] += float(getattr(u, "cost", 0) or 0)
        return resp

    def complete(messages: list[dict[str, Any]], model: str | None = None, **kw: Any) -> Any:
        delay = 10
        for attempt in range(6):
            if time.time() - start > QUESTION_BUDGET:
                raise TimeoutError(f"这题超过 {QUESTION_BUDGET} 秒")
            try:
                with sem:
                    return tally(llm.complete(messages, model=model, timeout=300, **{"max_tokens": MAX_TOKENS, **extra, **kw}))
            except Exception as e:
                if ("429" not in str(e) and "RateLimit" not in type(e).__name__) or attempt == 5:
                    raise
                time.sleep(delay + random.random() * 5)
                delay *= 2
    complete.stats = stats
    return complete
