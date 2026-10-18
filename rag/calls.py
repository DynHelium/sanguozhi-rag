"""发模型请求的一层包装: 启动检查 (key, 模型名), 限流重试, 每题超时, 输出上限, token 和花费记账.
问答 agent, 游戏, 网页聊天和评测脚本共用; 真正发请求的是 rag/llm.complete.

后端由 .env 的 LLM_BACKEND 决定 (rag/config.py), 入口脚本不按后端分支:
    openai     (默认) DeepSeek V4.1 Flash 经 OpenRouter (OpenAI 兼容接口, key 是 LLM_API_KEY).
               每次请求加 extra_body: 让 OpenRouter 在 usage 里返回花费, 并只用 PROVIDERS 里给这个模型指定的服务商.
    anthropic  Claude 原生接口 (rag/claude_llm.py, key 是 ANTHROPIC_API_KEY), 花费由它按价目表算.
暂时只支持这两种: 本地模型 (把接口地址改到本机 Ollama 等) 暂不支持. 要恢复, 把 rag/config.LLM_BASE_URL 改回从 .env 读,
再让 setup 接受第三种后端 (请求不加 OpenRouter 的 extra_body).

入口脚本的用法:
    model = calls.setup(args.model)                 # 检查 key, 短名补全, 模型和后端对得上; 不对时退出并提示
    sem = threading.Semaphore(calls.PER_MODEL)       # 同一模型共用
    complete = calls.make_complete(model, sem)       # 每题新建一个 (计时和记账按题), 交给 run_agent 等
"""
import os
import random
import sys
import threading
import time
from typing import Any, Callable

from rag import config, llm

PROVIDERS = {"deepseek/deepseek-v4.1-flash": config.LLM_PROVIDERS}  # 模型全名 -> 只用这几家服务商 (按顺序优先; 空列表 = 不限定)
PER_MODEL = 4  # 同一个模型同时在途的请求数上限: 调用方建 threading.Semaphore(PER_MODEL) 传给 make_complete 的 sem
MAX_TOKENS = 8192  # 每次请求的输出上限: 不设时 OpenRouter 按模型的最大输出预留额度, 余额低时会被拒 (402)
QUESTION_BUDGET = 1200  # 每题总时长上限 (秒), 防服务商卡住; 正常一题几十秒到几分钟
# 429 里带这些字样的是花费到顶 (到下个月或调高上限才恢复), 重试没用. 字样取自 Claude 原生接口的文档 (Rate limits > Reaching your
# spend cap): error_code "enforced_spend_limit_reached", 提示 "You have reached your API usage limits ..."
NO_RETRY = ("enforced_spend_limit_reached", "usage limits")


def is_claude(model: str) -> bool:
    """是不是 claude-* 模型 (claude-haiku-5-5 之类, 不分大小写; 带 "anthropic/" 前缀也算)."""
    return model.split("/")[-1].lower().startswith("claude-")


def full_name(m: str) -> str:
    """短名 (deepseek-v4.1-flash, 不分大小写) 补成 OpenRouter 的模型全名 (deepseek/deepseek-v4.1-flash); 不认识的原样返回."""
    return next((k for k in PROVIDERS if k.split("/")[-1] == m.lower()), m)


def setup(model: str | None) -> str:
    """入口脚本启动时调用一次 (多个模型就每个调一次): 返回请求里用的模型名; 有问题时 SystemExit 并说明怎么改 .env.

    - 模型: model 为 None 或空串时用 .env 的 LLM_MODEL (去掉首尾空白), 结果为空时退出. .env 的 LLM_MODEL 为空时填成这个模型
      (rag/llm.client 要求它非空). LLM_BACKEND 不是 openai / anthropic 时退出.
    - anthropic 后端: 模型要是 claude-* ("anthropic/" 前缀去掉, 别的 "提供方/" 退出); 要有 ANTHROPIC_API_KEY.
      模型不在 rag/claude_llm.PRICES 里时提醒: 花费会记成 0, 网页的每日上限不起作用.
    - openai 后端 (OpenRouter): 没有 "提供方/" 的 claude-* 退出 (要么设 LLM_BACKEND=anthropic, 要么写 OpenRouter 的全名
      anthropic/claude-...); 要有 LLM_API_KEY (或 OPENROUTER_API_KEY); 短名补成全名 (full_name), 补完仍没有 "提供方/"
      (如本地的 qwen3.5:4b) 时退出. key 在这里查, 免得第一次请求才报 401.
    - .env 里还有原来连本地模型的 LLM_BASE_URL 时打印一句提醒 (已经不起作用)."""
    if os.environ.get("LLM_BASE_URL", "").rstrip("/") not in ("", config.LLM_BASE_URL):
        print(f"提醒: .env 里的 LLM_BASE_URL={os.environ['LLM_BASE_URL']} 不再起作用 (暂时不支持本地模型), 可以删掉", file=sys.stderr)
    m = (model or config.LLM_MODEL or "").strip()
    if not m:
        raise SystemExit("没有模型: 用 --model, 或在 .env 里填 LLM_MODEL")
    if config.LLM_BACKEND == "anthropic":
        if "/" in m:
            if not m.lower().startswith("anthropic/"):
                raise SystemExit(f"LLM_BACKEND=anthropic 时只能用 claude-* 模型, 收到 {m}; "
                                 "用 DeepSeek 要把 .env 的 LLM_BACKEND 改回 openai")
            m = m.split("/", 1)[1]  # Claude 原生接口的模型名不带 "anthropic/"
        m = m.lower()
        if not is_claude(m):
            raise SystemExit(f"LLM_BACKEND=anthropic 时只能用 claude-* 模型, 收到 {m}; "
                             "用 DeepSeek 要把 .env 的 LLM_BACKEND 改回 openai")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("没有 Claude 原生接口的 key: 在 .env 里填 ANTHROPIC_API_KEY")
        from rag import claude_llm  # 不 import anthropic 包 (那在第一次请求时才 import)
        if m not in claude_llm.PRICES:
            print(f"提醒: rag/claude_llm.PRICES 里没有 {m} 的价格: 花费会记成 0, 网页的每日上限 (--daily-budget) 不起作用",
                  file=sys.stderr)
    else:
        if config.LLM_BACKEND != "openai":
            raise SystemExit(f"LLM_BACKEND 只能是 openai 或 anthropic, 收到 {config.LLM_BACKEND}")
        if is_claude(m) and "/" not in m:
            raise SystemExit(f"{m} 是 claude-* 模型: 用 Claude 原生接口要在 .env 里设 LLM_BACKEND=anthropic 和 ANTHROPIC_API_KEY "
                             "(经 OpenRouter 调就写全名 anthropic/...)")
        if not config.LLM_API_KEY:
            raise SystemExit("没有 OpenRouter 的 key: 在 .env 里填 LLM_API_KEY (或 OPENROUTER_API_KEY)")
        m = full_name(m)
        if "/" not in m:
            raise SystemExit(f"OpenRouter 的模型名要带提供方, 形如 deepseek/deepseek-v4.1-flash, 收到 {m} "
                             "(本地模型暂时不支持); 改 --model 或 .env 的 LLM_MODEL")
    if not config.LLM_MODEL:
        config.LLM_MODEL = m
    return m


def make_complete(model: str, sem: threading.Semaphore) -> Callable[..., Any]:
    """返回包了一层的 complete(messages, model=None, **kw) -> ChatCompletion, 签名同 rag.llm.complete,
    作为 agent 循环发请求用的 complete. 每题 (网页里每次提问) 新建一个: 计时和记账按题.

        model  模型全名 (setup 的返回值), 用来查 PROVIDERS (请求里的模型名是调用时传的 model=)
        sem    同一模型共用的信号量, 限制同时在途的请求数

    openai 后端 (OpenRouter) 的请求加 extra_body = {"usage": {"include": True}, "provider": {...}}: 让 OpenRouter 返回花费
    (没有它 stats["cost"] 一直是 0, 网页的每日上限就不起作用), 并只用 PROVIDERS[model] 里的服务商 (按顺序, 第一家失败时换下一家);
    model 不在 PROVIDERS 里或列表为空时不加 "provider", 由 OpenRouter 自选. anthropic 后端不加 (Claude 原生接口用不上这些参数).

    每次发请求前检查: 自创建起已超过 QUESTION_BUDGET 秒就抛 TimeoutError (在途的请求不中断, 一题实际耗时可略超);
    单次请求另有超时 (默认 300 秒) 和输出上限 MAX_TOKENS, 调用方传了 timeout / max_tokens 时以调用方为准.
    被限流 (错误信息含 "429", 或异常类型名含 "RateLimit") 时等 10, 20, 40, 80, 160 秒 (各加 0~5 秒随机) 再试, 最多试 6 次;
    花费到顶的 429 (错误信息含 NO_RETRY 里的字样) 和其他错误直接抛出. 等待时不占 sem. 其他参数 (tools, reasoning_effort) 原样透传.
    Claude 原生接口的 SDK 自己还会先重试几次 (rag/claude_llm.client 的 max_retries, 按 retry-after 等), 用完才抛到这一层:
    所以普通 429 是两层重试; 花费到顶的 429 没有 retry-after, SDK 那几次很快就用完.

    返回的函数带 .stats, 累加每次响应的 usage (缺的字段当 0):
        {"prompt": 41510, "completion": 1261, "cached": 17536, "cost": 0.001}
         输入 token        输出 token           命中缓存的输入 token  花费 (美元; OpenRouter 返回的, 或 Claude 后端按价目表算的)
    """
    extra = {}
    if config.LLM_BACKEND != "anthropic":
        body: dict[str, Any] = {"usage": {"include": True}}
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
                    return tally(llm.complete(messages, model=model, **{"timeout": 300, "max_tokens": MAX_TOKENS, **extra, **kw}))
            except Exception as e:
                limited = "429" in str(e) or "RateLimit" in type(e).__name__
                if not limited or any(s in str(e) for s in NO_RETRY) or attempt == 5:
                    raise
                time.sleep(delay + random.random() * 5)
                delay *= 2
    complete.stats = stats  # type: ignore[attr-defined]
    return complete
