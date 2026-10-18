"""LLM 调用的入口. 默认走 OpenAI 兼容接口, 用 DeepSeek V4.1 Flash, 经 OpenRouter 调用 (配置见 .env.example);
LLM_BACKEND=anthropic 时 complete 转给 rag/claude_llm.py (Claude 原生接口, 如 Claude Haiku 5.5), 调用方式和返回值形状不变.

    uv run python -m rag.llm          # 发一句话测连通 (需要 .env)
    uv run python -m tests.test_llm   # 假客户端, 不连网

配置 (rag/config.py; 除 LLM_BASE_URL 外从 .env 读):
    LLM_BASE_URL          "https://openrouter.ai/api/v1" (固定, 暂时不支持本地模型)
    LLM_API_KEY           OpenRouter 的 key (也认 OPENROUTER_API_KEY)
    LLM_MODEL             "deepseek/deepseek-v4.1-flash"; 必填
    LLM_REASONING_EFFORT  "low"; 留空时不传这个参数
    LLM_BACKEND           "openai" (默认) 或 "anthropic"

三个函数的关系: chat 调 complete, complete 调 client().chat.completions.create.
    client()    建 OpenAI 客户端, 整个进程只建一次
    complete()  发一次请求, 返回完整响应 (用工具时要看 tool_calls, finish_reason, usage)
    chat()      只要文字时用: complete(...).choices[0].message.content

不在这里做: 重试, 429 退避, 并发控制, 记账; 这些由调用方按需包一层. SDK 抛的异常原样往外抛,
调用方靠异常信息判断要不要重试 (如字符串里有 "429").
"""
from typing import TYPE_CHECKING, cast, Any

from openai import OpenAI
import functools

from . import config

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

@functools.cache
def client() -> OpenAI:
    """返回进程里唯一的 OpenAI 客户端; 第一次调用时才建, 之后返回同一个 (functools.cache; client.cache_clear() 清掉后重建).

    第一次调用时才读 config 并建客户端: 调用方可能在 import 之后, 第一次请求之前改
    config.LLM_BASE_URL / LLM_API_KEY (例如测试换成假的接口地址), import 时就建好的客户端会静默连到原来的接口.

    config.LLM_MODEL 为空时抛 RuntimeError, 提示去配 .env. 客户端用 config.LLM_BASE_URL 和 config.LLM_API_KEY 建;
    api_key 为空时传占位串 "empty": SDK 不接受空 key; 正常启动时 rag/calls.setup 已经确认 key 填了, 这里只是让没有 key 时 (如测试) 也能建出客户端.
    """
    if not config.LLM_MODEL:
        raise RuntimeError("LLM_MODEL 未设置. LLM_MODEL is not set up.")
    return OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY or "empty")


def chat(messages: list[dict[str, Any]], **kwargs: Any) -> str:
    """发一次请求, 只返回回答的文字.

    messages 是 OpenAI 对话格式:
        [{"role": "system", "content": "你是三国志问答助手"},
         {"role": "user", "content": "用一句话介绍陈寿。"}]

    调 complete(messages, **kwargs) (kwargs 包括 model, 原样交给 complete), 返回
    resp.choices[0].message.content; content 是 None (模型只回了 tool_calls) 时返回 "".
    reasoning_effort 只在 complete 里处理, chat 不管.
    """
    return complete(messages, **kwargs).choices[0].message.content or ""


def complete(messages: list[dict[str, Any]], model: str | None = None, **kwargs: Any) -> "ChatCompletion":
    """发一次请求, 返回 SDK 的完整响应对象 (ChatCompletion), 不转 dict, 不只取 message.

    做的事就是:
        client().chat.completions.create(model=..., messages=messages, **kwargs)

    model: None 就用 config.LLM_MODEL. 不传和显式传 model=None 一样.

    reasoning_effort (OpenAI 标准参数, 控制模型作答前 "思考" 多少; 本项目用 "low"):
      - kwargs 里已经有 reasoning_effort -> 不动 (调用方优先)
      - 否则 config.LLM_REASONING_EFFORT 非空 -> 加上
      - 否则不传这个参数 (不支持它的接口会报 400)

    其余 kwargs 原样交给 create, llm.py 不解析它们. 例如:
        tools=[{"type": "function", "function": {"name": "search", "description": "...", "parameters": {...}}}]
        temperature=0, max_tokens=5, timeout=300,
        extra_body={"usage": {"include": True}, "provider": {...}}     (OpenRouter: 返回花费, 钉服务商)

    messages 里可能有带 tool_calls 的 assistant 消息和 role "tool" 的工具结果, 也原样发:
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_0", "type": "function", "function": {"name": "search", "arguments": "{\"query\": \"吕蒙\"}"}}]}
        {"role": "tool", "tool_call_id": "call_0", "content": "[1] 吕蒙传 ..."}

    返回值 (属性访问, 不是 dict), 调用方用到的字段:
        resp
        ├─ choices[0]
        │  ├─ finish_reason      "stop" | "tool_calls" | "length" (被 max_tokens 截断)
        │  └─ message
        │     ├─ content         str 或 None (只有 tool_calls 时是 None)
        │     └─ tool_calls      None 或 [tc, ...]
        │        └─ tc.id "call_0"; tc.function.name "search"; tc.function.arguments '{"query": "吕蒙"}' (JSON 字符串)
        └─ usage                 可能是 None
           ├─ prompt_tokens, completion_tokens, prompt_tokens_details.cached_tokens
           └─ cost               只有 OpenRouter (请求里带 usage.include) 才有

    异常不捕获, 不重试, 原样抛出.
    """
    if config.LLM_REASONING_EFFORT and "reasoning_effort" not in kwargs:
        kwargs["reasoning_effort"] = config.LLM_REASONING_EFFORT
    if config.LLM_BACKEND == "anthropic":  # Claude 原生接口: 同样的参数, 同样形状的返回值 (见 rag/claude_llm.py)
        from rag import claude_llm
        return claude_llm.complete(messages, model=model or config.LLM_MODEL, **kwargs)
    # messages 是普通 dict, SDK 要的是 ChatCompletionMessageParam (几种 TypedDict 的联合), 类型检查器认不出来;
    # kwargs 里可能有 stream, 检查器也推不出返回的是 ChatCompletion 而不是 Stream. 两处都用 cast 告诉它.
    resp = client().chat.completions.create(
        model=model or config.LLM_MODEL, messages=cast("list[ChatCompletionMessageParam]", messages), **kwargs)
    return cast("ChatCompletion", resp)


if __name__ == "__main__":
    print(chat([{"role": "user", "content": "用一句话介绍陈寿。"}]))
