"""Claude 原生接口后端: 函数签名和返回值形状同 rag.llm.complete (OpenAI 格式), 内部用 Anthropic SDK 调 Messages API.
.env 里设 LLM_BACKEND=anthropic 时, rag/llm.complete 把请求转到这里; 问答 agent, 游戏, 网页和评测脚本都不用改.

    uv sync --extra anthropic              # 装 Anthropic SDK (可选依赖)
    # .env: LLM_BACKEND=anthropic, LLM_MODEL=claude-haiku-5-5, LLM_REASONING_EFFORT=medium, ANTHROPIC_API_KEY=...
    uv run python -m eval.run_agent --models claude-haiku-5-5 --questions eval/questions-holdout.jsonl --name hk

为什么不用 Anthropic 的 OpenAI 兼容端点: 那里 reasoning_effort 被忽略 (思考档位定不下来), 不支持 prompt caching,
对话中途的 system / developer 消息会被挪到最前面. 原生接口三样都能做: output_config.effort, 自动缓存, 对话中途 system 消息.

格式转换 (OpenAI -> Anthropic):
    第一条 system 消息        -> 顶层 system
    之后的 developer / system -> 对话中途的 {"role": "system"} 消息 (留在原来的位置; Claude Haiku 5.5 等模型支持, 不用 beta)
    assistant (带 tool_calls) -> 这条回复当初的原始内容块 (thinking + text + tool_use), 按 tool_call id 从 _RAW 取回
    assistant (纯文字)        -> 文字和当初的回复逐字相同时取回原始内容块, 否则 (被代码改过, 如自查改写) 只发文字
    连续的 tool 消息          -> 一条 user 消息, 内含对应的 tool_result 块
    tools (OpenAI function)   -> {"name", "description", "input_schema"}

思考块必须原样回传: 开着思考调用工具时, 带 tool_result 的下一次请求要带上上一条回复的 thinking 块; 而且回传的思考块之前的内容
(system, tools, 更早的消息) 一个字都不能改, 否则 400. 所以:
    - 原始内容块存在进程内的 _RAW 里, 转换时取回, 不在 messages 里另存字段 (agent 循环只认 content / tool_calls);
    - 只回放这一轮 agent 循环 (最后一条 user 之后) 的原始块; 更早几轮的回复只发文字和工具调用;
    - 被改写过的回复只发文字, 不带它的思考块 (改了内容, 思考块就作废);
    - 万一接口仍说某个思考块的签名对不上 (400, "bound to a different conversation"), 去掉历史里的思考块重发一次;
    - tools 每次请求都发同一份: agent 循环最后一次请求不带 tools (rag/loop.py), 这里改成照发这段对话上一次的 tools
      (按对话里最后一个 tool_call id 查 _TOOLS, 几段对话同时进行也不会串) 并加 tool_choice {"type": "none"},
      禁止调用工具由 API 保证 (撤掉 tools 本身也算改了前缀).

返回值 (属性访问, 只有 agent 和 rag/calls.make_complete 用到的字段):
    resp.choices[0].message.content      回答文字 (多个 text 块拼接; 没有时为 None)
    resp.choices[0].message.tool_calls   [tc, ...] 或 None; tc.id, tc.function.name, tc.function.arguments (JSON 字符串)
    resp.choices[0].finish_reason        end_turn -> "stop", tool_use -> "tool_calls", max_tokens -> "length", refusal -> "content_filter"
    resp.usage.prompt_tokens / completion_tokens / prompt_tokens_details.cached_tokens / cost (美元, 按 PRICES 算)
"""
import json
import os
import sys
import threading
from types import SimpleNamespace
from typing import Any

PRICES = {"claude-haiku-5-5": (0.10, 0.50)}  # 模型 -> (输入, 输出) 美元 / 百万 token, prompt 不超过 10 万 token 时的价目
CACHE_READ, CACHE_WRITE = 0.1, 1.25  # 缓存读 / 写 (5 分钟) 相对输入价的倍数
FINISH = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length", "refusal": "content_filter", "stop_sequence": "stop"}

_RAW: dict[str, list[dict]] = {}   # "tool:<tool_use id>" / "text:<回复全文>" -> 当初的原始内容块
_TOOLS: dict[str, list[dict]] = {}  # tool_use id -> 发出它的那次请求的工具声明 (最后一次不带 tools 时照发这一份)
RAW_MAX = 5000  # _RAW / _TOOLS 最多留多少条 (先进先出): 网页长期运行时不让它一直涨
_LOCK = threading.Lock()
_client = None


def client() -> Any:
    """第一次调用时建 Anthropic 客户端, key 取环境变量 ANTHROPIC_API_KEY (SDK 默认)."""
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(max_retries=4)
    return _client


def to_tools(tools: list[dict]) -> list[dict]:
    """OpenAI function calling 的工具声明 -> Anthropic 的 {"name", "description", "input_schema"}."""
    return [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
             "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}}} for t in tools]


def to_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """OpenAI 格式的对话 -> (顶层 system, Anthropic 的 messages). 规则见模块 docstring.

    相邻的几条 system / developer 合成一条 (接口不许两条 system 消息相连). 原始内容块 (含思考块) 只回放最后一条 user 之后的
    assistant 消息, 即这一轮 agent 循环里的: 更早的回复只发文字和工具调用, 网页截断历史, 游戏删掉提示后前缀变了,
    回放旧思考块会被判作别的对话的. 内容为空又没有工具调用的 assistant 消息跳过 (接口不收空的 text 块)."""
    system, out = "", []
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
    for i, m in enumerate(messages):
        role, content = m["role"], m.get("content") or ""
        if role in ("system", "developer"):
            if i == 0 and role == "system":
                system = content
            elif out and out[-1]["role"] == "system":
                out[-1]["content"] += "\n\n" + content
            else:
                out.append({"role": "system", "content": content})
        elif role == "user":
            out.append({"role": "user", "content": content})
        elif role == "assistant":
            tcs = m.get("tool_calls") or []
            raw = None
            if i > last_user:
                raw = _RAW.get(f"tool:{tcs[0]['id']}") if tcs else _RAW.get(f"text:{content}")
            if raw is not None:
                out.append({"role": "assistant", "content": raw})
            else:
                blocks = ([{"type": "text", "text": content}] if content else []) + [
                    {"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"],
                     "input": json.loads(tc["function"]["arguments"] or "{}")} for tc in tcs]
                if blocks:
                    out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": content}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                    and out[-1]["content"] and out[-1]["content"][0].get("type") == "tool_result":
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return system, out


def _usage(u: Any, model: str) -> SimpleNamespace:
    """Anthropic 的 usage -> OpenAI 形状, 并按 PRICES 算花费 (输入 token 数含缓存读写)."""
    read = getattr(u, "cache_read_input_tokens", 0) or 0
    write = getattr(u, "cache_creation_input_tokens", 0) or 0
    fresh, out = u.input_tokens or 0, u.output_tokens or 0
    pin, pout = PRICES.get(model, (0.0, 0.0))
    cost = (fresh * pin + read * pin * CACHE_READ + write * pin * CACHE_WRITE + out * pout) / 1e6
    return SimpleNamespace(prompt_tokens=fresh + read + write, completion_tokens=out, total_tokens=fresh + read + write + out,
                           prompt_tokens_details=SimpleNamespace(cached_tokens=read), cost=cost)


def complete(messages: list[dict[str, Any]], model: str | None = None, tools: list[dict] | None = None,
             reasoning_effort: str | None = None, max_tokens: int = 8192, timeout: float = 300, **_: Any) -> Any:
    """发一次 Messages API 请求, 返回 OpenAI 形状的响应 (见模块 docstring). 其他 kwargs (OpenRouter 的 extra_body 等) 忽略.

    tools 为 None 而对话里已经有工具调用时 (agent 循环的最后一次), 照发上一次的工具声明并加 tool_choice none.
    max_tokens 含思考的 token."""
    model = model or os.environ.get("LLM_MODEL", "claude-haiku-5-5")
    system, msgs = to_messages(messages)
    kw: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": msgs, "timeout": timeout,
                          "cache_control": {"type": "ephemeral"}}
    if system:
        kw["system"] = system
    if reasoning_effort:
        kw["output_config"] = {"effort": reasoning_effort}
    if tools:
        kw["tools"] = to_tools(tools)
    else:
        ids = [tc["id"] for m in messages if m.get("role") == "assistant" for tc in (m.get("tool_calls") or [])]
        if ids and ids[-1] in _TOOLS:  # 查不到 (被淘汰, 或对话来自别的进程) 就不带: 发空的 tools 接口会报错
            kw["tools"] = _TOOLS[ids[-1]]
            kw["tool_choice"] = {"type": "none"}
    try:
        r = client().messages.create(**kw)
    except Exception as e:  # 回传的思考块被判作别的对话的 (签名对不上): 去掉历史里的思考块重发一次, 只是少了之前的推理
        if "signature" not in str(e) or "thinking" not in str(e):
            raise
        print(f"[claude_llm] 思考块签名被拒, 去掉思考块重发: {str(e)[:120]}", file=sys.stderr)
        kw["messages"] = [{**m, "content": [b for b in m["content"] if b.get("type") not in ("thinking", "redacted_thinking")]}
                          if m["role"] == "assistant" and isinstance(m["content"], list) else m for m in kw["messages"]]
        r = client().messages.create(**kw)
    raw = [b.to_dict() for b in r.content]
    text = "".join(b.text for b in r.content if b.type == "text") or None
    tcs = [SimpleNamespace(id=b.id, type="function",
                           function=SimpleNamespace(name=b.name, arguments=json.dumps(b.input, ensure_ascii=False)))
           for b in r.content if b.type == "tool_use"]
    with _LOCK:
        if tcs:
            _RAW[f"tool:{tcs[0].id}"] = raw
            for tc in tcs:
                _TOOLS[tc.id] = kw.get("tools", [])
        elif text is not None:
            _RAW[f"text:{text}"] = raw
        for d in (_RAW, _TOOLS):
            while len(d) > RAW_MAX:
                d.pop(next(iter(d)))
    msg = SimpleNamespace(content=text, tool_calls=tcs or None, role="assistant")
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=FINISH.get(r.stop_reason, r.stop_reason))],
                           usage=_usage(r.usage, model), model=r.model)
