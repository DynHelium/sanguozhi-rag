"""Minimal tool-calling loop: call the model, run the tools it asks for, send results back as role "tool" messages,
repeat until it answers in plain text. The round limit is enforced in code: the last request carries no tools,
so the model can only answer in text (relying on tool_choice="none" instead failed in an earlier trial:
6 of 196 multi-round runs still returned tool calls). The model call and the tool runner are injected, so tests drive the loop with a fake model.

最小 agent 循环: 调模型 -> 执行模型要的工具 -> 结果作为 role "tool" 消息回传 -> 再调模型, 直到它直接回答.

    uv run python -m tests.test_loop      # 假模型驱动的测试, 不连网

不认识检索和工具: 怎么调模型 (complete), 怎么执行工具 (run_tool) 由调用方传进来, 测试时换成假的.
完整的 agent (system prompt, 工具实现, 查得太少时提示再查) 在它上面另写一层, 本模块只管消息往返. 轮数上限由代码保证: 最后一次请求不传 tools.
"""
from typing import Any, Callable

MAX_ROUNDS = 5  # 模型最多调用几轮工具 (一轮里可以有多个调用); 模型要自己从零检索 (代码不预先检索), 所以多给几轮
FINAL_TURN = ("工具调用次数已经用完, 不能再调用任何工具. 请只根据上面已经查到的资料, 用文字完整回答原问题; "
              "资料里没有的就说没有找到. 不要输出工具调用.")
LAST_TOOL_ROUND = "This is the last round of tool calling."  # 倒数第二次请求前追加: 这是最后一次能调工具


def agent_loop(messages: list[dict], tools: list[dict], complete: Callable[..., Any], run_tool: Callable[[str, str], str],
               max_rounds: int = MAX_ROUNDS) -> str:
    """反复请求模型, 执行它要的工具并把结果发回, 直到它直接作答; 返回回答文字. messages 原地追加.

    参数:
        messages    [{"role": "system", ...}, {"role": "user", "content": "曹操因为什么逃离洛阳？"}]
        tools       工具声明, OpenAI function calling 格式 (形状见 rag/agent.py 的 TOOLS), 原样交给 complete
        complete    complete(messages, tools=None) -> 模型的一条回复 message (有 .content 和 .tool_calls, 形状见下).
                    tools 为 None 表示这次不给工具
                    它不是 rag/llm.complete 本身 (那个返回整个响应): 调用方包一层, 返回 resp.choices[0].message.
        run_tool    run_tool(name, arguments) -> 交给模型看的文字. 两个参数就是 tc.function.name 和 tc.function.arguments
        max_rounds  最多几轮工具调用 (一轮 = 一次请求, 里面可以有多个调用); 可以是 0 (直接走最后一次)

    message
    ├─ content       "董卓表太祖为骁骑校尉……" 或 None (只调工具时)
    └─ tool_calls    None 或 [tc, ...];  tc.id "call_0" (少数模型给 None),  tc.function.name "search",
                                         tc.function.arguments '{"query": "浚儀"}' (JSON 字符串)

    规则 (r = 0, 1, ..., max_rounds, 最多 max_rounds + 1 次请求):
      - r < max_rounds: 带 tools 请求. 没有 tool_calls -> 返回 content or "".
        有 tool_calls -> messages 追加下面的 assistant 消息和每个调用一条 tool 消息, 接着下一次请求.
      - r == max_rounds - 1 (最后一次能调工具): 请求前追加 {"role": "developer", "content": LAST_TOOL_ROUND}, 提醒模型.
      - r == max_rounds (最后一次): 先追加 {"role": "developer", "content": FINAL_TURN}, 以 tools=None 请求 (不带工具声明);
        不管有没有 tool_calls, 返回 content or "".
    两条提示只在循环真的走到那一次时才追加; 模型提前作答就直接返回. 回答文字不写回 messages.
    max_rounds=0 时只有最后一次: 只追加 FINAL_TURN.
    这两条提示是代码的话, 不是用户的话, 所以用 developer 角色 (OpenAI 接口里给开发者指令用的角色), 都留在 messages 里.
    多轮对话 (如一局猜人物, 每轮都是一次 agent_loop) 接着用同一个 messages 再调 agent_loop 时, 要先删掉这些 developer 消息: 留着的 FINAL_TURN 会让模型在后面几轮
    以为再也不能调工具.
    不认 developer 角色的模型会忽略这两条提示 (不报错); 本项目用的 DeepSeek V4.1 Flash 认.

    一轮追加的消息 (示例里假设调用方提供了 search 和 era 两个工具; 本模块不定义工具. 同一条回复可以有多个调用,
    按列表顺序逐个执行):
        {"role": "assistant", "content": "",                                       content 为 None 时写 ""
         "tool_calls": [{"id": "a", "type": "function", "function": {"name": "search", "arguments": '{"query": "浚儀"}'}},
                        {"id": "b", "type": "function", "function": {"name": "era", "arguments": '{"era": "建安", "year": 7}'}}]}
        {"role": "tool", "tool_call_id": "a", "content": run_tool("search", '{"query": "浚儀"}')}
        {"role": "tool", "tool_call_id": "b", "content": run_tool("era", '{"era": "建安", "year": 7}')}
    tool 消息只有这三个键, 顺序和 tool_calls 一致. tc.id 为空时 (少数模型不给) 用 f"call_{r}_{i}" 代替, assistant 和 tool 两条消息里用同一个 id.
    """
    for r in range(max_rounds + 1):
        last = r == max_rounds
        if last:  # 只是不传 tools 时, 有的模型 (deepseek) 仍在文字里写工具调用标记, 所以明说不能再查
            messages.append({"role": "developer", "content": FINAL_TURN})
        if r + 1 == max_rounds:
            messages.append({"role": "developer", "content": LAST_TOOL_ROUND})
        msg = complete(messages, tools=None if last else tools)
        tcs = msg.tool_calls or []
        if last or not tcs:
            return msg.content or ""
        batch = [{"id": tc.id or f"call_{r}_{i}", "name": tc.function.name, "arguments": tc.function.arguments}
                 for i, tc in enumerate(tcs)]
        messages.append(
            {
                "role": "assistant", 
                "content": msg.content or "", 
                "tool_calls": [
                    {
                        "id": c["id"], 
                        "type": "function", 
                        "function": {"name": c["name"], "arguments": c["arguments"]}
                    } for c in batch
                ]
            }
        )
        for c in batch:
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": run_tool(c["name"], c["arguments"])})
    return ""  # 走不到: 最后一次一定 return
