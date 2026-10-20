"""附录: 早期的 function calling 实验, 看模型能不能稳定地用 OpenAI tools 接口 (只有一个 search 工具).

不代表当前配置: 现在的 agent (rag/agent.py) 用 DeepSeek V4.1 Flash 或 Claude Haiku, 有四个工具. 保留这个脚本是为了
记录 "轮数上限靠最后一次不传 tools" 这个决定的依据. 接口走 rag/llm (OpenAI 兼容接口, 经 OpenRouter).

    uv run python -m eval.tool_calling --models deepseek/deepseek-v4.1-flash   # 跑给定的模型 (没有实跑过, 只有假模型测试)
    uv run python -m eval.tool_calling --summary <结果文件>   # 只打印汇总
    uv run python -m tests.test_tool_loop                   # 假模型测试 check_call / loop_one

每个模型测三项, 写进结果文件 OUT (不进 git) 的 "single" "required" "loop" 三个列表, 实验配置在 "config":
    A. 单轮: eval/retrieval.jsonl 每题请求 REPS 次, 加几句闲聊 (OFF_TOPIC), 把回复分成 call / leaked / answer / empty (classify).
    B. tool_choice="required": 前 REQUIRED_N 题各一次, 看强制调用时格式还对不对.
    C. 多轮循环 (loop_one): 最多 MAX_ROUNDS 轮, 比较模型自己搜到的块和直接用原问题检索前 5 / 前 8 的 gold 召回.

早期实测的结论: 搜满上限后传 tool_choice="none", 仍有约 3% 返回 tool_calls, 所以轮数上限靠最后一次不传 tools.
"""
import argparse
import json
import re
import sys
import time
from typing import Any, Literal, NotRequired, TypedDict, cast

from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall

from rag import config, context, llm, retrieve
from rag.schema import ChunkDict, Question, Search


REPS = 3
TEMPERATURE = 0.7
REQUIRED_N = 10
MAX_ROUNDS = 3
TOOL_K = 5
OUT = config.DATA_DIR / "processed" / "eval_runs" / "tools.json"

# 只有一个 search 工具 (OpenAI tools 格式); 模型调用时返回 name "search", arguments '{"query": "吕蒙"}'.
SYSTEM = ("你是三国史问答助手。回答前先用 search 工具在《三国志(裴松之注)》原文里检索, "
          "可以换不同的检索词多搜几次; 只依据检索到的原文回答, 资料里没有就说没有。")
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "在《三国志(裴松之注)》中搜索原文，给出文言文的原文内容，包括对应的章节和Chunk编号。检索词用原文里可能出现的人名、地名、事件用词。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键字"}
                },
                "required": ["query"]
            }
        }
    }
]
OFF_TOPIC = ["你好", "1+1 等于几？", "用一句话介绍你自己。", "今天天气怎么样？", "把 hello 翻译成中文。", "谢谢你的帮助。"]

LEAK = re.compile(r'"query"\s*:|"name"\s*:\s*"search"|\bsearch\s*\(|<tool_call>|<\|tool', re.I)


def check_call(call: "ChatCompletionMessageToolCall") -> tuple[bool, list[str], str | None]:
    """检查模型返回的一个工具调用能不能执行, 返回 (valid, errors, query).

    valid 要求: id 非空, name 为 "search", arguments 是 JSON 对象且 "query" 是去掉空白后非空的字符串.
    errors 每个不满足的条件一条说明. query 是去掉首尾空白的检索词, 取不到或不是字符串时为 None.
        check_call(<id="call_0", name="search", arguments='{"query": "吕蒙"}'>)  -> (True, [], "吕蒙")
        check_call(<id="call_0", name="search", arguments='{query: 吕蒙'>)       -> (False, ["Failed to load query as a JSON, ..."], None)
    格式问题都作为 errors 返回而不抛异常, 因为坏的调用是要统计的实验结果.
    """
    id, name = call.id, call.function.name
    errors, query = [], None
    if not id:
        errors.append(f"Invalid id as {id}")
    if name != "search":
        errors.append(f"Name is not 'search', got {name}")
    try:
        query_json = json.loads(call.function.arguments or "")  # arguments 为 None 时按空串解析, 记为 JSON 错误
        if not isinstance(query_json, dict):
            errors.append("Arguments value is not a valid dict.")
        elif "query" not in query_json:
            errors.append("'query' is not in the arguments dict.")
        else:
            query = query_json["query"]
            if isinstance(query, str):
                query = query.strip()
                if len(query) == 0:
                    errors.append("Query is an empty string.")
            else:
                errors.append("Query is not a valid string")
                query = None
    except json.JSONDecodeError as e:
        errors.append(f"Failed to load query as a JSON, {e.msg}")    
    return not errors, errors, query


class Reply(TypedDict):
    """classify 的结果: 模型的一条回复是什么.

        调用:      {"kind": "call", "valid": True, "errors": [], "query": "吕蒙", "n_calls": 2}
        文字回答:  {"kind": "answer", "valid": False, "errors": [], "query": None, "n_calls": 0, "content": "是吕蒙..."}
    """
    kind: Literal["call", "leaked", "answer", "empty"]  # 见 classify
    valid: bool          # 每个调用都通过 check_call; 不是 "call" 时为 False
    errors: list[str]    # 所有调用的 check_call 错误; 不是 "call" 时为 []
    query: str | None    # 第一个调用的检索词 (check_call 的结果); 不是 "call" 时为 None
    n_calls: int         # 调用的个数; 不是 "call" 时为 0
    content: NotRequired[str]  # 不是 "call" 时的回复文字 (最多 300 字, 只用于记录)


def classify(msg: ChatCompletionMessage) -> Reply:
    """模型的一条回复 -> Reply. 有 tool_calls 就是 "call"; 否则看文字: 空 -> "empty",
    像工具调用 (LEAK 正则, 多半是对话模板没接好; 启发式) -> "leaked", 其他 -> "answer"."""
    calls = msg.tool_calls or []
    if calls:
        # tool_calls 的类型是 function / custom 两种调用的联合; 这里只声明了 function 工具, 按 function 调用检查
        checks = [check_call(cast("ChatCompletionMessageToolCall", c)) for c in calls]
        return {"kind": "call", "valid": all(v for v, _, _ in checks), "errors": [e for _, es, _ in checks for e in es],
                "query": checks[0][2], "n_calls": len(calls)}
    text = (msg.content or "").strip()
    if not text:
        kind = "empty"
    elif LEAK.search(text):
        kind = "leaked"
    else:
        kind = "answer"
    return {"kind": kind, "valid": False, "errors": [], "query": None, "n_calls": 0, "content": text[:300]}


def call(model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = TOOLS,
         **kw: Any) -> tuple[ChatCompletionMessage, str, float]:
    """调用一次模型, 返回 (message, finish_reason, 秒). tools=None 时不带工具声明 (循环的最后一轮用它引导模型作答;
    模型仍返回调用时, loop_one 以 "cap" 结束, 不再执行)."""
    t = time.time()
    if tools:
        kw["tools"] = tools
    resp = llm.complete(messages, model=model, temperature=TEMPERATURE, **kw)
    return resp.choices[0].message, resp.choices[0].finish_reason, time.time() - t


def gold_hit(chunk_ids: list[str], gold: list[str], by_id: dict[str, ChunkDict]) -> bool:
    """这些块的 sources (所属段落 id) 里有没有任一个 gold 段落."""
    return any(s in gold for cid in chunk_ids for s in by_id[cid]["sources"])


def run_single(model: str, questions: list[Question]) -> list[dict[str, Any]]:
    """A. 单轮: 每题和每句闲聊各请求 REPS 次, 每次一行 (题号, finish_reason, 秒数加上 classify 的结果)."""
    rows = []
    prompts = [(q["id"], q["q"]) for q in questions] + [(f"off-{i}", t) for i, t in enumerate(OFF_TOPIC)]
    for rep in range(REPS):
        for qid, text in prompts:
            msg, finish, sec = call(model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}])
            rows.append({"model": model, "qid": qid, "q": text, "rep": rep, "finish": finish, "sec": round(sec, 2),
                         **classify(msg)})
    return rows


def run_required(model: str, questions: list[Question]) -> list[dict[str, Any]]:
    """B. 前 REQUIRED_N 题各请求一次 tool_choice="required", 每题一行; 接口不接受时记为 kind "error"."""
    rows = []
    for q in questions[:REQUIRED_N]:
        try:
            msg, finish, sec = call(model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q["q"]}],
                                    tool_choice="required")
            rows.append({"model": model, "qid": q["id"], "finish": finish, "sec": round(sec, 2), **classify(msg)})
        except Exception as e:  # 接口不接受 required
            rows.append({"model": model, "qid": q["id"], "kind": "error", "valid": False,
                         "errors": [f"{type(e).__name__}: {e}"], "query": None})
    return rows


def run_loop(model: str, questions: list[Question], search: Search,
             by_id: dict[str, ChunkDict]) -> list[dict[str, Any]]:
    """C. 每题跑一次 loop_one, 每题一行; 接口报错时记为 end "error" 并照样算基线召回, 不中断实验."""
    rows = []
    for q in questions:
        try:
            rows.append(loop_one(model, q, search, by_id))
        except Exception as e:  # 接口报错 (如服务端拒绝某种消息格式): 记下来, 不中断整个实验
            rows.append({"model": model, "qid": q["id"], "q": q["q"], "gold": q["gold"], "end": "error",
                         "error": f"{type(e).__name__}: {e}"[:300], "rounds": [], "n_search": 0, "repeat_query": False,
                         "first_is_question": False, "answer": "", "sec": 0.0, "recall": False,
                         "base5": gold_hit([c["id"] for c in search(q["q"], 5)], q["gold"], by_id),
                         "base8": gold_hit([c["id"] for c in search(q["q"], 8)], q["gold"], by_id)})
        r = rows[-1]
        print(f"  {model:12} {q['id']} {r['end']:9} 搜{r['n_search']}次 召回{'✓' if r['recall'] else '·'}", flush=True)
    return rows


def loop_row(model: str, q: Question, end: str, rounds: list[dict[str, Any]], answer: str | None, sec: float,
             search: Search, by_id: dict[str, ChunkDict]) -> dict[str, Any]:
    """loop_one 的一行结果 (结果文件的 "loop"): 循环的结局加上统计字段.

        {"model": "deepseek-v4.1-flash", "qid": "r-10", "q": "白衣渡江是谁的事迹？", "gold": ["54-0032", "36-0004n0"],
         "end": "answer",                    answer / no_call / malformed / cap / leaked / empty
         "rounds": [{"queries": ["吕蒙"], "chunks": [...]}, ...],
         "n_search": 1, "repeat_query": False, "first_is_question": False,
         "answer": "...", "sec": 3.2,
         "recall": True,                     各轮检索到的块里有没有 gold (只看 sources)
         "base5": True, "base8": True}       对照: 直接用原问题检索前 5 / 前 8 有没有 gold
    """
    retrieved = [cid for r in rounds for cid in r["chunks"]]
    all_q = [x for r in rounds for x in r["queries"]]
    return {"model": model, "qid": q["id"], "q": q["q"], "gold": q["gold"], "end": end, "rounds": rounds,
            "n_search": len(all_q), "repeat_query": len(all_q) != len(set(all_q)),
            "first_is_question": bool(all_q) and all_q[0] == q["q"],
            "answer": answer or "", "sec": round(sec, 1),
            "recall": gold_hit(retrieved, q["gold"], by_id),
            "base5": gold_hit([c["id"] for c in search(q["q"], 5)], q["gold"], by_id),
            "base8": gold_hit([c["id"] for c in search(q["q"], 8)], q["gold"], by_id)}


def loop_one(model: str, q: Question, search: Search, by_id: dict[str, ChunkDict]) -> dict[str, Any]:
    """一道题的多轮工具循环: 模型搜 -> 代码检索并把结果作为 role "tool" 消息发回 -> 模型再搜或回答. 返回 loop_row 的一行.

    q 是一道检索题, 如 {"id": "r-10", "q": "白衣渡江是谁的事迹？", "gold": ["54-0032", "36-0004n0"]};
    search(query, k) 返回前 k 个 chunk; by_id 用来按 sources 算召回 (不看 parent 和 equiv, 比 eval/metrics.py 严).
    每轮执行这次回复里的全部调用 (各取 TOOL_K 块); 搜满 MAX_ROUNDS 轮后最后一次请求不传 tools.
    结局 end: 文字回答为 "answer" (一次没搜为 "no_call"); 调用格式不对为 "malformed"; 不传 tools 仍调用为 "cap";
    其余为 classify 的 "leaked" / "empty".
    """
    t0, answer = time.time(), ""
    message = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": q['q']},
    ]
    rounds = list[dict]()
    reply, finished_reason, seconds = call(model, messages=message)
    stop = False
    while not stop:
        reply_dict = classify(reply)
        match reply_dict["kind"]:
            case "call":
                if len(rounds) < MAX_ROUNDS:
                    if reply_dict["valid"]:
                        message.append({
                            "role": "assistant", "content": reply.content if reply.content else "",
                            "tool_calls": [
                                {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in reply.tool_calls
                            ]
                        })
                        queries = []
                        chunk_ids = []
                        for tc in reply.tool_calls:
                            query = json.loads(tc.function.arguments)["query"]
                            chunks = search(query, TOOL_K)
                            chunk_ids.extend([chunk['id'] for chunk in chunks])
                            queries.append(query)
                            message.append(
                                {"role": "tool", "tool_call_id": tc.id, "content": context.format_context(chunks)}
                            )
                        rounds.append({"queries": queries, "chunks": chunk_ids})
                        reply, _, _ = call(model, message, tools=TOOLS if len(rounds) < MAX_ROUNDS else None)
                    else:
                        rounds.append({"queries": [], "chunks": [], "errors": reply_dict["errors"]})
                        stop = True
                        end = "malformed"
                else:
                    end = "cap"
                    stop = True
            case "answer":
                answer = reply.content
                end = "answer" if rounds else "no_call"
                stop = True
            case _:
                end = reply_dict["kind"]
                stop = True
    return loop_row(model, q, end, rounds, answer, time.time() - t0, search, by_id)




def kn(k: int, n: int) -> str:
    """汇总表里的 "k/n"."""
    return f"{k}/{n}"


def summarize(data: dict) -> None:
    models = data["config"]["models"]
    print(f"\nA. 单轮 (temperature {data['config']['temperature']}, 每题 {data['config']['reps']} 次)")
    print(f"{'模型':12} {'调用':>9} {'格式正确/调用':>13} {'泄漏成文字':>9} {'直接回答':>9} {'空':>6} {'闲聊误调用':>9} {'中位秒':>6}")
    for m in models:
        rs = [r for r in data["single"] if r["model"] == m and not r["qid"].startswith("off-")]
        off = [r for r in data["single"] if r["model"] == m and r["qid"].startswith("off-")]
        n = len(rs)
        calls = [r for r in rs if r["kind"] == "call"]
        secs = sorted(r["sec"] for r in rs)
        print(f"{m:12} {kn(len(calls), n):>9} {kn(sum(r['valid'] for r in calls), len(calls)):>13}"
              f" {kn(sum(r['kind'] == 'leaked' for r in rs), n):>9} {kn(sum(r['kind'] == 'answer' for r in rs), n):>9}"
              f" {kn(sum(r['kind'] == 'empty' for r in rs), n):>6} {kn(sum(r['kind'] == 'call' for r in off), len(off)):>9}"
              f" {secs[len(secs) // 2]:>6.1f}")
    print(f"\nB. tool_choice=\"required\" (前 {data['config']['required_n']} 题各一次)")
    for m in models:
        rs = [r for r in data["required"] if r["model"] == m]
        err = [r for r in rs if r["kind"] == "error"]
        print(f"{m:12} 调用 {kn(sum(r['kind'] == 'call' for r in rs), len(rs))}, 格式正确 {kn(sum(r['valid'] for r in rs), len(rs))}"
              + (f", 接口报错 {len(err)}: {err[0]['errors'][0][:80]}" if err else ""))
    print(f"\nC. 多轮循环 (最多 {data['config']['max_rounds']} 轮搜索, 每次取前 {data['config']['tool_k']} 块)")
    print(f"{'模型':12} {'正常回答':>8} {'没调用':>6} {'格式错':>6} {'泄漏':>5} {'空':>5} {'超限':>5} {'报错':>5} {'平均搜索':>8} "
          f"{'重复检索词':>9} {'召回':>7} {'基线前5':>7} {'基线前8':>7} {'中位秒':>6}")
    for m in models:
        rs = [r for r in data["loop"] if r["model"] == m]
        scored = [r for r in rs if r["gold"]]
        n = len(rs)
        cnt = lambda e: kn(sum(r["end"] == e for r in rs), n)
        secs = sorted(r["sec"] for r in rs)
        print(f"{m:12} {cnt('answer'):>8} {cnt('no_call'):>6} {cnt('malformed'):>6} {cnt('leaked'):>5} {cnt('empty'):>5}"
              f" {cnt('cap'):>5} {cnt('error'):>5} {sum(r['n_search'] for r in rs) / n:>8.1f} {kn(sum(r['repeat_query'] for r in rs), n):>9}"
              f" {kn(sum(r['recall'] for r in scored), len(scored)):>7} {kn(sum(r['base5'] for r in scored), len(scored)):>7}"
              f" {kn(sum(r['base8'] for r in scored), len(scored)):>7} {secs[len(secs) // 2]:>6.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="function calling 实测")
    ap.add_argument("--summary", help="只打印已有结果文件的汇总")
    ap.add_argument("--models", nargs="*", default=[], help="模型名 (OpenRouter 全名), 实测时必填")
    ap.add_argument("--limit", type=int, help="只跑前 N 题 (冒烟用), 结果写到 tools-smoke.json")
    args = ap.parse_args()
    if args.summary:
        summarize(json.loads(open(args.summary, encoding="utf-8").read()))
        return
    if not args.models:
        raise SystemExit("实测要用 --models 给出模型名")
    with open(config.ROOT / "eval" / "retrieval.jsonl", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    out = OUT
    if args.limit:
        questions, out = questions[:args.limit], OUT.with_name("tools-smoke.json")
    chunks = retrieve.load_chunks()
    by_id = {c["id"]: c for c in chunks}
    search = retrieve.make_search(chunks)
    search("预热", 1)  # 先加载 bge-m3, 不计入模型用时
    data = {"config": {"models": args.models, "system": SYSTEM, "tools": TOOLS, "temperature": TEMPERATURE,
                       "reps": REPS, "required_n": REQUIRED_N, "max_rounds": MAX_ROUNDS, "tool_k": TOOL_K,
                       "reasoning_effort": config.LLM_REASONING_EFFORT, "base_url": config.LLM_BASE_URL,
                       "date": time.strftime("%Y-%m-%d")},
            "single": [], "required": [], "loop": []}
    for m in args.models:
        print(f"== {m}", flush=True)
        llm.complete([{"role": "user", "content": "你好"}], model=m, max_tokens=5)  # 预热, 不计时
        data["single"] += run_single(m, questions)
        print(f"  单轮完成 {len(data['single'])}", flush=True)
        data["required"] += run_required(m, questions)
        data["loop"] += run_loop(m, questions, search, by_id)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")  # 每个模型跑完存一次
    print("->", out)
    summarize(data)


if __name__ == "__main__":
    sys.exit(main())
