"""RAG 效果评测: 同一模型分几组回答同一批题, 只差 "有没有资料, 资料怎么来", 看这套 RAG 给模型带来多少提升.

    uv run python -m eval.run_rag --group closed --models deepseek-v4.1-flash --effort low \\
        --questions eval/questions-rag50.jsonl eval/questions-rag50-en.jsonl --name rag-ds-closed
    LLM_BACKEND=anthropic uv run python -m eval.run_rag --group agentic --models claude-haiku-5-5 --effort medium \\
        --questions eval/questions-rag50.jsonl --name rag-hk-agentic       # Haiku 走 Claude 原生接口 (环境变量覆盖 .env)
    ... --ids i-01 h-03 ...                                                # 只跑这几题 (试跑)

各组 (--group), 一次只跑一组 (agent 的代码介入开关 agent.OFF 是全局的):
    closed    闭卷: 只有 P_BASE, 不给资料和工具, 调用一次
    static    静态 RAG: 用问题检索一次 (rag/library.Library.search, 前 SEARCH_K 组), 资料放进用户消息, 调用一次
    oracle    上限: 把 gold 段落所在的块直接放进用户消息 (和 static 同样的格式), 调用一次
    agentic   Agentic RAG: 给 search / read / who / era 四个工具 (TOOLS_PLAIN, 只写功能), system 多一句 TOOL_LINE;
              搜不搜, 搜几次由模型定, 没有任何代码介入
    harness   agentic + 搜法规则 (HARNESS) + 三种再查提示 (nudge, agg_nudge, nudge_notfound, 阈值见 rag/agent.py)
    production  生产配置, 和网页聊天 (app/chat.py) 一样: agent.system_for(模型) + 游戏路由说明, agent.TOOLS + 两个游戏工具,
              默认的代码介入 (agent.OFF, 含自查和引文修正), 原来的循环提示, 输出上限 rag/calls.MAX_TOKENS.
              不受上面 "控制住的" 各条约束 (prompt 不允许用自己的知识, 要求引原文), 只用来看生产环境比闭卷好多少;
              和网页聊天唯一的差别: 不接游戏的 handlers (问答题用不到; 模型若调用 start_game, 收到 "没有这个工具")

控制住的: 同一模型五组共用 P_BASE (两个模型也先用同一份); 都允许用自己的知识; 都不要求引原文和编号;
都不自查, 不做引文修正, 编号修正, 语言提示和语言检查; 都在截断或空回答时重答一次 (agent 组用 after_answer 里的 retry,
其他组见 one_call); 输出上限都是 MAX_TOKENS; 思考档由 --effort 给, 同一模型五组相同; 温度用接口默认.
agent 两组的循环提示换成中性的 NEUTRAL_FINAL (原来的 rag/loop.FINAL_TURN 要求 "资料里没有的就说没有找到", 和允许用自己的知识矛盾).
算作 RAG 本身的: 资料是原文 + 白话译文; agent 组看到的块带 "此段还有裴注" 的提示, who 给出传末的评, era 换算公元年.

结果 data/processed/eval_runs/answers/<name>.jsonl (几套题时按题目文件分, 见 eval.common.out_paths), 每行一个 (模型, 题),
字段同 eval/run_agent.py 的结果行, 另有 "group" 和 "retried" (重答过一次); 判卷用 eval/ai_judge.py.
续跑: 跳过已有且没出错的 (模型, 题); 跑完后同一 (模型, 题) 只留最后一行.
"""
import argparse
import copy
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.common import OUT_DIR, is_refusal, out_paths
from eval.run_agent import _sha, coverage
from eval.run_retrieval import load_jsonl
from rag import agent, calls, llm, loop
from rag.calls import PER_MODEL, make_complete, setup
from rag.library import READ_MAX, Seen, load_library

GROUPS = ("closed", "static", "oracle", "agentic", "harness", "production")
MAX_TOKENS = 16384  # 闭卷思考得多 (旧闭卷结果平均输出约 4800 token, 有题被 8192 截断), 五组一起提高
P_BASE = ("你回答三国 (东汉末到西晋初) 的历史问题, 以正史《三国志》(陈寿正文和裴松之注) 的记载为准.\n"
          "直接陈述事实; 不确定的具体事实 (年份, 人名, 数字) 说不确定, 不要编造.\n"
          "中文提问用简体中文回答, 英文提问用英文回答.")
TOOL_LINE = "你可以调用工具 search, read, who, era 查阅《三国志》原文 (含裴注) 和白话译文; 要不要查, 查什么, 查几次, 由你决定."
HARNESS = """查资料的做法:
a. 回答之前先用 search 查, 至少换两种说法搜两次. 资料按白话译文检索: 检索词用现代汉语, 人名写全名, 加事件关键词 (如 "曹操 去世 年龄"). 没找到就换说法再搜: 换成相关人物, 地名, 官职或人物的字. 问题没给人名的, 先用问题里的事件词搜, 不要先猜一个人名.
b. 问题里的人只给了称号 (先主, 宣王, 明帝), 或问 "有几个X" "X是谁" 时, 先用 who 查; 同名的人分别查.
c. 原文里 "帝" "公" "太祖" 这类不确定指谁, 或 "七年" "明年" 这种没写年号的年份, 用 read 往前读, 弄清楚再答. 公元年用 era 换算.
d. 搜到裴注却没看到它注的正文时, 对这块裴注 read, 事情的主干常在正文里.
e. 问 "有哪些/有几个" 的汇总题, 每个可能相关的人分别搜; 找到主要的那段正文后 read 一次, 裴注里常有正文没写的人和事. 问儿子时另外搜一次 "某某 养子". 问关系或经过的, 按阶段分别搜.
f. 说 "《三国志》没有记载" 或 "出自《三国演义》" 之前, 换几种说法搜过, 并读过相关人物传末陈寿的评 (who 的结果里写了在哪一段).
g. 英文提问时, 把人名, 地名, 官职译回中文再搜; 人名也可以直接用拼音 who."""
CONTEXT = "下面是从《三国志》(含裴注) 里找到的几段原文和白话译文:\n\n{context}\n\n问题: {q}"  # static 和 oracle 共用
NEUTRAL_FINAL = "不能再调用工具了, 请直接回答问题."
INTERVENTIONS = {"nudge", "nudge_notfound", "nudge_scope", "agg_nudge", "pinyin_who", "lang_note", "lang_fix", "review",
                 "quote_fix", "cite_fix"}  # rag/agent.OFF 认的全部名字
HARNESS_ON = {"nudge", "agg_nudge", "nudge_notfound"}  # harness 组打开的介入


def tools_plain() -> list[dict]:
    """agent.TOOLS 的副本, 说明只写功能 (输入什么, 返回什么); 去掉原来夹在说明里的搜法 ("汇总题对每个人分别搜",
    "读前后 1~2 块再作答", "先 read 这块注", "查不到就直接 search" 等), 那些放在 HARNESS 里."""
    tools = copy.deepcopy(agent.TOOLS)
    desc = {
        "search": "在《三国志》(含裴注) 里检索, 按白话译文匹配, 返回最相关的几块: 每块有原文 (文言繁体), 白话译文和块 id. "
                  "标 \"裴注\" 的块 (id 带 n, 如 46-0016n0) 是给正文 46-0016 作的注.",
        "read": f"读取某一块, 以及它在同一卷里前后相邻的块; 裴注块会附上它所注的正文. chunk_id 是块 id, 如 01-0053. "
                f"before / after 各最多 {READ_MAX}.",
        "era": "把年号的第几年换算成公元年, 如 era=建安, year=7 -> 202. 同名年号属于不同政权时 (如 建興 有蜀汉和孙吴的), "
               "可以用 regime 指定政权 (东汉, 曹魏, 蜀汉, 孙吴, 西晋 等); 不指定就返回各政权的结果.",
        "who": "查人: 给一个姓名, 称号, 字或拼音 (如 马忠, 先主, 曹孟德, Zhou Tai), 返回对得上的人: 姓名, 字, 传的首段 id, "
               "书里的其他称呼, 同名或拼音相同的人. 只收有传的人和少数常被只称称号的人.",
    }
    params = {"query": "检索词", "book": "可选: 只在这一书里搜", "part": "可选: 只搜陈寿正文或只搜裴注",
              "chunk_id": "块 id, 如 01-0053 或 01-0004n1", "name": "姓名, 称号, 字或拼音"}
    for t in tools:
        f = t["function"]
        f["description"] = desc[f["name"]]
        for p, d in f["parameters"]["properties"].items():
            if p in params:
                d["description"] = params[p]
    return tools


TOOLS_PLAIN = tools_plain()


def system_for(group: str) -> str:
    """各组的 system prompt: P_BASE 的最后一行 (作答语言) 始终放在最后, agent 组在它前面插入 TOOL_LINE (和 HARNESS)."""
    if group not in ("agentic", "harness"):
        return P_BASE
    head, last = P_BASE.rsplit("\n", 1)
    return "\n".join([head, TOOL_LINE] + ([HARNESS] if group == "harness" else []) + [last])


def oracle_chunks(chunks: list[dict], gold: list[str]) -> list[dict]:
    """sources 含某个 gold 源 id 的块, 按书里的顺序 (切块文件的顺序). gold 为空时为 []."""
    want = set(gold)
    return [c for c in chunks if want & set(c["sources"])]


def one_call(complete, model: str, system: str, user: str, extra: dict) -> tuple[str, int, bool]:
    """调用一次; 截断或空回答 (agent.bad_reply) 时原样再请求一次. -> (回答, 请求次数, 是否重答过)."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    r = complete(msgs, model=model, **extra)
    reply, finish = r.choices[0].message.content or "", getattr(r.choices[0], "finish_reason", None)
    if not agent.bad_reply(reply, finish):
        return agent.strip_markup(reply), 1, False
    r = complete(msgs, model=model, **extra)
    return agent.strip_markup(r.choices[0].message.content or ""), 2, True


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG 效果评测 (五组, 见模块说明)")
    ap.add_argument("--group", choices=GROUPS, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--questions", type=Path, nargs="+", required=True)
    ap.add_argument("--ids", nargs="+", help="只跑这几题")
    ap.add_argument("--name", required=True)
    ap.add_argument("--effort")
    args = ap.parse_args()
    prod = args.group == "production"
    if not prod:  # 生产配置保留 rag/calls 和 rag/agent 的默认值
        calls.MAX_TOKENS = MAX_TOKENS  # make_complete 每次请求时读它
        agent.OFF = INTERVENTIONS - (HARNESS_ON if args.group == "harness" else set())
        agent.FINAL_TURN = loop.FINAL_TURN = NEUTRAL_FINAL  # agent_loop 最后一轮和 after_answer 的 retry 都用它
    args.models = [setup(m) for m in args.models]
    if prod:
        from rag.skills import Router, load_skills
        router = Router({}, load_skills())  # 只用它的 agent_note 和 tools, 不开游戏
        system, tools = agent.system_for(args.models[0]) + router.agent_note(""), agent.TOOLS + router.tools()
    else:
        system, tools = system_for(args.group), TOOLS_PLAIN
    lib = load_library("trans")
    sems = {m: threading.Semaphore(PER_MODEL) for m in args.models}
    lock = threading.Lock()
    meta = {"group": args.group, "effort": args.effort, "index": "trans", "off": sorted(agent.OFF), "max_tokens": calls.MAX_TOKENS,
            "system_sha": _sha(system), "tools_sha": _sha(json.dumps(tools, ensure_ascii=False))}

    def one(job: tuple[str, dict, Path]) -> None:
        m, q, out = job
        gold, equiv, t = q.get("gold", []), q.get("equiv", {}), time.time()
        complete = make_complete(m, sems[m])
        extra = {"reasoning_effort": args.effort} if args.effort else {}
        rounds, retried, err = [], False, None
        try:
            if args.group in ("agentic", "harness", "production"):
                res = agent.run_agent(q["q"], lib, model=m, complete=complete, system=system, effort=args.effort,
                                      tools=tools, review=None if prod else False)
                reply, seen_chunks, rounds, n_calls = res["reply"], res["chunks"], res["rounds"], res["calls"]
                retried = any(r["tool"] == "retry" for r in rounds)
            else:
                got = {"closed": [], "static": lib.search(q["q"]) if args.group == "static" else [],
                       "oracle": oracle_chunks(lib.chunks, gold)}[args.group]
                seen = Seen()  # 不带 "此段还有裴注" 的提示: 这几组没有 read
                user = CONTEXT.format(context=seen.add(got) if got else "(没有找到)", q=q["q"]) if args.group != "closed" else q["q"]
                reply, n_calls, retried = one_call(complete, m, system, user, extra)
                seen_chunks = seen.chunks
                if args.group == "static":
                    rounds = [{"tool": "search", "args": {"query": q["q"]}, "chunks": [c["id"] for c in got], "error": None, "by": "code"}]
        except Exception as e:  # 接口报错: 记下来继续, 下次重跑
            reply, seen_chunks, n_calls, err = "", [], 0, f"{type(e).__name__}: {e}"[:500]
        row = {"model": m.split("/")[-1], "model_full": m, "qid": q["id"], "type": q.get("type", "?"), "q": q["q"], "gold": gold,
               "retrieved": [c["id"] for c in seen_chunks], **coverage(seen_chunks, gold, equiv),
               "reply": reply, "refused": is_refusal(reply), "retried": retried, "sec": round(time.time() - t, 1),
               "rounds": rounds, "calls": n_calls, "usage": complete.stats, **meta}
        if err:
            row["error"] = err
        with lock, open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_tools = sum(1 for r in rounds if r.get("by") == "model")
        print(f"{row['model']} {args.group} {q['id']} {row['sec']}s 工具 {n_tools} 次 {len(reply)} 字"
              + (" 重答" if retried else "") + (f" 错误 {err}" if err else ""), file=sys.stderr, flush=True)

    outs = out_paths(args.name, args.questions)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for qp, out in zip(args.questions, outs):
        done = {(r["model"], r["qid"]) for r in load_jsonl(out) if not r.get("error")} if out.exists() else set()
        qs = [q for q in load_jsonl(qp) if not args.ids or q["id"] in args.ids]
        jobs += [(m, q, out) for q in qs for m in args.models if (m.split("/")[-1], q["id"]) not in done]
    print(f"要跑 {len(jobs)} 个 (模型, 题)", file=sys.stderr)
    llm.client()  # 先在主线程建好客户端, 同 eval/run_agent.py
    with ThreadPoolExecutor(max(1, PER_MODEL * len(args.models))) as ex:
        list(ex.map(one, jobs))
    for out in outs:  # 同一 (模型, 题) 重跑过的, 只留最后一条
        if out.exists():
            keep = {(r["model"], r["qid"]): r for r in load_jsonl(out)}
            out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep.values()), encoding="utf-8")


if __name__ == "__main__":
    main()
