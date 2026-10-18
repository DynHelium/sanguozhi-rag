"""agent 回答一题的时间花在哪: 跑几道题, 把每题的时间拆成 "模型请求" 和 "检索" 两部分, 每次模型请求单独记.

    uv run python -m eval.bench_latency --model deepseek-v4.1-flash --effort low -n 5
    uv run python -m eval.bench_latency --model deepseek-v4.1-flash --effort low --off review   # 关掉自查对比

配置和 rag/library.load_library 的默认一致: 白话索引, 别名扩展; 另用 rag/agent.py 的 SYSTEM / TOOLS, rag/calls.make_complete.
题目默认取 eval/questions-holdout.jsonl 的前 n 题 (--questions 换文件, --ids 指定题), 一题一题顺序跑
(不并发, 和一个人逐题提问时一样). 只打印, 不写文件.

每次模型请求在内存里记一条, 用来打印每题的拆分和最后的汇总:
    {"qid": "h-01", "i": 0, "sec": 6.3, "prompt": 9120, "completion": 210, "tools": True, "stage": "loop"}
    i: 这题的第几次请求; prompt / completion: 输入 / 输出 token 数; tools: 这次请求带没带工具声明
    stage: loop = agent 循环里的请求 (含提示再查); after = 作答之后 (自查, 重答, 改引文) 的请求
检索 (Library.search / read) 另外计时; 总时间减去模型请求和检索剩下的记为 "其他".
"""
import argparse
import json
import statistics
import threading
import time
from collections.abc import Callable
from typing import Any

from eval.run_agent import PROMPTS
from rag import calls
from rag import agent, config

QUESTIONS = config.ROOT / "eval" / "questions-holdout.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description="一题的时间花在哪")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--effort", default="low")
    ap.add_argument("-n", type=int, default=5, help="跑题目文件的前几题 (给了 --ids 时不用)")
    ap.add_argument("--off", nargs="*", default=None, help="关掉哪些代码介入 (同 eval.run_agent --off)")
    ap.add_argument("--review-mode", default=None, choices=["rewrite", "short", "layered", "patch"], help="自查的做法 (agent.REVIEW_MODE)")
    ap.add_argument("--review-effort", help="自查等作答之后几步的 reasoning_effort (agent.REVIEW_EFFORT); 不给就和作答相同")
    ap.add_argument("--questions", default=str(QUESTIONS), help="题目文件 (默认留出题)")
    ap.add_argument("--ids", nargs="+", help="只跑这些题 (不给就取前 n 题)")
    ap.add_argument("--prompt", help="system prompt 换成 eval/agent_prompts/<名字>.txt (仓库里不带这个目录, 要自己建, 放对照用的 prompt; 同 eval.run_agent --prompt); 不给就用 rag/agent.py 的 SYSTEM")
    args = ap.parse_args()
    if args.off is not None:
        agent.OFF = set(args.off) - {"none"}
    if args.review_mode:
        agent.REVIEW_MODE = args.review_mode
    agent.REVIEW_EFFORT = args.review_effort
    model = calls.setup(args.model)  # 后端由 .env 的 LLM_BACKEND 决定

    t = time.time()
    lib = agent.load_library()
    lib.search("曹操")  # 预热: 第一次检索要加载 bge-m3
    print(f"加载索引 + 模型预热: {time.time() - t:.1f} 秒")

    search_sec: list[float] = []
    for name in ("search", "read"):
        orig = getattr(lib, name)

        def timed(*a: Any, _orig: Callable[..., Any] = orig, **kw: Any) -> Any:
            t0 = time.time()
            try:
                return _orig(*a, **kw)
            finally:
                search_sec.append(time.time() - t0)
        setattr(lib, name, timed)

    qs = [json.loads(line) for line in open(args.questions, encoding="utf-8")]
    qs = [q for q in qs if q["id"] in args.ids] if args.ids else qs[: args.n]
    system = (PROMPTS / f"{args.prompt}.txt").read_text(encoding="utf-8").strip() if args.prompt else None
    reqs: list[dict] = []
    totals = []
    for q in qs:
        base = calls.make_complete(model, threading.Semaphore(1))
        stage = {"now": "loop"}
        n0, s0 = len(reqs), len(search_sec)

        def complete(messages: list[dict], model: str | None = None, _base: Callable[..., Any] = base,
                     _q: dict = q, **kw: Any) -> Any:
            t0 = time.time()
            resp = _base(messages, model=model, **kw)
            u = getattr(resp, "usage", None)
            reqs.append({"qid": _q["id"], "i": len([r for r in reqs if r["qid"] == _q["id"]]),
                         "sec": round(time.time() - t0, 1),
                         "prompt": getattr(u, "prompt_tokens", 0) or 0,
                         "completion": getattr(u, "completion_tokens", 0) or 0,
                         "tools": "tools" in kw, "stage": stage["now"]})
            return resp

        def on_event(kind: str, info: Any, _stage: dict[str, str] = stage) -> None:
            if kind == "stage" and info not in ("nudge", "nudge_notfound"):  # 提示再查仍在循环里
                _stage["now"] = "after"

        t0 = time.time()
        agent.run_agent(q["q"], lib, model=model, complete=complete, system=system, effort=args.effort, on_event=on_event)
        total = time.time() - t0
        totals.append(total)
        mine = reqs[n0:]
        llm = sum(r["sec"] for r in mine)
        srch = sum(search_sec[s0:])
        print(f"{q['id']}  总 {total:5.1f}s  模型请求 {len(mine)} 次 {llm:5.1f}s  检索 {len(search_sec) - s0} 次 {srch:4.1f}s  "
              f"其他 {total - llm - srch:4.1f}s  | 每次请求秒数 {[r['sec'] for r in mine]}")

    print("\n汇总")
    loop = [r for r in reqs if r["stage"] == "loop"]
    after = [r for r in reqs if r["stage"] == "after"]
    print(f"  每题总时间 中位 {statistics.median(totals):.1f}s, 平均 {statistics.mean(totals):.1f}s")
    for name, rs in (("agent 循环", loop), ("作答之后 (自查等)", after)):
        if rs:
            print(f"  {name}: {len(rs)} 次请求, 共 {sum(r['sec'] for r in rs):.1f}s, 每次中位 "
                  f"{statistics.median(r['sec'] for r in rs):.1f}s, 平均输出 {statistics.mean(r['completion'] for r in rs):.0f} tokens, "
                  f"平均输入 {statistics.mean(r['prompt'] for r in rs):.0f} tokens")
    print(f"  检索: {len(search_sec)} 次, 共 {sum(search_sec):.1f}s, 每次中位 {statistics.median(search_sec):.2f}s")


if __name__ == "__main__":
    main()
