"""agent 评测: 每题 x 每个模型跑一遍 rag/agent.run_agent, 记下模型看过哪些块, 回答和引文.

    # 默认配置: DeepSeek V4.1 Flash 经 OpenRouter (只用 rag/openrouter.PROVIDERS 指定的服务商), 思考 low, 留出题
    uv run python -m eval.run_agent --openrouter --models deepseek-v4.1-flash --effort low \
        --questions eval/questions-holdout.jsonl --name <名>
    uv run python -m eval.run_agent --openrouter --models deepseek-v4.1-flash --effort low \
        --questions eval/questions-holdout.jsonl eval/questions-en.jsonl --name cv    # 几套题一个进程跑完 (索引和 embedding 模型只加载一次) -> cv-holdout.jsonl, cv-en.jsonl
    uv run python -m eval.run_agent --models deepseek-v4.1-flash --ids r-03 r-16 --name dev   # 不加 --openrouter: 用 .env 里的 LLM_BASE_URL / LLM_API_KEY 直连
    uv run python -m eval.run_agent --off review ... --name <名>                # 消融: 关掉某个代码介入 (覆盖 agent.OFF)
    uv run python -m eval.run_agent --rescore --name <名> --questions eval/questions-holdout.jsonl   # 改了 gold / equiv 后重算覆盖字段, 不调模型; --questions 和 --index 要与当初运行时一致
    uv run python -m eval.run_agent --summary --name <名> [--questions ...]     # 只打印汇总 (几套题一起跑的, 带上同样的 --questions)

判对错的工具 (AI 判卷, 浏览器人工判读, 实验对比) 在 eval-judging 分支; 判卷结果写成 <名>.judged.jsonl 时, --summary 会一并统计.

默认配置: 白话索引 (--index trans), 别名扩展 agent.ALIAS, system prompt 和工具声明用 rag/agent.py 的 SYSTEM / TOOLS.
--prompt / --tools 做对照: 读 eval/agent_prompts/<名字>.txt / <名字>.tools.json (仓库里不带这个目录, 要自己建, 放对照用的 prompt 和工具声明).
题目: 默认 eval/retrieval.jsonl, 去掉 SKIP 里的演义题; 给了 --ids 时只跑列出的题 (SKIP 不再生效).
none 题 (资料里没有答案, gold 为空) 照跑, 看模型会不会老实说没找到.
续跑: 每跑完一题追加一行; 重跑时跳过已有且没出错的 (模型, 题); 跑完后同一 (模型, 题) 只留最后一行.

结果 data/processed/eval_runs/answers/<name>.jsonl, 每行一个 (模型, 题). 示例 (留出题 h-01, 有删节):

    {"model": "deepseek-v4.1-flash",           短名 (模型全名 "/" 后面的部分); 给了 --tag 时写 tag (盲判用)
     "model_full": "deepseek/deepseek-v4.1-flash",
     "qid": "h-01", "type": "named", "q": "刘备从葭萌南下打刘璋时留霍峻守城，……", "gold": ["41-0000"],
     "retrieved": ["41-0000", "41-0017", ...], 模型看过的所有块 id (run_agent 返回的 chunks), 按资料编号 [n] 的顺序 (模型第一次看到的先后), retrieved[n-1] 是 [n]
     "gold_in_ctx": true,                       严格口径: 看过的块里, 至少一块的 sources 含某个 gold 或它的等价段落
     "gold_all": true,                          严格口径: 每个 gold (或它的等价段落) 都被看到了; 要求列举的题 "找全" 看这个
     "gold_in_ctx_loose": true,                 宽口径 (metrics.is_hit: 挂在 gold 正文上的裴注块也算), 和检索评测对照用
     "gold_all_loose": true,                    宽 - 严 = "只搜到注, 没看到 gold 正文" 的题
     "gold_in_first": true,                     模型第一次成功的 search 是否命中 gold (严格口径)
     "reply": "**结论**：……", "quotes": [["小人頭可得，城不可得", "41-0000"], ...], 回答里的「」引文和找到它的块 id (找不到为 null)
     "cites": [{"quote": "...", "cites": [3], "status": "ok", "found": [3]}, ...],   引文所标编号的核对 (同 run_agent)
     "refused": false,                          回答像不像拒答 (common.is_refusal, 启发式)
     "sec": 33.0,                               这题用时 (秒)
     "rounds": [{"tool": "search", "args": {...}, "chunks": [...], "error": null, "by": "model"}, ...,
                {"by": "code", "tool": "review", "args": {}, "chunks": [], "error": null}],   同 run_agent 的 rounds
     "calls": 4, "prompt_tokens": [2630, 10882, 14246, 13752],   模型请求次数, 每次请求的输入 token 数; 接口没返回 usage 时该项为 null
     "usage": {"prompt": 41510, "completion": 1261, "cached": 17536, "cost": 0.001},   rag/openrouter.make_complete 的记账
     "prompt": "SYSTEM", "tools": "TOOLS", "effort": "low", "index": "trans", "alias": "route2", "off": ["nudge_notfound"],
     "review_mode": "layered", "system_sha": "3f2a…", "review_sha": "9c1e…",   这次运行的配置; *_sha 是 prompt 和自查指令全文的哈希
     "error": "TimeoutError: ..."}              只有出错时才有 (这时 reply 为 "", retrieved 为 []); 续跑时重跑这题

为什么用严格口径: 检索评测里 "搜到某段的裴注也算找到这段" 说得过去 (读的人顺着能看到正文), 但 agent 评测问的是
"模型手里有没有答案所在的原文"; 一段正文常挂好几条注, 讲的往往是别的事.

等价段落: 题目的可选字段 "equiv": {"46-0016": ["46-0016n0"]}, 意思是单独读 46-0016n0 也能回答
46-0016 回答的那一点 (同一件事在别的传或注引书里的记载). 看到等价段落算这个 gold 已找到; 不并进 gold, 免得改变
"找全" 的含义.

注意: agent 不预先检索, 看到多少块由模型调用几次 search / read 决定, 所以这里的 "看到" 和检索评测
(eval/run_retrieval.py, 固定取前 k 块) 的 hit@k 不是同样的块数预算.
"""
import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval import metrics
from eval.common import OUT_DIR, is_refusal, out_paths, summary
from eval.run_retrieval import QUESTIONS_PATH, load_jsonl
from rag import agent, llm
from rag.openrouter import PER_MODEL, full_name, make_complete, use_openrouter
from rag.retrieve import load_chunks

PROMPTS = Path(__file__).with_name("agent_prompts")
SKIP = {"r-20"}  # 演义题 (割须弃袍): 问的是小说情节, 只在检索评测里用



def _sha(text: str) -> str:
    """prompt 文字的短哈希, 记进结果: 同名 prompt 改过之后也能分清是哪一版."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def covered(chunks: list[dict], gold_id: str, equiv: dict[str, list[str]]) -> bool:
    """严格口径: 某块的 sources 含 gold_id 或它的等价段落 (不认 parent)."""
    ok = {gold_id, *equiv.get(gold_id, [])}
    return any(s in ok for c in chunks for s in c["sources"])


def coverage(chunks: list[dict], gold: list[str], equiv: dict[str, list[str]]) -> dict[str, bool]:
    """结果行里的四个覆盖字段 (严格 / 宽口径, 见模块 docstring). gold 为空 (none 题) 时 gold_all 和 gold_all_loose 为 False."""
    loose = [any(metrics.is_hit(c, [g]) or covered([c], g, equiv) for c in chunks) for g in gold]
    strict = [covered(chunks, g, equiv) for g in gold]
    return {"gold_in_ctx": any(strict), "gold_all": bool(gold) and all(strict),
            "gold_in_ctx_loose": any(loose), "gold_all_loose": bool(gold) and all(loose)}


def rescore(out: Path, index: str, questions_path: Path = QUESTIONS_PATH) -> None:
    """改了 gold 或补了等价段落后, 用每行的 retrieved 重算覆盖字段 (不调模型)."""
    by_id = {c["id"]: c for c in load_chunks("" if index == "orig" else index)}
    qs = {q["id"]: q for q in load_jsonl(questions_path)}
    rows = load_jsonl(out)
    for r in rows:
        q = qs[r["qid"]]
        first = next((x["chunks"] for x in r["rounds"] if x["tool"] == "search" and not x.get("error")), [])
        r["gold"] = q.get("gold", [])
        r.update(coverage([by_id[i] for i in r["retrieved"]], r["gold"], q.get("equiv", {})))
        r["gold_in_first"] = any(covered([by_id[c]], g, q.get("equiv", {})) for c in first for g in r["gold"])
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"重算 {len(rows)} 行 -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="agent 评测")
    ap.add_argument("--models", nargs="+", help="必填 (--summary / --rescore 时不用)")
    ap.add_argument("--ids", nargs="+", help="只跑这些题")
    ap.add_argument("--questions", type=Path, nargs="+", default=[QUESTIONS_PATH],
                    help="题目文件, 可以给几个 (默认 eval/retrieval.jsonl); 给几个时一个进程跑完, 结果按题集分文件 (见 eval.common.out_paths)")
    ap.add_argument("--name", default="agent")
    ap.add_argument("--index", default="trans", choices=["trans", "orig"], help="检索哪套切块 (默认白话)")
    ap.add_argument("--prompt", help="system prompt 换成 eval/agent_prompts/<名字>.txt (仓库里不带这个目录, 要自己建); 不给就用 rag/agent.py 的 SYSTEM")
    ap.add_argument("--effort", help="reasoning_effort (none / low / medium / high), 覆盖 .env")
    ap.add_argument("--tools", help="工具声明换成 eval/agent_prompts/<名字>.tools.json (仓库里不带这个目录, 要自己建); 不给就用 rag/agent.py 的 TOOLS")
    ap.add_argument("--alias", default=agent.ALIAS, help=f"检索的别名扩展方式 (默认 {agent.ALIAS}; \"\" = 不扩展)")
    ap.add_argument("--openrouter", action="store_true", help="走 OpenRouter (key 见 rag/openrouter.use_openrouter), 只用 rag/openrouter.PROVIDERS 指定的服务商")
    ap.add_argument("--jobs", type=int, default=1, help="同时跑几题 (单模型另有并发上限 PER_MODEL; --openrouter 时至少为 PER_MODEL x 模型数)")
    ap.add_argument("--review-mode", default=agent.REVIEW_MODE, choices=["rewrite", "short", "layered", "patch"],
                    help="自查的做法 (agent.REVIEW_MODE): rewrite = 一律重写整篇, short = 没问题只回 无误, "
                         "layered = 事实 / 推断 / 背景分开核对 (配合把回答分成这三类的 system prompt), "
                         "patch = 核对标准同 layered, 只列出要改的句子由代码替换")
    ap.add_argument("--review-effort", help="自查等作答之后几步的 reasoning_effort (agent.REVIEW_EFFORT); 不给就和作答相同")
    ap.add_argument("--off", nargs="+", choices=["nudge", "nudge_notfound", "review", "quote_fix", "cite_fix", "none"],
                    help=f"消融: 关掉这些代码介入, 覆盖 agent.OFF (默认 {sorted(agent.OFF)}); none = 全开")
    ap.add_argument("--tag", help="结果里 model 字段写这个 (盲判/同一文件放多种配置用); 只能配一个模型")
    ap.add_argument("--summary", action="store_true", help="不跑模型, 只打印汇总 (有判读结果时一并统计); 几套题的结果要带上跑时同样的 --questions")
    ap.add_argument("--rescore", action="store_true",
                    help="不跑模型: 按 --questions 指定的题目文件 (gold / equiv) 重算 <name>.jsonl 里已有结果的覆盖字段")
    args = ap.parse_args()
    if args.summary:
        for o in out_paths(args.name, args.questions):
            summary(o)
        return
    if args.rescore:
        for qp, o in zip(args.questions, out_paths(args.name, args.questions)):
            rescore(o, args.index, qp)
        return
    if not args.models:
        ap.error("--models 必填")
    if args.tag and len(args.models) > 1:
        ap.error("--tag 只能配一个模型")
    if args.off is not None:
        agent.OFF = set(args.off) - {"none"}
    agent.REVIEW_MODE = args.review_mode
    agent.REVIEW_EFFORT = args.review_effort
    if args.openrouter:
        use_openrouter()
        args.models = [full_name(m) for m in args.models]
        args.jobs = max(args.jobs, PER_MODEL * len(args.models))
    system = (PROMPTS / f"{args.prompt}.txt").read_text(encoding="utf-8").strip() if args.prompt else None
    tool_decl = json.loads((PROMPTS / f"{args.tools}.tools.json").read_text(encoding="utf-8")) if args.tools else None
    outs = out_paths(args.name, args.questions)
    sets = []  # [(结果文件, 题目, 已跑完的 (模型, 题))], 每个题目文件一项
    for qp, out in zip(args.questions, outs):
        questions = [q for q in load_jsonl(qp) if q["id"] not in SKIP]
        if args.ids:
            questions = [q for q in load_jsonl(qp) if q["id"] in args.ids]
        done = {(r["model"], r["qid"]) for r in load_jsonl(out) if not r.get("error")} if out.exists() else set()
        sets.append((out, questions, done))
    lib = agent.load_library("" if args.index == "orig" else args.index, args.alias)
    by_id = {c["id"]: c for c in lib.chunks}
    sems = {m: threading.Semaphore(PER_MODEL) for m in args.models}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"prompt": args.prompt or "SYSTEM", "tools": args.tools or "TOOLS", "effort": args.effort, "index": args.index,
            "alias": args.alias, "off": sorted(agent.OFF), "review_mode": agent.REVIEW_MODE, "review_effort": agent.REVIEW_EFFORT,
            "system_sha": _sha(system or agent.system_for(args.models[0])),
            "review_sha": _sha({"short": agent.REVIEW_TEXT_SHORT, "layered": agent.REVIEW_TEXT_LAYERED,
                                "patch": agent.REVIEW_TEXT_PATCH}
                               .get(agent.REVIEW_MODE, agent.REVIEW_TEXT))}
    label = lambda m: args.tag or m.split("/")[-1]  # noqa: E731

    def one(job: tuple[str, dict, Path]) -> None:
        m, q, out = job
        gold, equiv, t = q.get("gold", []), q.get("equiv", {}), time.time()
        complete = make_complete(m, args.openrouter, sems[m])
        try:
            res, err = agent.run_agent(q["q"], lib, model=m, complete=complete,
                                       system=system, effort=args.effort, tools=tool_decl), None
        except Exception as e:  # 接口报错: 记下来继续, 下次重跑
            res, err = {"reply": "", "chunks": [], "first": [], "quotes": [], "rounds": [], "calls": 0,
                        "prompt_tokens": []}, f"{type(e).__name__}: {e}"[:500]
        cov = coverage(res["chunks"], gold, equiv)
        row = {"model": label(m), "model_full": m, "qid": q["id"], "type": q.get("type", "?"), "q": q["q"], "gold": gold,
               "retrieved": [c["id"] for c in res["chunks"]], **cov,
               "gold_in_first": any(covered([by_id[cid]], g, equiv) for cid in res["first"] for g in gold),
               "reply": res["reply"], "draft": res.get("draft"), "review_raw": res.get("review_raw"), "quotes": [list(x) for x in res["quotes"]], "cites": res.get("cites", []), "refused": is_refusal(res["reply"]),
               "sec": round(time.time() - t, 1), "rounds": res["rounds"], "calls": res["calls"],
               "prompt_tokens": res["prompt_tokens"], "usage": complete.stats, **meta}
        if err:
            row["error"] = err
        with lock_out:
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tools = [f"{r['tool']}({next(iter(r['args'].values()), '') if r['args'] else ''})" for r in res["rounds"] if r.get("by") == "model"]
        print(f"{row['model']} {q['id']} {row['sec']}s 工具 {tools or '无'} gold 首轮{'✓' if row['gold_in_first'] else '✗'}"
              f"/看到{'✓' if row['gold_in_ctx'] else '✗'}/找全{'✓' if row['gold_all'] else '✗'}"
              f" 引文 ✓{sum(1 for _, c in res['quotes'] if c)}✗{sum(1 for _, c in res['quotes'] if not c)}"
              + (f" 错误 {err}" if err else ""), file=sys.stderr, flush=True)

    lock_out = threading.Lock()
    jobs = [(m, q, out) for out, questions, done in sets for q in questions for m in args.models if (label(m), q["id"]) not in done]
    print(f"要跑 {len(jobs)} 个 (模型, 题)", file=sys.stderr)
    llm.client()  # 先在主线程建好客户端: 几个线程同时第一次建时会同时导入 httpx, 偶发 "partially initialized module"
    with ThreadPoolExecutor(max(1, args.jobs)) as ex:
        list(ex.map(one, jobs))
    for out in outs:
        if not out.exists():
            continue
        rows = load_jsonl(out)
        keep = {}
        for r in rows:  # 同一 (模型, 题) 重跑过的, 只留最后一条
            keep[(r["model"], r["qid"])] = r
        if len(keep) < len(rows):
            out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep.values()), encoding="utf-8")
        print(f"\n#### {out.name}")
        summary(out)


if __name__ == "__main__":
    main()
