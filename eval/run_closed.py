"""闭卷基线: 不给工具, 不给资料, 模型凭记忆直接回答, 看 RAG 帮了多少. 每题只调 1 次模型.
结果格式和 eval/run_agent.py 相同 (没有检索相关字段, usage 同样是 token 和花费记账), 可以直接用 eval/ai_judge.py 判卷.

    uv run python -m eval.run_closed --openrouter --models deepseek-v4.1-flash --effort low \\
        --questions eval/questions-holdout.jsonl --name closed-holdout
    uv run python -m eval.run_closed ... --questions eval/questions-*.jsonl --name cb      # 几套题一起跑 -> cb-holdout.jsonl, cb-en.jsonl, ...
    uv run python -m eval.ai_judge prep closed-holdout --questions eval/questions-holdout.jsonl

system prompt (CLOSED; 英文提问时加 CLOSED_EN) 只要求依据《三国志》, 说清正史与演义的区别, 不知道就说不知道; 不要求引原文 (没有资料可引).
"""
import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag.openrouter import PER_MODEL, full_name, make_complete, use_openrouter
from eval.common import OUT_DIR, is_refusal, out_paths
from eval.run_retrieval import QUESTIONS_PATH, load_jsonl

CLOSED = ("你是一位讲解三国历史的老师. 依据《三国志》(陈寿正文和裴松之注) 回答提问, 用简体中文, 先给结论再给依据. "
          "问题出自《三国演义》而正史不同的, 说明这是演义的说法, 正史怎么记载. 不确定的事实 (年份, 人名, 数字) 不要编造, 直接说不确定.")
CLOSED_EN = " 用户用英文提问时用英文回答, 人名用汉语拼音."  # 只在英文提问时加在 CLOSED 后面, 中文题的 prompt 不变


def system_for(question: str) -> str:
    """英文提问 (问题里没有汉字) 用 CLOSED + CLOSED_EN, 否则用 CLOSED."""
    return CLOSED + CLOSED_EN if not re.search(r"[\u4e00-\u9fff]", question) else CLOSED


def main() -> None:
    ap = argparse.ArgumentParser(description="闭卷基线")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--questions", type=Path, nargs="+", default=[QUESTIONS_PATH], help="题目文件, 可以给几个 (见 eval.common.out_paths)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--effort")
    ap.add_argument("--openrouter", action="store_true")
    args = ap.parse_args()
    if args.openrouter:
        use_openrouter()
        args.models = [full_name(m) for m in args.models]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sems = {m: threading.Semaphore(PER_MODEL) for m in args.models}
    lock = threading.Lock()

    def one(job: tuple[str, dict, Path]) -> None:
        m, q, out = job
        t = time.time()
        complete = make_complete(m, args.openrouter, sems[m])
        extra = {"reasoning_effort": args.effort} if args.effort else {}
        try:
            r = complete([{"role": "system", "content": system_for(q["q"])}, {"role": "user", "content": q["q"]}], model=m, **extra)
            reply, err = r.choices[0].message.content or "", None
        except Exception as e:
            reply, err = "", f"{type(e).__name__}: {e}"[:300]
        row = {"model": m.split("/")[-1], "qid": q["id"], "type": q.get("type", "?"), "q": q["q"], "gold": q.get("gold", []),
               "retrieved": [], "reply": reply, "quotes": [], "refused": is_refusal(reply), "sec": round(time.time() - t, 1),
               "rounds": [], "calls": 1, "usage": complete.stats, "prompt": "CLOSED", "effort": args.effort}
        if err:
            row["error"] = err
        with lock, open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{row['model']} {q['id']} {row['sec']}s {len(reply)} 字" + (f" 错误 {err}" if err else ""), file=sys.stderr, flush=True)

    jobs = []
    for qp, out in zip(args.questions, out_paths(args.name, args.questions)):
        done = {(r["model"], r["qid"]) for r in load_jsonl(out) if not r.get("error")} if out.exists() else set()
        jobs += [(m, q, out) for q in load_jsonl(qp) for m in args.models if (m.split("/")[-1], q["id"]) not in done]
    with ThreadPoolExecutor(max(1, PER_MODEL * len(args.models))) as ex:
        list(ex.map(one, jobs))


if __name__ == "__main__":
    main()
