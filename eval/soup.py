"""测海龟汤裁判 (rag/soup.py): 用每题的标准问答 (data/soup/checks.jsonl) 看裁判判得对不对.

    uv run python -m eval.soup --openrouter --model deepseek-v4.1-flash --effort low   # 全部题
    uv run python -m eval.soup --openrouter --ids soup-08 soup-09                       # 只测几题

checks.jsonl 一行一题:
    {"id": "soup-01", "questions": [{"q": "他哭是因为身体不好吗", "a": "不是"}, ...],
     "good_guess": "说到全部要点的解释", "partial_guess": "只说到一部分", "wrong_guess": "和汤底矛盾的解释"}

输出: 每题问题判对几个 (列出判错的问题, 期望和实际判词), 三种解释是否判对 (good 应 solved, partial / wrong 不应);
最后按档位 (common / rare / obscure) 汇总. 结果另存 data/processed/eval_runs/soup/<name>.jsonl, 一行一题 (有删节):

    {"id": "soup-01", "tier": "common", "person": "刘备",
     "questions": [{"q": "他哭是因为身体不好吗",        标准问题
                    "want": "不是",                     标准判词
                    "got": "不是",                      裁判的判词 (VERDICTS 之一)
                    "raw": "不是",                      模型原本的判词 (不在 VERDICTS 里时 got 记为 "无法判断"; 代码拦下的元问题为 null)
                    "evidence": [0]}, ...],             裁判给的原文编号 (Puzzle.evidence 的下标)
     "guesses": {"good_guess": true, "partial_guess": false, "wrong_guess": false},   三种解释各自是否判 solved
     "q_ok": 9, "q_n": 10,                              问题判对几个 / 共几个
     "guess_ok": 3}                                     解释判对几个: good 判 solved, partial 判不 solved, wrong 判不 solved 三项之和 (0 到 3)
"""
import argparse
import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from rag import config, llm, soup

CHECKS = config.DATA_DIR / "soup" / "checks.jsonl"
OUT_DIR = config.DATA_DIR / "processed" / "eval_runs" / "soup"


def run_one(p: soup.Puzzle, check: dict, complete: Callable[..., Any], model: str | None,
            effort: str | None) -> dict:
    """测一题: 每个标准问题问一次裁判, 三种解释各判一次. 返回结果行 (见模块 docstring)."""
    qs = []
    for c in check["questions"]:
        r = soup.judge_question(p, c["q"], complete, model, effort)
        qs.append({"q": c["q"], "want": c["a"], "got": r["verdict"], "raw": r.get("raw"), "evidence": r["evidence"]})
    guesses = {k: soup.judge_solution(p, check[k], complete, model, effort)["solved"]
               for k in ("good_guess", "partial_guess", "wrong_guess")}
    return {"id": p["id"], "tier": p["tier"], "person": p["person"], "questions": qs, "guesses": guesses,
            "q_ok": sum(q["want"] == q["got"] for q in qs), "q_n": len(qs),
            "guess_ok": guesses["good_guess"] + (not guesses["partial_guess"]) + (not guesses["wrong_guess"])}


def main() -> None:
    ap = argparse.ArgumentParser(description="测海龟汤裁判")
    ap.add_argument("--model", help="裁判用的模型; 不给时用 .env 的 LLM_MODEL (加 --openrouter 时会补全成 OpenRouter 的全名)")
    ap.add_argument("--effort", help="思考程度, 如 low; 不给时用 .env 的 LLM_REASONING_EFFORT (为空就不传)")
    ap.add_argument("--openrouter", action="store_true")
    ap.add_argument("--ids", nargs="+")
    ap.add_argument("--name", default="judge", help=f"结果文件名, 存到 {OUT_DIR}/<name>.jsonl (同名覆盖)")
    args = ap.parse_args()
    model, complete = args.model, llm.complete
    if args.openrouter:
        from rag.openrouter import PER_MODEL, full_name, make_complete, use_openrouter
        use_openrouter()
        model = full_name(model or config.LLM_MODEL)
        sem = threading.Semaphore(PER_MODEL)
        cost = [0.0]

        def complete(*a, **kw):  # 每次请求新包一层 (make_complete 按创建时刻计时), 花费另外累计
            c = make_complete(model, True, sem)
            r = c(*a, **kw)
            cost[0] += c.stats["cost"]
            return r
        complete.cost = cost
    llm.client()  # 先在主线程建客户端, 免得几个线程同时建
    puzzles = {p["id"]: p for p in soup.load_puzzles()}
    with open(CHECKS, encoding="utf-8") as f:
        checks = [json.loads(line) for line in f if line.strip()]
    checks = [c for c in checks if not args.ids or c["id"] in args.ids]
    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(lambda c: run_one(puzzles[c["id"]], c, complete, model, args.effort), checks))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{args.name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                                encoding="utf-8")
    for r in rows:
        bad = [q for q in r["questions"] if q["want"] != q["got"]]
        print(f"{r['id']} [{r['tier']}] {r['person']}: 问题 {r['q_ok']}/{r['q_n']}, 解释 {r['guess_ok']}/3 {r['guesses']}")
        for q in bad:
            print(f"    ✗ {q['q']}  期望 {q['want']}, 判 {q['got']}")
    for tier in ("common", "rare", "obscure"):
        rs = [r for r in rows if r["tier"] == tier]
        if rs:
            print(f"{tier:8s} 问题 {sum(r['q_ok'] for r in rs)}/{sum(r['q_n'] for r in rs)}, "
                  f"解释 {sum(r['guess_ok'] for r in rs)}/{3 * len(rs)}")
    if hasattr(complete, "cost"):
        print(f"花费 ${complete.cost[0]:.4f}")


if __name__ == "__main__":
    main()
