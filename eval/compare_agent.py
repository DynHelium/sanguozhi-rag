"""对比几次 agent 实验的工具使用和找资料的情况. 只读结果文件, 不调模型:

    uv run python -m eval.compare_agent <名1> <名2> ...

每个 <name> 是 data/processed/eval_runs/answers/<name>.jsonl (eval/run_agent.py 的输出). 按 (实验, 模型) 各一行:

    用工具   至少调用了一次工具的题数 / 题数
    search  平均每题 search 次数          换说法  search 了 2 次以上且检索词不全相同的题数
    read    平均每题 read 次数            era    平均每题 era 次数
    看到     至少一个 gold 出现在模型看过的块里 (none 题不算; 严格口径, 见 eval/run_agent.py)
    找全(宽) 每个 gold 都看到了 (严格口径); 括号里是宽口径 (挂在 gold 正文上的裴注块也算)
    中位秒   每题用时中位数
    报错     调用接口出错的题数
只比 "有没有去找, 找没找到", 不评回答对错 (那要判读, 见 eval/ai_judge.py, eval/judge.py).
"""
import statistics
import sys

from eval.common import OUT_DIR
from eval.run_retrieval import load_jsonl


def row_stats(rows: list[dict]) -> dict:
    """一个 (实验, 模型) 的结果行 -> 上表各列的数字. 只数模型自己发起的工具调用 (rounds 里 by == "model"),
    不算代码介入 (自查等). 找全的两列在缺字段时互相代替 (旧结果文件只有其中一种)."""
    def calls(r, tool):
        return [x for x in r["rounds"] if x.get("by") == "model" and x["tool"] == tool]
    scored = [r for r in rows if r["gold"]]
    n = len(rows)
    return {
        "n": n,
        "used": sum(1 for r in rows if any(x.get("by") == "model" for x in r["rounds"])),
        "search": sum(len(calls(r, "search")) for r in rows) / n,
        "rephrase": sum(1 for r in rows if len({str(x["args"]) for x in calls(r, "search")}) >= 2),
        "read": sum(len(calls(r, "read")) for r in rows) / n,
        "era": sum(len(calls(r, "era")) for r in rows) / n,
        "found": (sum(r["gold_in_ctx"] for r in scored), len(scored)),
        "all": sum(r.get("gold_all", r.get("gold_all_loose", False)) for r in scored),
        "all_loose": sum(r.get("gold_all_loose", r.get("gold_all", False)) for r in scored),
        "sec": statistics.median(r["sec"] for r in rows),
        "errors": sum(1 for r in rows if r.get("error")),
    }


def main() -> None:
    names = sys.argv[1:]
    if not names:
        sys.exit(__doc__)
    print(f"{'实验':<16}{'模型':<18}{'用工具':>8}{'search':>8}{'换说法':>7}{'read':>6}{'era':>6}{'看到':>9}{'找全(宽)':>10}{'中位秒':>7}{'报错':>5}")
    for name in names:
        rows = load_jsonl(OUT_DIR / f"{name}.jsonl")
        for m in dict.fromkeys(r["model"] for r in rows):
            s = row_stats([r for r in rows if r["model"] == m])
            print(f"{name:<16}{m:<18}{s['used']:>4}/{s['n']:<3}{s['search']:>8.1f}{s['rephrase']:>7}{s['read']:>6.1f}"
                  f"{s['era']:>6.1f}{s['found'][0]:>5}/{s['found'][1]:<3}{s['all']:>5}({s['all_loose']}){s['sec']:>7.1f}{s['errors']:>5}")


if __name__ == "__main__":
    main()
