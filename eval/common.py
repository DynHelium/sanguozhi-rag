"""agent 评测脚本 (eval/run_agent.py, 以及 eval-judging 分支里的判卷工具) 共用的东西: 结果目录, 拒答判断, 汇总."""
import statistics
from pathlib import Path

from eval.run_retrieval import load_jsonl
from rag import config

OUT_DIR = config.DATA_DIR / "processed" / "eval_runs" / "answers"
REFUSAL_WORDS = ["没有记载", "沒有記載", "未记载", "未記載", "无记载", "沒有相關", "没有相关", "不知道", "无法回答", "無法回答", "没有提及", "未提及"]


def is_refusal(reply: str) -> bool:
    """启发式: 回答里出现 REFUSAL_WORDS 之一. 会把 "资料没有记载 X, 但记载了 Y" 这种部分回答也算拒答, 判读时以人工为准."""
    return any(w in reply for w in REFUSAL_WORDS)


def summary(out: Path) -> None:
    """打印一个结果文件 (eval/run_agent.py 的 <name>.jsonl) 按模型的汇总.

    每个模型一段, none 题 (gold 为空) 单独统计拒答:

        == deepseek-v4.1-flash  (30 题, 中位用时 33.0s)
          gold 在上下文: 28/30
          拒答: gold 在上下文时 0/28 (越少越好), 不在时 1/2 (越多越老实)
          引文: ✓95 ✗3 (✗ 占 3%)
          判读 30/30 题:          回答对   错/部分/拒答
            gold 在上下文         25        3
            gold 不在上下文        0        2

    "gold 在上下文" 用严格口径 gold_in_ctx. 同目录下有 <name>.judged.jsonl (判卷工具写的, 见 eval-judging 分支;
    每行 {"model", "qid", "verdict"}) 时加上最后的 2x2 表: 把回答错分成 "资料没找到" 和 "找到了却答错" 两类.
    """
    rows = load_jsonl(out)
    judged_path = out.with_suffix(".judged.jsonl")
    verdicts = {(j["model"], j["qid"]): j["verdict"] for j in load_jsonl(judged_path)} if judged_path.exists() else {}
    for model in dict.fromkeys(r["model"] for r in rows):
        rs = [r for r in rows if r["model"] == model and r["type"] != "none"]
        nones = [r for r in rows if r["model"] == model and r["type"] == "none"]
        ok = sum(1 for r in rs for _, c in r["quotes"] if c)
        bad = sum(1 for r in rs for _, c in r["quotes"] if not c)
        in_ctx = [r for r in rs if r["gold_in_ctx"]]
        print(f"\n== {model}  ({len(rs)} 题, 中位用时 {statistics.median(r['sec'] for r in rs):.1f}s)")
        print(f"  gold 在上下文: {len(in_ctx)}/{len(rs)}")
        print(f"  拒答: gold 在上下文时 {sum(r['refused'] for r in in_ctx)}/{len(in_ctx)} (越少越好), "
              f"不在时 {sum(r['refused'] for r in rs if not r['gold_in_ctx'])}/{len(rs) - len(in_ctx)} (越多越老实)")
        print(f"  引文: ✓{ok} ✗{bad}" + (f" (✗ 占 {bad / (ok + bad):.0%})" if ok + bad else ""))
        if nones:
            print(f"  none 题拒答: {sum(r['refused'] for r in nones)}/{len(nones)}")
        errs = [r for r in rs if r.get("error")]
        if errs:
            print(f"  调用出错: {len(errs)} 题")
        mine = {r["qid"]: verdicts.get((model, r["qid"])) for r in rs}
        if any(mine.values()):
            cell = {(a, b): 0 for a in (True, False) for b in ("对", "其他")}
            for r in rs:
                v = mine[r["qid"]]
                if v:
                    cell[(r["gold_in_ctx"], "对" if v == "对" else "其他")] += 1
            n = sum(1 for v in mine.values() if v)
            print(f"  判读 {n}/{len(rs)} 题:          回答对   错/部分/拒答")
            print(f"    gold 在上下文     {cell[(True, '对')]:>6}   {cell[(True, '其他')]:>6}")
            print(f"    gold 不在上下文   {cell[(False, '对')]:>6}   {cell[(False, '其他')]:>6}")
        else:
            print(f"  还没有判读 ({judged_path.name})")
