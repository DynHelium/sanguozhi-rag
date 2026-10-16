"""AI 判卷: 把 eval/run_agent.py 的回答连同题目和 gold 原文打包, 交给判卷模型对照原文判 "对 / 部分 / 错", 再汇总.
判卷标准写在 eval/judge_prompt.md (给判卷模型看的).

    uv run python -m eval.ai_judge prep it1          # 打包待判的回答 -> data/processed/eval_runs/judge/it1/batchNN.json
                                                     # --questions 要和 run_agent 时一致 (默认 eval/retrieval.jsonl)
    (判卷模型逐包按 eval/judge_prompt.md 判, 每包写一个 batchNN.out.jsonl)
    uv run python -m eval.ai_judge merge it1         # 汇总 -> data/processed/eval_runs/answers/it1.judged.jsonl, 打印每个模型 对/部分/错 和没答对的题
    uv run python -m eval.ai_judge prep r1 --questions eval/questions-reason.jsonl   # 换题目文件
    uv run python -m eval.ai_judge prep it2 --carry it1                              # 回答没变的沿用 it1 的判决

本脚本不调模型: 判卷那一步交给一个能读写文件, 能跑命令的 AI agent (要用 eval.find 回原文核对回答里 gold 以外的事).
为什么要 AI 判: 迭代 prompt 和代码介入时, 要逐题知道对不对, 错在哪 (尤其是年份和事实), 人工判跟不上.
年号 -> 公元年先由代码按 data/eras.json 核一遍 (check_years), 结果附在包里, 判卷时 ok 为 false 的一律算错.

沿用: 同一 (模型, 题) 的回答文字和上次判过的一样, 直接沿用上次的判决 (按 reply 的 sha1 比对, 存在 judged.jsonl 的
"reply_sha1"), 只判新回答; 迭代时只重跑错题, 判卷也只判这些.

包 batchNN.json (一包 PER_BATCH 道题, 一题带所有待判模型的回答):

    [{"qid": "r-03", "q": "曹操什么时候去过浚仪县？做了什么？", "note": "...",
      "gold": [{"id": "01-0053", "orig": "七年春正月，公軍譙，…", "trans": "建安七年春正月，曹公驻军谯县，…"}],
      "answers": [{"model": "deepseek-v4.1-flash", "reply": "...",
                   "year_checks": [{"text": "建安七年（公元202年）", "ok": true, "why": "应为 202"}]}]},
     ...]

    题目的 type 原样带上; ref / expect / evidence / real / fiction / from_text / needs_reference / judge 这些标注
    (eval/questions-reason.jsonl 有) 有就带上, 判法见 eval/judge_prompt.md.

判决 batchNN.out.jsonl (一行一个回答; 新题集另有 yanyi_flagged / bg_errors):

    {"model": "deepseek-v4.1-flash", "qid": "r-03", "verdict": "对" | "部分" | "错", "errors": ["..."], "comment": "一句话",
     "fiction_as_fact": false, "unsupported": ["..."], "uncertain": ""}
     把演义说成正史  原文查不到却说得很确定的事实  判卷人拿不准的原因 (拿得准为 "")

merge 后的 <name>.judged.jsonl 每行是判决加上 "reply_sha1" 和 "judge" (--judge 给的判卷模型名, 默认 "ai"; 区别于 eval/judge.py 的人工判读);
eval/run_agent.py --summary 读其中的 verdict.
"""
import hashlib
import json
import re
from pathlib import Path

from eval.common import OUT_DIR
from eval.run_retrieval import QUESTIONS_PATH, load_jsonl
from ingest import eras as E
from rag import config
from rag.normalize import to_simplified

JUDGE_DIR = config.DATA_DIR / "processed" / "eval_runs" / "judge"
PER_BATCH = 6  # 每包几道题
_NUM = "元一二三四五六七八九十"


def sha1(text: str) -> str:
    """回答文字的 sha1, 用来判断回答和上次判过的是否相同."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def check_years(reply: str, table: dict[str, dict]) -> list[dict]:
    """回答里 "年号N年 ... 公元Y年" 的写法 (两者相隔不超过 8 字), 按年号表 (ingest.eras.load()) 核对.

    返回每处一条 {"text": 匹配到的原文, "ok": True / False / None, "why": "应为 202"}; ok 为 None 表示年号表里没有
    这个年号或没有元年数据, 核不了. 同名年号 (如蜀汉和吴都有建興) 任一政权对上就算对.
    写成 "公元前N年" 的不核对, 也不返回: 年号表只有公元后的年份, 三国题基本不涉及公元前.
    """
    keys = {to_simplified(k): k for k in table}
    out = []
    for m in re.finditer(rf"([一-鿿]{{2}})([{_NUM}]+)年[^。；\n]{{0,8}}?公元\s*(前?)(\d{{2,3}})\s*年", reply):
        if m.group(3):
            continue
        era, n, ce = m.group(1), m.group(2), int(m.group(4))
        k = keys.get(to_simplified(era))
        if not k:
            out.append({"text": m.group(0), "ok": None, "why": f"年号表里没有 {era}"})
            continue
        y = E.cn_number(n)
        e = table[k]
        cands = [s + y - 1 for s in (e.get("by_regime") or {}).values()] or ([E.era_to_ce(k, y, table)] if e.get("start") else [])
        out.append({"text": m.group(0), "ok": ce in cands if cands else None, "why": f"应为 {'/'.join(map(str, cands)) or '待核'}"})
    return out


def sources() -> tuple[dict[str, str], dict[str, str]]:
    """读原文和白话译文: ({源 id: 原文}, {源 id: 译文}). 裴注的源 id 是 "<段落 id>n<序号>", 如 01-0076n1."""
    src, trans = {}, {}
    with open(config.DATA_DIR / "processed" / "sanguozhi.jsonl", encoding="utf-8") as f:
        for p in map(json.loads, f):
            src[p["id"]] = p["text"]
            for i, n in enumerate(p["notes"]):
                src[f"{p['id']}n{i}"] = n["text"]
    for path in sorted((config.DATA_DIR / "processed" / "translations").glob("卷*.jsonl")):
        with open(path, encoding="utf-8") as f:
            trans.update({r["id"]: r["text"] for r in map(json.loads, f)})
    return src, trans


def judged_path(name: str) -> Path:
    """<name>.judged.jsonl 的路径, 和 eval/judge.py 写的是同一个文件."""
    return OUT_DIR / f"{name}.judged.jsonl"


def load_judged(name: str) -> dict[tuple[str, str], dict]:
    """已有的判决 {(模型, 题号): 判决行}; 文件不存在时为空."""
    p = judged_path(name)
    return {(r["model"], r["qid"]): r for r in load_jsonl(p)} if p.exists() else {}


def prep(name: str, questions_path: Path = QUESTIONS_PATH, carry_from: str | None = None, per_batch: int = PER_BATCH) -> None:
    """把 <name>.jsonl 里没出错, 也不能沿用旧判决的回答打包到 JUDGE_DIR/<name>/batchNN.json (先清掉旧包).

    carry_from: 另一次实验的名字, 它的判决也拿来沿用 (本实验自己的判决优先).
    per_batch: 每包几道题; 一题的回答多 (如同一文件里放了很多种配置做盲判) 时调小, 让一包的量适合一次判完.
    """
    rows = {(r["model"], r["qid"]): r for r in load_jsonl(OUT_DIR / f"{name}.jsonl") if not r.get("error")}
    old = load_judged(name)
    if carry_from:
        old = {**load_judged(carry_from), **old}
    qs = {q["id"]: q for q in load_jsonl(questions_path)}
    src, trans = sources()
    table = E.load()
    todo: dict[str, list] = {}
    for (m, qid), r in rows.items():
        o = old.get((m, qid))
        if o and o.get("reply_sha1") == sha1(r["reply"]):
            continue
        todo.setdefault(qid, []).append({"model": m, "reply": r["reply"], "year_checks": check_years(r["reply"], table)})
    d = JUDGE_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("batch*.json"):
        f.unlink()
    extra = ("type", "ref", "expect", "evidence", "real", "fiction", "from_text", "needs_reference", "judge")  # 新题集的标注, 有就带上
    items = [{"qid": qid, "q": qs[qid]["q"], "note": qs[qid].get("note", ""),
              **{k: qs[qid][k] for k in extra if qs[qid].get(k)},
              "gold": [{"id": g, "orig": src.get(g, ""), "trans": trans.get(g, "")} for g in qs[qid].get("gold", [])],
              "answers": ans} for qid, ans in sorted(todo.items())]
    for i in range(0, len(items), per_batch):
        (d / f"batch{i // per_batch:02d}.json").write_text(json.dumps(items[i:i + per_batch], ensure_ascii=False, indent=1),
                                                           encoding="utf-8")
    print(f"{sum(len(x['answers']) for x in items)} 个回答待判 ({len(items)} 题) -> {d}, {(len(items) + per_batch - 1) // per_batch} 包;"
          f" 沿用 {len(rows) - sum(len(x['answers']) for x in items)} 个")


def merge(name: str, carry_from: str | None = None, judge: str = "ai") -> None:
    """合并各包的 batchNN.out.jsonl 和可沿用的旧判决, 重写 <name>.judged.jsonl, 打印每个模型的 对 / 部分 / 错.

    结果文件里有, 却既没有新判决也不能沿用的回答列为 "没判的". judge 写进新判决的 "judge" 字段 (判卷模型的名字, 默认 "ai").
    """
    rows = {(r["model"], r["qid"]): r for r in load_jsonl(OUT_DIR / f"{name}.jsonl") if not r.get("error")}
    old = load_judged(name)
    if carry_from:
        old = {**load_judged(carry_from), **old}
    new = {}
    for f in sorted((JUDGE_DIR / name).glob("batch*.out.jsonl")):
        for r in load_jsonl(f):
            new[(r["model"], r["qid"])] = r
    out, missing = {}, []
    for k, r in rows.items():
        j = new.get(k)
        if j:
            out[k] = {**j, "reply_sha1": sha1(r["reply"]), "judge": judge}
        elif old.get(k) and old[k].get("reply_sha1") == sha1(r["reply"]):
            out[k] = old[k]
        else:
            missing.append(k)
    judged_path(name).write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in out.values()), encoding="utf-8")
    print(f"{len(out)} 个判决 -> {judged_path(name)}" + (f"; 没判的 {len(missing)}: {missing[:10]}" if missing else ""))
    for m in dict.fromkeys(k[0] for k in out):
        vs = [v for k, v in out.items() if k[0] == m]
        c = {x: sum(v["verdict"] == x for v in vs) for x in ("对", "部分", "错")}
        bad = sorted(v["qid"] for v in vs if v["verdict"] != "对")
        print(f"  {m:20s} 对 {c['对']:2d}  部分 {c['部分']:2d}  错 {c['错']:2d}   不对的: {' '.join(bad)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="AI 判卷")
    ap.add_argument("cmd", choices=["prep", "merge"])
    ap.add_argument("name")
    ap.add_argument("--carry", help="沿用另一次实验里回答相同的判决")
    ap.add_argument("--questions", type=Path, default=QUESTIONS_PATH, help="题目文件 (prep 用)")
    ap.add_argument("--per-batch", type=int, default=PER_BATCH, help=f"每包几道题 (prep 用, 默认 {PER_BATCH})")
    ap.add_argument("--judge", default="ai", help="判卷模型的名字, 写进判决的 judge 字段 (merge 用)")
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a.name, a.questions, carry_from=a.carry, per_batch=a.per_batch)
    else:
        merge(a.name, carry_from=a.carry, judge=a.judge)
