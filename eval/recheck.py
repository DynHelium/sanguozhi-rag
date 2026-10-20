"""判卷的两个抽查: 主对照的判决是不是受了回答格式影响, 后端对比的判决是不是受了评委和 Haiku 同属一家的影响.
只读 eval/results 里的回答和判决; cross run 调 OpenRouter 上的 OpenAI 模型, 其他命令不调模型.

    uv run python -m eval.recheck nocite prep      # 30 题去掉 [n] 后打包 -> data/processed/eval_runs/judge/nocite30/
    (判卷 agent 按 eval/judge_prompt.md 逐包判, 同 eval/ai_judge.py)
    uv run python -m eval.recheck nocite report    # -> eval/results/recheck-nocite30.judged.jsonl, 打印和原判决的对比
    uv run python -m eval.recheck cross run --dry  # 只拼 prompt, 打印字符数 (估费用)
    uv run python -m eval.recheck cross run        # 20 题交给 gpt-6.1-sol 判 (约 $0.4), 已判的题跳过
    uv run python -m eval.recheck cross report     # -> eval/results/recheck-cross-gpt-6.1-sol.jsonl, 打印和原评委判决的对比

nocite (格式统一后重判): 主对照里闭卷回答没有 [n] 编号, 一次检索和 agent 的回答有, 标签打乱了评委也认得出闭卷.
    按题集分层抽 30 题 (QUOTA), 去掉一次检索和 agent 回答里的 [n] (CITE), 三份回答重新打乱标签, 用同样的判卷标准重判,
    和公开的最终判决 (含作者改判) 逐份对比. 回答里 "裴注引《X》" 这类出处的说法照留: 那是内容, 评委要核对.
cross (第三家评委): 后端对比的评委和 Haiku 是同一家的模型. 从 eval/questions-probe62.jsonl 抽 20 题,
    DeepSeek 和 Haiku (medium) 的回答一题一次调用、标签打乱, 交给 OpenAI 的模型 (CROSS_MODEL) 判, 和原评委逐份对比.
    API 评委不能像判卷 agent 那样用 eval.find 检索全书, 所以把 gold 段落和每份回答 [n] 引到的块 (原文 + 白话) 附上,
    并说明所附原文里查不到的事只记 unsupported、不因此判错 (CROSS_RULES). 这一点和原评委的判法不同, 分歧要逐条看.
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from eval import ai_judge
from eval.common import OUT_DIR
from eval.run_retrieval import load_jsonl
from ingest import eras as E
from rag import config

RESULTS = config.ROOT / "eval" / "results"
GROUPS = {"closed": "closed", "single": "single", "agent": "agent"}  # 组名 -> eval/results 里的文件前缀
QUOTA = {"holdout": 12, "facts": 7, "persons": 3, "reason": 8}       # 每个题集抽几题, 大致按题数比例 (40 / 24 / 9 / 30)
CITE = re.compile(r"\s?\[\d+(?:\s*[,，、\-–]\s*\d+)*\]")              # [3] [3, 5] [3-5] 这类编号
NOCITE = "nocite30"
BACKENDS = {"deepseek": "backends-deepseek-probe62", "haiku": "backends-haiku-medium-probe62"}
CROSS_N = 20
CROSS_MODEL = "openai/gpt-6.1-sol"
CROSS_EFFORT = "medium"
CROSS_RULES = """
## 这次判卷的特别说明（优先于上面关于检索的说明）

- 你不能检索全书。每份回答附上了它引用的原文块（`passages`，键是回答里的编号 n），加上 gold 段落，这些就是你核对的依据。
- 回答里 gold 和所附原文块以外的具体事实：和这些原文矛盾 → 按规则算错；原文里查不到但也不矛盾的，写进 `unsupported`
  （写明"所附原文里没有"），**不因为查不到而判错**，因为你看不到全书。
- 两份回答标签随机（g1 / g2），来自不同系统，各自独立判，不要比较、不要猜来源。
- 出题人补充的边界规则：答到核心只是夹带编造或张冠李戴的引文、出处或细节 → 部分；错误非常小（一个字、措辞、出处标错但引文和事实都对）→ 对；
  核心答错、有答案却说没找到、公元年错、把演义说成正史 → 错。
- 只输出一个 JSON 数组（不要别的文字），每份回答一个对象，字段：model, qid, verdict（对 / 部分 / 错）, errors, comment,
  fiction_as_fact, unsupported, uncertain；qid 以 i- / y- / g- 开头的另加 yanyi_flagged, bg_errors。
"""
NOCITE_SEED = "nb30-2026-10-19"   # 随机种子 (抽题和打乱标签); 结果文件按它抽的题判, 改了就对不上
CROSS_SEED = "xjudge-2026-10-19"
EXTRA = ("type", "ref", "expect", "evidence", "real", "fiction", "from_text", "needs_reference", "judge")  # 同 ai_judge.prep


def tally(verdicts) -> str:
    """一组判决 -> "对/部分/错" 的计数."""
    c = Counter(verdicts)
    return f"{c['对']}/{c['部分']}/{c['错']}"


# ---------- nocite: 格式统一后重判 ----------

def nocite_prep() -> None:
    """抽题, 去 [n], 打乱标签 -> OUT_DIR/nocite30.jsonl (标签 g1..g3 当作 model), 标签对应表和合并的题目文件放在
    JUDGE_DIR/nocite30/, 再按 ai_judge.prep 打包 (一包 3 题 9 份回答)."""
    rng = random.Random(NOCITE_SEED)
    rows, mapping, qs = [], {}, []
    for s, n in QUOTA.items():
        by: dict[str, dict] = {}
        for g, prefix in GROUPS.items():
            for r in load_jsonl(RESULTS / f"{prefix}-{s}.jsonl"):
                if not r.get("error") and r["reply"]:
                    by.setdefault(r["qid"], {})[g] = r
        pick = sorted(rng.sample(sorted(q for q, d in by.items() if len(d) == len(GROUPS)), n))  # 三种配置都有回答的题
        for qid in pick:
            gs = list(GROUPS)
            rng.shuffle(gs)
            for i, g in enumerate(gs):
                r = by[qid][g]
                mapping[f"{qid}|g{i + 1}"] = g
                rows.append({"model": f"g{i + 1}", "qid": qid, "q": r["q"],
                             "reply": r["reply"] if g == "closed" else CITE.sub("", r["reply"])})
        qs += [q for q in load_jsonl(config.ROOT / f"eval/questions-{s}.jsonl") if q["id"] in pick]
    d = ai_judge.JUDGE_DIR / NOCITE
    d.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{NOCITE}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (d / "map.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "questions.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs), encoding="utf-8")
    ai_judge.prep(NOCITE, d / "questions.jsonl", per_batch=3)


def nocite_report() -> None:
    """各包的判决按标签还原成组 -> eval/results/recheck-nocite30.judged.jsonl; 打印和公开最终判决的一致数与每条分歧."""
    d = ai_judge.JUDGE_DIR / NOCITE
    mapping = json.loads((d / "map.json").read_text(encoding="utf-8"))
    new = []
    for f in sorted(d.glob("batch*.out.jsonl")):
        for j in load_jsonl(f):
            new.append({**j, "group": mapping[f"{j['qid']}|{j['model']}"], "blind_tag": j["model"], "judge": "claude-sonnet"})
    (RESULTS / f"recheck-{NOCITE}.judged.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new),
                                                            encoding="utf-8")
    old = {(g, j["qid"]): j for g, prefix in GROUPS.items() for s in QUOTA
           for j in load_jsonl(RESULTS / f"{prefix}-{s}.judged.jsonl")}
    same = sum(old[(r["group"], r["qid"])]["verdict"] == r["verdict"] for r in new)
    raw = sum((old[(r["group"], r["qid"])].get("verdict_ai") or old[(r["group"], r["qid"])]["verdict"]) == r["verdict"] for r in new)
    print(f"{len(new)} 份判决, 和公开的最终判决一致 {same}; 和作者改判前的 AI 原判 (verdict_ai) 一致 {raw}")
    for g in GROUPS:
        mine = [r for r in new if r["group"] == g]
        print(f"  {g:7s} 一致 {sum(old[(g, r['qid'])]['verdict'] == r['verdict'] for r in mine)}/{len(mine)}"
              f"   重判 对/部分/错 {tally(r['verdict'] for r in mine)}"
              f"   原判 {tally(old[(g, r['qid'])]['verdict'] for r in mine)}")
    for r in sorted(new, key=lambda r: (r["group"], r["qid"])):
        o = old[(r["group"], r["qid"])]["verdict"]
        if o != r["verdict"]:
            print(f"  {r['group']:7s} {r['qid']:7s} 原 {o} -> 重判 {r['verdict']}: {r['comment'][:80]}")


# ---------- cross: 第三家评委 ----------

def cross_items() -> tuple[list[dict], dict[str, str]]:
    """抽 CROSS_N 题, 每题两份回答打乱标签, 附上 gold 和每份回答 [n] 引到的块. -> (每题一个判卷输入, {"qid|g1": 组})."""
    chunks = {c["id"]: c for c in load_jsonl(config.DATA_DIR / "processed" / "chunks-trans.jsonl")}
    qs = {q["id"]: q for q in load_jsonl(config.ROOT / "eval/questions-probe62.jsonl")}
    rows = {g: {r["qid"]: r for r in load_jsonl(RESULTS / f"{f}.jsonl")} for g, f in BACKENDS.items()}
    rng = random.Random(CROSS_SEED)
    pick = sorted(rng.sample(sorted(set(rows["deepseek"]) & set(rows["haiku"])), CROSS_N))
    src, trans = ai_judge.sources()
    table = E.load()
    items, mapping = [], {}
    for qid in pick:
        gs = list(BACKENDS)
        rng.shuffle(gs)
        answers = []
        for i, g in enumerate(gs):
            r = rows[g][qid]
            # 引到的编号: [3] [3, 5] 这类. 不认 CITE 里的范围写法 [3-5]: 这批回答里没有范围引用 ([18-0001] 这种照抄的块 id 也不该当编号)
            ns = sorted({int(n) for m in re.findall(r"\[([\d,，、\s]+)\]", r["reply"]) for n in re.findall(r"\d+", m)})
            seen = r["retrieved"]  # 第 n 块 = retrieved[n - 1]
            passages = {str(n): {"id": seen[n - 1], "orig": chunks[seen[n - 1]]["text"], "trans": chunks[seen[n - 1]].get("trans", "")}
                        for n in ns if 0 < n <= len(seen)}
            answers.append({"model": f"g{i + 1}", "reply": r["reply"], "year_checks": ai_judge.check_years(r["reply"], table),
                            "passages": passages})
            mapping[f"{qid}|g{i + 1}"] = g
        q = qs[qid]
        items.append({"qid": qid, "q": q["q"], "note": q.get("note", ""), **{k: q[k] for k in EXTRA if q.get(k)},
                      "gold": [{"id": x, "orig": src.get(x, ""), "trans": trans.get(x, "")} for x in q.get("gold", [])],
                      "answers": answers})
    return items, mapping


def calls_path(model: str) -> Path:
    """cross run 的原始调用记录 (一题一行: 判决, 用量, 费用)."""
    return ai_judge.JUDGE_DIR / f"cross-{model.split('/')[-1]}" / "calls.jsonl"


def cross_run(model: str = CROSS_MODEL, dry: bool = False) -> None:
    """逐题调用评委, 追加到 calls_path(model); 已有记录的题跳过. dry 只打印 prompt 的字符数."""
    system = (config.ROOT / "eval" / "judge_prompt.md").read_text(encoding="utf-8") + CROSS_RULES
    items, mapping = cross_items()
    users = [json.dumps(it, ensure_ascii=False, indent=1) for it in items]
    if dry:
        sizes = [len(system) + len(u) for u in users]
        print(f"{len(items)} 题, prompt 字符 平均 {sum(sizes) / len(sizes):.0f}, 最大 {max(sizes)}, 合计 {sum(sizes)}")
        return
    from openai import OpenAI
    client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    out = calls_path(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "map.json").write_text(json.dumps(mapping, indent=1), encoding="utf-8")
    done = {r["qid"] for r in load_jsonl(out)} if out.exists() else set()
    with open(out, "a", encoding="utf-8") as f:
        for it, u in zip(items, users):
            if it["qid"] in done:
                continue
            resp = client.chat.completions.create(model=model, reasoning_effort=CROSS_EFFORT,
                                                  messages=[{"role": "system", "content": system}, {"role": "user", "content": u}],
                                                  extra_body={"usage": {"include": True}})  # OpenRouter 在 usage 里给出费用
            text = resp.choices[0].message.content or ""
            cost = (resp.usage.model_extra or {}).get("cost") or 0
            m = re.search(r"\[.*\]", text, re.S)
            try:
                verdicts = json.loads(m.group(0)) if m else None
            except json.JSONDecodeError:
                verdicts = None
            f.write(json.dumps({"qid": it["qid"], "verdicts": verdicts, "raw": None if verdicts else text, "cost": cost,
                                "usage": {"in": resp.usage.prompt_tokens, "out": resp.usage.completion_tokens}},
                               ensure_ascii=False) + "\n")
            f.flush()
            print(it["qid"], "ok" if verdicts else "解析失败", f"${cost:.4f}", flush=True)


def cross_report(model: str = CROSS_MODEL) -> None:
    """调用记录按标签还原成组 -> eval/results/recheck-cross-<模型>.jsonl (一份回答一行, 带这一题那次调用的费用);
    打印和原评委判决 (AI 原判) 的一致数与每条分歧."""
    out = calls_path(model)
    mapping = json.loads((out.parent / "map.json").read_text(encoding="utf-8"))
    son = {(g, j["qid"]): j for g, f in BACKENDS.items() for j in load_jsonl(RESULTS / f"{f}.judged.jsonl")}
    new, cost, fails = [], 0.0, []
    for c in load_jsonl(out):
        cost += c["cost"]
        if not c["verdicts"]:
            fails.append(c["qid"])
            continue
        for v in c["verdicts"]:
            new.append({**v, "group": mapping[f"{c['qid']}|{v['model']}"], "blind_tag": v["model"], "judge": model,
                        "call_cost": c["cost"], "call_usage": c["usage"]})
    name = model.split("/")[-1]
    (RESULTS / f"recheck-cross-{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new),
                                                          encoding="utf-8")
    sv = lambda r: son[(r["group"], r["qid"])].get("verdict_ai") or son[(r["group"], r["qid"])]["verdict"]  # noqa: E731
    print(f"{len(new)} 份判决, 费用 ${cost:.4f}" + (f", 解析失败 {fails}" if fails else "")
          + f"; 和原评委一致 {sum(sv(r) == r['verdict'] for r in new)}")
    for g in BACKENDS:
        mine = [r for r in new if r["group"] == g]
        print(f"  {g:8s} 一致 {sum(sv(r) == r['verdict'] for r in mine)}/{len(mine)}"
              f"   {name} 对/部分/错 {tally(r['verdict'] for r in mine)}   原评委 {tally(sv(r) for r in mine)}")
    for r in sorted(new, key=lambda r: (r["group"], r["qid"])):
        if sv(r) != r["verdict"]:
            print(f"  {r['group']:8s} {r['qid']:7s} 原评委 {sv(r)} -> {name} {r['verdict']}: {r.get('comment', '')[:80]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="判卷抽查")
    ap.add_argument("check", choices=["nocite", "cross"])
    ap.add_argument("cmd", choices=["prep", "report", "run"])
    ap.add_argument("--dry", action="store_true", help="cross run: 只打印 prompt 字符数")
    ap.add_argument("--model", default=CROSS_MODEL, help=f"cross 的评委模型 (默认 {CROSS_MODEL})")
    a = ap.parse_args()
    if a.check == "nocite" and a.cmd in ("prep", "report"):
        nocite_prep() if a.cmd == "prep" else nocite_report()
    elif a.check == "cross" and a.cmd in ("run", "report"):
        cross_run(a.model, a.dry) if a.cmd == "run" else cross_report(a.model)
    else:
        sys.exit(f"{a.check} 没有 {a.cmd} 这个命令")
