"""RAG 效果评测的判卷 (标准见 eval/judge_prompt_v2.md): 打包 -> 判卷 agent 按事实清单逐项标 -> 代码算判定, 还原到各组.

    uv run python -m eval.judge_v2 prep rag-ds          # 打包 DeepSeek 五组的回答 -> data/processed/eval_runs/judge/rag-ds/
    (判卷 agent 按 eval/judge_prompt_v2.md 逐包判, 每包写 batchNN.out.jsonl)
    uv run python -m eval.judge_v2 merge rag-ds         # -> data/processed/eval_runs/answers/rag-ds.judged.jsonl, 打印各组 对 / 部分 / 错

<前缀> 是 eval/run_rag.py 的 --name 去掉组名的部分: 读 <前缀>-<组>-<题集>.jsonl (组见 GROUPS, 题集见 SETS).
判的是闭卷, 静态 RAG, oracle, 只给工具, 生产配置五组; harness 组 (run_rag 的 harness) 不判: 作者审卷时发现它的回答常在回应
"再查" 提示 ("我按您的要求重新核实了"), 改用生产配置组代替.
同一题同一语言的五组回答放进一个包 (一包 PER_BATCH 题), 打包前统一格式 (normalize), 每题的标签 g1..g5 随机分配,
对应表写在包目录的 map.json. 题目带上事实清单 (CHECKLISTS, 英文题 e-<id> 用中文题 <id> 的清单) 和相关原文.

判定 (verdict) 由代码按清单的标记算, 评委不写:
    错    有 core 项不是 yes (no / wrong / hedged), 或 must_not 有 yes;
          例外: 标了 "core_partial": true 的 core 项只是没答到 (no / hedged) 时算部分, 说错 (wrong) 仍算错
          (用于题目自带判法写了 "只答到 X 算部分对" 的题: i-02, f-22, p-01)
    部分  不是错, 但有: 题目要求的 (asked) 非 core 项是 no / wrong / hedged; core_partial 项没答到;
          背景细节项 (asked 为 false) 说错了 (wrong); 编造的史料 (fabricated) 或清单外的事实错误 (extra_errors) 不为空
    对    其余. 背景细节项没写 (no) 不扣分; 查不到的确定说法 (unsupported) 只记录.
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from eval.ai_judge import JUDGE_DIR, sources
from eval.common import OUT_DIR
from eval.run_retrieval import load_jsonl
from rag import config

SETS = ("rag50", "rag50-en")
QUESTIONS = {s: config.ROOT / "eval" / f"questions-{s}.jsonl" for s in SETS}
CHECKLISTS = config.ROOT / "eval" / "checklists-rag50.jsonl"
PER_BATCH = 2
GROUPS = ("closed", "static", "oracle", "agentic", "production")  # 判卷的五组 (见模块说明)
CITE = re.compile(r"\s?\[\d+(?:\s*[,，、\-–]\s*\d+)*\]")              # [3] [3, 5] [3-5]
QUOTES = str.maketrans({"「": "“", "」": "”", "『": "‘", "』": "’"})  # 引号统一成 “” ‘’


def normalize(reply: str) -> str:
    """去掉 [n] 编号, 「」『』 换成 “” ‘’: 引文和编号不计分, 也不让评委凭格式认出组别."""
    return CITE.sub("", reply).translate(QUOTES)


def base_id(qid: str) -> str:
    """英文题 e-h-01 -> 中文题 h-01 (共用事实清单)."""
    return qid[2:] if qid.startswith("e-") else qid


def verdict(marks: dict, checklist: dict) -> str:
    """按清单标记算判定 (规则见模块说明). 标记缺项时当作 no."""
    must = marks.get("must", {})
    core = [m for m in checklist["must"] if m.get("core")]
    if any(v == "yes" for v in marks.get("must_not", {}).values()):
        return "错"
    missed = [m for m in core if must.get(m["id"]) != "yes"]
    if any(not m.get("core_partial") or must.get(m["id"]) == "wrong" for m in missed):
        return "错"
    rest = [m for m in checklist["must"] if not m.get("core")]
    if (missed or marks.get("fabricated") or marks.get("extra_errors")
            or any(must.get(m["id"]) != "yes" for m in rest if m.get("asked", True))
            or any(must.get(m["id"]) == "wrong" for m in rest if not m.get("asked", True))):
        return "部分"
    return "对"


def prep(prefix: str) -> None:
    """打包 (见模块说明); 先清掉旧包. 缺回答的 (组, 题) 跳过并打印."""
    checks = {r["id"]: r for r in load_jsonl(CHECKLISTS)}
    src, trans = sources()
    rng = random.Random(f"judge-v2-{prefix}")
    items, mapping, missing = [], {}, []
    for s in SETS:
        qs = {q["id"]: q for q in load_jsonl(QUESTIONS[s])}
        by: dict[str, dict] = {}
        for g in GROUPS:
            p = OUT_DIR / f"{prefix}-{g}-{s}.jsonl"
            for r in (load_jsonl(p) if p.exists() else []):
                if not r.get("error"):
                    by.setdefault(r["qid"], {})[g] = r
        for qid in sorted(by):
            q, ck = qs[qid], checks[base_id(qid)]
            gs = list(by[qid])
            missing += [f"{g}|{qid}" for g in GROUPS if g not in gs]
            rng.shuffle(gs)
            answers = []
            for i, g in enumerate(gs):
                mapping[f"{qid}|g{i + 1}"] = g
                answers.append({"model": f"g{i + 1}", "reply": normalize(by[qid][g]["reply"])})
            items.append({"qid": qid, "q": q["q"], "gold": [{"id": x, "orig": src.get(x, ""), "trans": trans.get(x, "")} for x in q.get("gold", [])],
                          "checklist": ck["checklist"], "related": ck.get("related", []), "answers": answers})
    d = JUDGE_DIR / prefix
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("batch*.json"):
        f.unlink()
    for i in range(0, len(items), PER_BATCH):
        (d / f"batch{i // PER_BATCH:02d}.json").write_text(json.dumps(items[i:i + PER_BATCH], ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "map.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(items)} 题 {sum(len(x['answers']) for x in items)} 份回答 -> {d}, {(len(items) + PER_BATCH - 1) // PER_BATCH} 包"
          + (f"; 缺回答 {len(missing)}: {missing[:10]}" if missing else ""))


def merge(prefix: str) -> None:
    """各包判决 + 代码算的判定, 按标签还原组 -> OUT_DIR/<前缀>.judged.jsonl; 打印各语言各组 对 / 部分 / 错."""
    d = JUDGE_DIR / prefix
    mapping = json.loads((d / "map.json").read_text(encoding="utf-8"))
    checks = {r["id"]: r for r in load_jsonl(CHECKLISTS)}
    rows = []
    for f in sorted(d.glob("batch*.out.jsonl")):
        for j in load_jsonl(f):
            ck = checks[base_id(j["qid"])]["checklist"]
            rows.append({**j, "group": mapping[f"{j['qid']}|{j['model']}"], "blind_tag": j["model"],
                         "lang": "en" if j["qid"].startswith("e-") else "zh", "verdict": verdict(j, ck)})
    want = set(mapping)
    got = {f"{r['qid']}|{r['blind_tag']}" for r in rows}
    (OUT_DIR / f"{prefix}.judged.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} 份判决 -> {OUT_DIR / f'{prefix}.judged.jsonl'}" + (f"; 没判的 {len(want - got)}" if want - got else ""))
    for lang in ("zh", "en"):
        for g in GROUPS:
            vs = [r for r in rows if r["lang"] == lang and r["group"] == g]
            if vs:
                c = Counter(r["verdict"] for r in vs)
                print(f"  {lang} {g:8s} 对 {c['对']:2d}  部分 {c['部分']:2d}  错 {c['错']:2d}"
                      f"   清单外错误 {sum(len(r.get('extra_errors', [])) for r in vs)}  拒答 {sum(bool(r.get('refused')) for r in vs)}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("prep", "merge"):
        sys.exit("用法: python -m eval.judge_v2 prep|merge <前缀>")
    (prep if sys.argv[1] == "prep" else merge)(sys.argv[2])
