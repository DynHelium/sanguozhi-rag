"""单次检索基线: 用原问题检索一次, 前 SEARCH_K 组资料一次性放进上下文, 模型只回答一次 (不给工具, 不自查).
和闭卷基线 (eval/run_closed.py), agent (eval/run_agent.py) 在同一批题上对照, 看 "多轮检索 + 代码介入" 比 "检索一次" 多帮了多少.
这只是评测基线, 产品里只有 agent 一条路径.

    uv run python -m eval.run_single --models deepseek-v4.1-flash --effort low \\
        --questions eval/questions-holdout.jsonl --name single-holdout
    uv run python -m eval.run_single ... --questions eval/questions-holdout.jsonl eval/questions-en.jsonl --name ss   # 几套题一个进程 -> ss-holdout.jsonl, ss-en.jsonl

为了只比较策略, 其他条件和 agent 一致:
    检索    rag/library.Library.search (白话索引 + 别名扩展 route2, 取 SEARCH_K 组), 也就是 agent 调一次 search 拿到的资料;
            查询固定是原问题
    资料    rag/library.Seen.add 编号后的文字 (出处行 + 原文 + 白话译文), 和 agent 看到的格式相同; 不带 "还有裴注" 的提示 (没有 read)
    prompt  SINGLE: rag/agent.SYSTEM 去掉只和工具有关的部分 (第 1-4 条的搜法, 第 12 条 "每个人分别搜, read 主段" 的搜法,
            第 13 条 "继续搜或 read", 第 15 条 "译回中文再搜"), 其余作答规则 (事实 / 推断 / 背景三分, 引原文标编号, 演义题的说法,
            养子, 能推出结论就推, 英文提问用英文答) 和举例照搬. 第 5 条 "公元年只能来自 era" 改为 "资料里没有就不写": 给模型看的原文和
            白话都不带公元年, 所以单次基线基本不写公元年; 和 agent 的差距里包含 era 工具这一项
    核对    引文 (rag/quotes.verify_quotes) 和编号 (check_cites) 都对照这 SEARCH_K 组资料, 和 agent 对照它看过的块一样

结果行是 eval/run_agent.py 结果行的子集 (rounds 只有一次 search, calls = 1; 没有 gold_in_first, prompt_tokens, draft, review_raw 和
自查相关的配置字段), 可以直接用 eval/ai_judge.py 判卷, run_agent --summary 汇总. system_sha 是 SINGLE 全文的哈希, 改过 prompt 能分清.
--no-context: 消融, 不检索, 资料一栏写 "(没有资料)", prompt 不变 (和闭卷基线不同, 规则和本基线完全一样, 只差检索).
续跑: 重跑时跳过已有且没出错的 (模型, 题); 跑完后同一 (模型, 题) 只留最后一行.
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.common import OUT_DIR, is_refusal, out_paths
from eval.run_agent import _sha, coverage
from eval.run_retrieval import QUESTIONS_PATH, load_jsonl
from rag import llm
from rag.library import Seen, load_library
from rag.calls import PER_MODEL, make_complete, setup
from rag.quotes import check_cites, extract_quotes, verify_quotes

SINGLE = """你是一位讲解三国历史的老师, 依据《三国志》(陈寿正文和裴松之注) 回答提问. 用户消息里附有按问题检索到的资料, 每条以 [n] 编号; 只能依据这些资料回答.

做法:
1. 公元年: 资料里没有写的公元年不要写, 不要凭记忆换算. 也不要凭记忆补资料里没有的具体事实 (身份, 官职, 情节, 出处, 某个词当时用不用). 第 2 条的背景段只能写那里列出的三种句子, 其他凭记忆的内容一律不写.
2. 回答里的内容分三类, 来源不同, 要分开写:
   - 事实 (人名, 年份, 官职, 谁做了什么, 书名作者): 只能来自资料里的原文, 引原文并标编号 (第 3 条). 一件事的时间只写原文记在这件事上的年月; 原文没给这件事写时间的, 不要从前后段落推定, 也不要把相关的另一件事的时间安到它身上 (如进攻某城的月份不等于起兵的月份); 要推就按推断写. 资料里找不到的, 说《三国志》里没有找到相关记载; 不把记忆里的事实当成资料写.
   - 推断 (先后, 因果, 对比, 能不能, 是否可能): 可以做, 但只能从你已经引出的事实推, 写明 "据此推断", 并把推理过程写出来 (先列依据的几条事实, 再说由此可知什么). 不要越过证据: 资料只说 A 在 B 之前, 就不说 A 导致 B; 没有记载不等于没有发生; 几种说法不一致 (如正文和裴注年份不同) 时, 分别考虑每种说法, 不挑一种当定论.
   - 背景段 (可选, 只在问题里的说法在资料里查不到或和资料不同, 或问到今天的地名, 官职职能时才写): 放在回答最后, 以 "以下不是《三国志》的记载:" 开头, 不加「」, 不标编号. 只能是下面三种句子, 每种一两句:
     (a) "问题里说的某某 (情节, 称号) 《三国志》里没有记载 (或记载不同), 可能出自《三国演义》或其他非正史的说法." 只用于问题里提到, 而资料里确认没有或不同的说法; 问题里的说法和资料不冲突时不写这一句, 只陈述原文. 不要把原文里有的事说成演义的说法;
     (b) "某地 (当时的地名) 在今天的某省某地一带.";
     (c) "某官职 (泛指这个官) 负责什么.";
     背景段里不能出现具体人物做了什么, 任过什么官, 哪一年, 当时有没有某种说法: 这些是史实, 要么在正文里引原文回答, 要么不写.
   - 转述 "谁做了什么" 要和原文的主语一致; 拿不准的地方直接引原文, 不要改写. 提到书名和作者, 照资料里写的 (如 "袁暐《献帝春秋》"), 不要换成别的书或别的作者. 原文里 "初" 开头的是追述以前的事.
3. 引用时只引原文 (文言), 不要引白话译文; 一字不改放在「」里, 后面用方括号里的数字标出处, 如「太祖乃變易姓名，間行東歸」[3]. 数字是资料标题开头的 [n]; 不要写资料标题括号里的 id (如 01-0004), 回答正文里也不要出现这种 id. 不要在「」里用省略号跳过内容, 不要把卷名, 人名, 书名单独放进「」.
4. 裴注要说明引自哪本书 (如 "裴注引《魏書》"); "臣松之以為" "臣松之案" 开头的是裴松之自己的按语, 说成 "裴松之按", 不要算到他引的书上. 几种说法不一致时并列说明, 不替读者裁决.
5. 问题出自《三国演义》而正史不同的, 照常回答, 说明 "这是演义的说法, 正史记载为……". 问题里的人物, 称号或情节在资料里查不到, 或者和原文对不上时, 想一想它是不是出自《三国演义》或戏曲 (如 "桃园结义" "三英战吕布"): 是的话要明说 "这是《三国演义》的说法, 《三国志》里没有这件事 (或: 记载为……)", 不要只说 "资料里没找到". 问题的说法查不到, 但资料里有能回答它的事实时 (如问两人是否见过面, 资料里有两人的生卒和行踪), 先按第 2 条的推断给出结论, 再在背景段说明这个说法出自哪里. 问题里的说法 (预设) 和原文不完全一致的 (如问 "谁是总指挥", 原文是两人分任左右督), 照原文说明, 不要顺着问题的说法下结论.
6. 只回答三国 (东汉末到西晋初) 的历史问题. 与此无关的请求, 一句话说明你只讲三国历史.
7. 每条资料都有《三国志》原文 (文言繁体) 和【白话译文】. 先看白话弄懂意思, 再对照原文核实; 两者有出入时以原文为准 (译文可能有错).
8. 问儿子, 子女时, 养子 (原文 "養子" "以為子") 也要列出并注明.
9. 作答前自查: 问题要的每一项都在资料里找到原文了吗? 确实找不到的明说没找到, 不要用记忆补事实. 查到的事实能不能回答问题? 能用时间先后或对比得出结论的, 不要停在 "资料里没有直接记载", 要把结论推出来并标明是推断.
10. 无论资料多少, 一定要给出文字回答, 不能空着.
11. 用户用英文提问时: 回答用英文, 人名用汉语拼音, 官职用通行英译 (Chancellor, Grand Commandant, Inspector, Administrator, Colonel); 引文照样引原文 (文言) 并在后面附英文大意.
中文提问用简体中文回答, 英文提问用英文回答; 先给结论 (包括推断出的结论), 再给依据, 背景段放在最后."""


def main() -> None:
    ap = argparse.ArgumentParser(description="单次检索基线")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--questions", type=Path, nargs="+", default=[QUESTIONS_PATH], help="题目文件, 可以给几个 (见 eval.common.out_paths)")
    ap.add_argument("--jobs", type=int, default=0, help="同时跑几题 (默认 PER_MODEL x 模型数)")
    ap.add_argument("--no-context", action="store_true",
                    help="消融: 不检索, 资料一栏写 (没有资料); prompt 不变, 和本基线对照只差检索")
    ap.add_argument("--name", required=True)
    ap.add_argument("--effort")
    args = ap.parse_args()
    args.models = [setup(m) for m in args.models]  # 检查 key, 短名补全, 模型和 .env 的 LLM_BACKEND 对得上
    outs = out_paths(args.name, args.questions)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lib = None if args.no_context else load_library("trans")
    sems = {m: threading.Semaphore(PER_MODEL) for m in args.models}
    lock = threading.Lock()

    def one(job: tuple[str, dict, Path]) -> None:
        m, q, out = job
        gold, equiv, t = q.get("gold", []), q.get("equiv", {}), time.time()
        complete = make_complete(m, sems[m])
        extra = {"reasoning_effort": args.effort} if args.effort else {}
        seen = Seen()
        got = [] if args.no_context else lib.search(q["q"])
        context = seen.add(got) if got else "(没有资料)"
        user = f"资料:\n\n{context}\n\n问题: {q['q']}"
        try:
            r = complete([{"role": "system", "content": SINGLE}, {"role": "user", "content": user}], model=m, **extra)
            reply, err = r.choices[0].message.content or "", None
        except Exception as e:
            reply, err = "", f"{type(e).__name__}: {e}"[:300]
        quotes = verify_quotes(extract_quotes(reply), seen.chunks)
        row = {"model": m.split("/")[-1], "model_full": m, "qid": q["id"], "type": q.get("type", "?"), "q": q["q"], "gold": gold,
               "retrieved": [c["id"] for c in seen.chunks], **coverage(seen.chunks, gold, equiv),
               "reply": reply, "quotes": [list(x) for x in quotes], "cites": check_cites(reply, seen.chunks),
               "refused": is_refusal(reply), "sec": round(time.time() - t, 1),
               "rounds": [{"tool": "search", "args": {"query": q["q"]}, "chunks": [c["id"] for c in got], "error": None, "by": "code"}],
               "calls": 1, "usage": complete.stats, "prompt": "SINGLE-nocontext" if args.no_context else "SINGLE", "system_sha": _sha(SINGLE), "effort": args.effort, "index": "trans"}
        if err:
            row["error"] = err
        with lock, open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{row['model']} {q['id']} {row['sec']}s 看到{'✓' if row['gold_in_ctx'] else '✗'} 引文 ✓{sum(1 for _, c in quotes if c)}"
              f"✗{sum(1 for _, c in quotes if not c)}" + (f" 错误 {err}" if err else ""), file=sys.stderr, flush=True)

    jobs = []
    for qp, out in zip(args.questions, outs):
        done = {(r["model"], r["qid"]) for r in load_jsonl(out) if not r.get("error")} if out.exists() else set()
        jobs += [(m, q, out) for q in load_jsonl(qp) for m in args.models if (m.split("/")[-1], q["id"]) not in done]
    llm.client()  # 先在主线程建好客户端, 同 eval/run_agent.py
    with ThreadPoolExecutor(args.jobs or PER_MODEL * len(args.models)) as ex:
        list(ex.map(one, jobs))
    for out in outs:  # 同一 (模型, 题) 重跑过的, 只留最后一条
        if out.exists():
            keep = {(r["model"], r["qid"]): r for r in load_jsonl(out)}
            out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep.values()), encoding="utf-8")


if __name__ == "__main__":
    main()
