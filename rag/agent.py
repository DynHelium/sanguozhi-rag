"""agent: 模型自己调用工具 (search / read / era / who) 查《三国志》再作答; 代码负责循环, 轮数上限, 作答后的检查.

    uv run python -m rag.agent "曹操因为什么逃离洛阳？"            # 用 .env 的后端和模型, 打印每次工具调用和回答
    uv run python -m rag.agent "问题" --model deepseek-v4.1-flash --effort low   # 指定模型 (后端看 .env 的 LLM_BACKEND)
    uv run python -m rag.agent "问题" --index orig              # 改搜原文索引 (对照用)
    uv run python -m tests.test_agent                              # 假模型驱动的测试, 不连网

代码不先检索: 搜什么, 读什么, 年号怎么换算, 全部由模型调用工具决定, 这样模型能顺着资料里出现的人名, 年号继续查.
用 OpenAI 标准 tools; 轮数上限由 rag/loop.agent_loop 保证 (最后一次请求不传 tools), 因为 tool_choice="none" 实测会被无视.

本模块的几层 (循环本身在 rag/loop.py, 工具背后的数据查询在 rag/library.py):
    run_tool       把模型的一个 tool_call 分发给 Library, 返回给模型的文字和日志
    make_check     循环结束后: 查得太少就作答时追加提示再跑一次 (nudge); 说没找到时要求换说法重搜 (nudge_notfound, 默认关)
    after_answer   作答后: 坏回答重答 (retry), 自查 (review), 引文核对不过的回给模型改 (quote_fix), 改正错编号 (cite_fix)
    run_agent      组 messages, 串起上面几层, 返回回答和全过程日志
OFF 里的名字关掉对应的代码介入, 消融实验用.

模型看到的每一块有一个全局编号 [n] (rag/library.Seen), 回答里的 [n] 指 seen.chunks[n-1]. 默认检索白话索引,
每块给模型看原文和白话译文; 引文只认原文.

SYSTEM_SMALL (模型名含 SMALL_MODELS 里的字样时由 system_for 选用) 和 run_agent 里遇到 "no user query" 报错的重试,
是早期在本地小模型上实验时留下的; 当前固定使用的 DeepSeek V4.1 Flash 用 SYSTEM.
"""
import argparse
import json
import re
from typing import Any, Callable

from rag import llm
from rag.alias import find_pinyin_names
from rag.library import ALIAS, READ_MAX, SEARCH_K, Library, Seen, load_library  # noqa: F401 (重新导出, 外部可用 agent.SEARCH_K 等)
from rag.loop import FINAL_TURN, MAX_ROUNDS, agent_loop
from rag.quotes import check_cites, extract_quotes, fix_cites, renumber_cites, verify_quotes

MIN_SEARCHES = 2  # search + read 合计不到 2 次就想作答时, 代码追加一句提示让它再查 (只提示一次, 见 make_check).
MIN_SEARCHES_AGG = 3  # 汇总题 (AGGREGATE: 有哪些, 有几个, ...) 的门槛 (agg_nudge): 漏人多半是只搜了一两次
AGGREGATE = re.compile(r"有哪些|有几个|几个人|哪几|哪些人|都有谁|分别|各自|how many|who were|which (?:people|men|ones)", re.I)
                  # 小模型常常搜 1 次就答, 只在 prompt 里写 "至少搜两次" 不管用. search 和 who 一次都没调的不提示 (闲聊/拒答); 只调了 who 就作答的也提示.
NUDGE = ("你只查了 {n} 次. 你看过的资料出自: {where}. 回答前请再查一次, 确认没有漏掉: "
         "资料里提到的相关人物, 用他们的全名分别 search (如问某人的儿子, 就对资料里出现的每个儿子的名字各搜一次); "
         "或用 read 读相关传记的上下文和裴注. 不要只换一个更泛的词重搜.")


NOT_FOUND = re.compile(r"(没有|未|没)(能)?(找到|查到|检索到|记载|说|提到|提及|写到|载)|查不到|找不到|无(此)?记载|未见")
_REFUSAL_TEMPLATE = re.compile(r"人工智能|语言模型|无法回答|换个话题|我还没学习")
NOT_FOUND_NUDGE = ("你的回答里说有内容没找到. 答案常常就在资料里, 只是用词不同: 请再换 2~3 种说法 search (同义词, 原文的文言用词, "
                   "相关的人物, 地点, 官职或称号, 如 \"刺杀\" 换成 \"刺 刃 杀害\", \"皇帝\" 换成具体的帝号), 必要时 read 最相关的段落, "
                   "然后再回答. 确实找不到再说没找到.")
LANG_NOTE = "Always reply in the language the user used."  # lang_note (默认开): 问题后面和自查指令末尾各追加一次.
# 资料是中文 (文言原文 + 白话), 英文提问时上下文里中文多得多, 有的模型就用中文回答; system prompt 末行的语言要求离作答太远.
SCOPE_REFUSAL = re.compile(r"只讲三国|只回答三国|不在.{0,8}范围|与三国(历史)?无关|不属于三国")  # "这不是三国的问题" 一类的拒答
SCOPE_NUDGE = ("你没有查资料就判断这个问题和三国无关. 有些问题只描述事件, 不写人名 (如 \"哪个皇帝被刺死了\" \"谁在船上称象\"), "
               "其实是三国时期的人和事. 请先用问题里的事件词 search 一次; 查到相关记载就照常回答, 查不到再说明你只讲三国历史.")
REVIEW = True  # 作答后让模型自查一遍 (指令按 REVIEW_MODE 选) (多一次请求); run_agent(review=...) 可以单独关
# 默认关的两个: nudge_notfound 见下; nudge_scope 的作用已写进 SYSTEM 第 10 条 (看不出是不是三国的事先搜), 打开时对真正无关的问题也会逼模型搜一次.
OFF: set[str] = {"nudge_notfound", "nudge_scope"}  # 关掉的代码介入, 可选 "nudge" "nudge_notfound" "nudge_scope" "agg_nudge" "pinyin_who" "lang_note" "review" "quote_fix" "cite_fix" (见 make_check / after_answer / run_agent).
    # 消融实验时由评测脚本覆盖. 默认关 "nudge_notfound": 消融评测里看不出作用, 且正则会误伤正当的 "正史未记载"
REVIEW_TEXT = ("请逐句核对你刚才的回答, 只能用上面工具返回的资料: "
               "1. 每个事实 (人名, 谁做了什么, 年份和公元年, 官职, 先后顺序, 因果, 书名和作者) 都要在资料里有原文依据; "
               "没有依据的句子删掉 (包括背景知识, 后世说法, \"通常认为\" 之类的名单, 自己推断的因果); 和资料不符的改成资料的说法. "
               "2. 问题要的每一项都答到了吗? 汇总题 (有哪些/有几个) 要把资料里所有符合条件的都列出, 包括裴注里顺带提到的 "
               "(如 \"某某的弟弟某某\" 也是同一父亲的儿子; 问儿子时养子也列出并注明是养子); 资料里有却漏了的补上, 资料里没有的明说没找到. "
               "3. 「」里的引文必须一字不差来自原文. "
               "直接输出修改后的完整回答 (格式同前), 不要说明改了什么, 不要调用工具.")
# 自查的做法 (REVIEW_MODE), 选用哪段自查指令:
#   "rewrite"  一律输出改后的完整回答 (REVIEW_TEXT); 配合只依据原文, 不推断不补背景的 prompt, 如 SYSTEM_SMALL.
#   "short"    没有问题只回 REVIEW_OK, 有问题才输出完整回答 (REVIEW_TEXT_SHORT); 大多数回答不用改, 省掉重写整篇的时间.
#   "layered"  默认. 配合分层的 system prompt (事实 / 推断 / 背景分开写, 即 SYSTEM), 分三类核对, 一律输出改后的完整回答
#              (REVIEW_TEXT_LAYERED); REVIEW_TEXT 会一律删掉推断和背景, 把分层 prompt 允许写的内容删光.
#   "patch"    核对标准同 layered, 但不重写整篇, 只列出要改的句子 ("原文 / 改为"), 由 apply_patch 逐条替换 (REVIEW_TEXT_PATCH).
#              重写整篇要吐出约 3000 token, 是一题里最慢的一步; 只列改动时输出短得多. 抄的原文在回答里找不到的那条跳过.
REVIEW_MODE = "layered"
REVIEW_EFFORT: str | None = None  # 作答之后几步 (自查, 改引文) 的 reasoning_effort; None = 和作答相同. 自查的时间大多花在推理上, 所以单独设: 调低它能缩短每题用时 (eval.bench_latency --review-effort 对比时间, 再跑留出题看正确率)
REVIEW_OK = "无误"
REVIEW_TEXT_SHORT = REVIEW_TEXT.replace(
    "直接输出修改后的完整回答 (格式同前), 不要说明改了什么, 不要调用工具.",
    f"如果三项都没有问题, 只输出两个字: {REVIEW_OK} (不要输出别的). "
    "如果有要改的, 直接输出修改后的完整回答 (格式同前), 不要说明改了什么, 不要调用工具.")
_REVIEW_OK_RE = re.compile(rf"^\W*{REVIEW_OK}\W*$")
REVIEW_TEXT_LAYERED = ("请逐句核对你刚才的回答, 只能用上面工具返回的资料. 回答里的内容分三类, 分别核对: "
                       "1. 事实 (人名, 谁做了什么, 年份和公元年, 官职, 书名和作者): 每一条都要在资料里有原文依据, 时间要是原文记在这件事上的时间; "
                       "没有依据的删掉, 和资料不符的改成资料的说法. 写在背景段以外, 又没有标明是推断的内容, 都按事实核对. "
                       "2. 推断 (写明 \"据此推断\" 的部分): 保留, 但检查它依据的事实是否都引出了, 结论是否越过了证据 "
                       "(只有先后不能说成因果, 没有记载不能说成没有发生); 越过的改成证据能支持的说法. "
                       "该推断却停在 \"没有直接记载\" 的, 把结论推出来并标明是推断. "
                       "3. 背景段 (以 \"以下不是《三国志》的记载:\" 开头的最后一段): 逐句看, 只保留三种句子: 问题里提到、原文确实没有或记载不同的说法可能出自《三国演义》或其他非正史 (说成演义的事如果资料里有记载, 这句删掉, 事实放回正文); "
                       "某地在今天哪里; 某官职负责什么. 句子里出现具体人物做了什么, 任过什么官, 哪一年, 当时有没有某种说法的, 整句删掉 "
                       "(这些是史实, 资料里有的放回正文并引原文); 不需要背景的不要加. "
                       "另外: 问题要的每一项都答到了吗? 资料里有却漏了的补上. 「」里的引文必须一字不差来自原文, "
                       "出处只用方括号里的数字, 不要出现资料的 id. "
                       "直接输出修改后的完整回答 (格式同前), 不要说明改了什么, 不要调用工具.")
REVIEW_TEXT_PATCH = REVIEW_TEXT_LAYERED.replace(
    "直接输出修改后的完整回答 (格式同前), 不要说明改了什么, 不要调用工具.",
    "不要重写整篇回答, 只列出要改的地方, 每处写两行:\n"
    "原文: 回答里要改的那一句, 从回答里一字不差地抄下来\n"
    "改为: 改好的句子本身 (整句删掉就只写 删除; 要补漏的内容, 就把它接在抄下来的那句后面一起写)\n"
    "只列出真要改的句子, 没问题的句子不要列出; \"改为\" 后面只写会出现在回答里的文字, 不写 \"无误\" \"保留\" 或括号里的理由. "
    f"几处就写几组, 不要写别的说明, 不要调用工具. 三类都没有问题时只输出两个字: {REVIEW_OK}")
_KEEP = re.compile(r"^[(（]?\s*(?:无误|保留|不变|不改|不用改|不需要改)")  # "改为" 写的是 "不用改": 跳过这条
_NOTE = re.compile(r"\s*[(（][^()（）]*(?:无误|保留|此句|删去|删除|核对|移至|不改)[^()（）]*[)）]\s*$")  # 句末括号里的自查说明
_PATCH = re.compile(r"原文[:：][ \t]*(.*?)[ \t]*\n\s*改为[:：][ \t]*(.*?)(?=\n\s*原文[:：]|\Z)", re.S)


def apply_patch(reply: str, patch: str) -> tuple[str, int, int]:
    """把 patch 模式自查的输出 ("原文: …\n改为: …" 若干组) 套到回答上, 返回 (新回答, 套上的条数, 没找到原文的条数).

    原文在回答里逐字出现才替换 (只换第一处); "改为" 以 "删除" 开头时删掉这句 (后面常跟着括号里的理由), 删完留下的多余空行合并.
    原文为空或找不到的那条跳过, 回答保持原样 (宁可不改, 也不改错地方). 模型常把自查的话写进 "改为": "改为" 是 "无误" "保留"
    之类 (_KEEP) 的跳过; 句末括号里的自查说明 (_NOTE, 如 "(事实, 依据 [1], 无误)") 去掉再替换; 删句后留下的空 "****" 行删掉."""
    done = missed = 0
    for old, new in _PATCH.findall(patch or ""):
        old, new = old.strip(), new.strip()
        if _KEEP.match(new):
            continue
        new = _NOTE.sub("", new)
        if not new or new == old:
            continue
        if not old or old not in reply:
            missed += 1
            continue
        reply = reply.replace(old, "" if new.startswith("删除") else new, 1)  # 模型常在 "删除" 后面加括号说明理由
        done += 1
    reply = re.sub(r"^\s*\*\*\*\*\s*$", "", reply, flags=re.M)  # 删掉粗体标题里的整句后留下的空 "****"
    return re.sub(r"\n{3,}", "\n\n", reply).strip(), done, missed


QUOTE_FIX = ("下面这些「」里的引文, 在你查到的资料原文里找不到 (可能有字记错, 或用省略号跳过了内容, 或引了白话译文):\n{quotes}\n"
             "请把它们改成资料原文里一字不差的句子; 找不到对应原文的, 去掉「」改成转述 (转述也要和原文一致). "
             "其他内容不要改. 直接输出修改后的完整回答, 不要说明改了什么, 不要调用工具.")
# 模型 (如 deepseek) 混在文字里的工具调用标记: 从 <｜...tool...> 起, 到 </｜...tool_calls...> 或文末为止
_MARKUP = re.compile(r"<[｜|][^>]*tool[^>]*>.*?(?:</[｜|][^>]*tool_calls[^>]*>|$)", re.S)


def strip_markup(reply: str) -> str:
    """去掉混在文字里的工具调用标记 (deepseek 的 <｜DSML｜tool_calls> 块)."""
    return _MARKUP.sub("", reply).strip()


def bad_reply(reply: str, finish_reason: str | None) -> bool:
    """回答不能用: 被截断 (finish_reason "length"), 带工具调用标记, 或去掉标记后为空.
    不按字数判: 正常的拒答也很短."""
    return finish_reason == "length" or bool(_MARKUP.search(reply)) or not strip_markup(reply)


def nudge_text(n_look: int, chunks: list[dict]) -> str:
    """提示语 NUDGE: 列出模型看过的块来自哪些传 (卷名·最后一级小标题, 去重, 最多 8 个), 给出具体的再查方向.
    n_look 是 search + read 的次数. 不用笼统的 "换个说法再搜": 小模型会去搜更泛的词.
        nudge_text(1, [<01-0002 的块>])  -> "你只查了 1 次. 你看过的资料出自: 魏書•武帝紀. 回答前请再查一次, ..." """
    where = []
    for c in chunks:
        w = c.get("juan_title", "") + ("·" + c["headings"][-1] if c.get("headings") else "")
        if w and w not in where:
            where.append(w)
    return NUDGE.format(n=n_look, where="; ".join(where[:8]) or "(无)")


# 默认 system prompt, 按白话索引和 TOOLS 写的; 评测时可以用 run_agent(system=...) 换成其他版本对照.
SYSTEM = """你是一位讲解三国历史的老师, 依据《三国志》(陈寿正文和裴松之注) 回答提问. 你手边有四个工具: search 检索原文, read 读某一块原文的前后文, era 把年号换算成公元年, who 查人 (一个名字或称号指谁, 有几个同名的人).

做法:
1. 回答任何三国历史问题之前, 必须先用 search 查资料, 至少换两种说法搜两次. 资料库按白话译文检索: 检索词用现代汉语, 人名写全名 (曹操, 刘备, 孙权), 加上事件关键词, 如 "曹操 去世 年龄" "荀彧 儿子 官职". 一次没找到就换说法再搜: 换成相关人物, 地名, 官职, 或人物的字; 译文里的说法常和问题不同 (洛阳 → 京城, 京都; 逃离 → 逃回, 东归, 改换姓名; 被杀 → 遇害, 被刺), 要换着词搜. 问题没给人名的 (如 "哪个皇帝被刺死了"), 先用问题里的事件词搜 ("皇帝 被刺 杀死"), 不要先猜一个人名去搜.
2. 问题里的人只给了称号 (先主, 宣王, 魏王, 明帝), 或问 "有几个X" "X是谁" 时, 先用 who 查: 它告诉你这个称呼指谁, 有几个同名的人, 各自的传在哪一段 (id 可以直接 read), 以及这个人在书里的其他称呼 (译文里常用称呼而不是名字, 如 魏太祖, 先主, 拿称呼去 search 更容易命中). 同名的人要分别查.
3. 原文常用简称和省略. 遇到 "帝" "公" "太祖" "先主" 等不确定指谁, 或 "七年" "明年" 这种没写年号的年份, 用 read 往前读, 直到弄清楚, 不要猜.
4. 资料标题里写 "裴注" 的块 (id 带 n, 如 46-0016n0) 是给正文 46-0016 作的注. 搜到裴注, 却没看到它所注的那段正文时, 对这块裴注调用 read, 会附上它注的正文: 事情的主干 (谁, 哪一年, 结果) 常常在正文里, 不要只凭注文作答.
5. 公元年只能来自 era 工具: 没有调用 era 就不写公元年; 白话译文里也没有公元年. 不要凭记忆写公元年, 也不要凭记忆补资料里没有的具体事实 (身份, 官职, 情节, 出处, 某个词当时用不用). 第 6 条的背景段只能写那里列出的三种句子, 其他凭记忆的内容一律不写.
6. 回答里的内容分三类, 来源不同, 要分开写:
   - 事实 (人名, 年份, 官职, 谁做了什么, 书名作者): 只能来自你查到的原文, 引原文并标编号 (第 7 条). 一件事的时间只写原文记在这件事上的年月; 原文没给这件事写时间的, 不要从前后段落推定, 也不要把相关的另一件事的时间安到它身上 (如进攻某城的月份不等于起兵的月份); 要推就按推断写. 资料是你自己检索到的, 不是用户提供的. 换了几种说法都找不到, 才说《三国志》里没有找到相关记载; 不把记忆里的事实当成资料写.
   - 推断 (先后, 因果, 对比, 能不能, 是否可能): 可以做, 但只能从你已经引出的事实推, 写明 "据此推断", 并把推理过程写出来 (先列依据的几条事实, 再说由此可知什么). 不要越过证据: 资料只说 A 在 B 之前, 就不说 A 导致 B; 没有记载不等于没有发生; 几种说法不一致 (如正文和裴注年份不同) 时, 分别考虑每种说法, 不挑一种当定论.
   - 背景段 (可选, 只在问题里的说法在《三国志》里查不到或和《三国志》不同, 或问到今天的地名, 官职职能时才写): 放在回答最后, 以 "以下不是《三国志》的记载:" 开头, 不加「」, 不标编号. 只能是下面三种句子, 每种一两句:
     (a) "问题里说的某某 (情节, 称号) 《三国志》里没有记载 (或记载不同), 可能出自《三国演义》或其他非正史的说法." 只用于问题里提到, 而你查过原文确认没有或不同的说法; 问题里的说法和《三国志》不冲突时不写这一句, 只陈述原文. 不要把原文里有的事说成演义的说法;
     (b) "某地 (当时的地名) 在今天的某省某地一带.";
     (c) "某官职 (泛指这个官) 负责什么.";
     背景段里不能出现具体人物做了什么, 任过什么官, 哪一年, 当时有没有某种说法: 这些是史实, 要么在正文里引原文回答, 要么不写.
   - 转述 "谁做了什么" 要和原文的主语一致; 拿不准的地方直接引原文, 不要改写. 提到书名和作者, 照资料里写的 (如 "袁暐《献帝春秋》"), 不要换成别的书或别的作者. 原文里 "初" 开头的是追述以前的事.
7. 引用时只引原文 (文言), 不要引白话译文; 一字不改放在「」里, 后面用方括号里的数字标出处, 如「太祖乃變易姓名，間行東歸」[3]. 数字是资料标题开头的 [n]; 不要写资料标题括号里的 id (如 01-0004), 回答正文里也不要出现这种 id. 不要在「」里用省略号跳过内容, 不要把卷名, 人名, 书名单独放进「」.
8. 裴注要说明引自哪本书 (如 "裴注引《魏書》"); "臣松之以為" "臣松之案" 开头的是裴松之自己的按语, 说成 "裴松之按", 不要算到他引的书上. 几种说法不一致时并列说明, 不替读者裁决.
9. 问题出自《三国演义》而正史不同的, 照常回答, 说明 "这是演义的说法, 正史记载为……". 问题里的人物, 称号或情节在资料里查不到, 或者和原文对不上时, 想一想它是不是出自《三国演义》或戏曲 (如 "桃园结义" "三英战吕布"): 是的话要明说 "这是《三国演义》的说法, 《三国志》里没有这件事 (或: 记载为……)", 不要只说 "资料里没找到". 问题的说法查不到, 但资料里有能回答它的事实时 (如问两人是否见过面, 资料里有两人的生卒和行踪), 先按第 6 条的推断给出结论, 再在背景段说明这个说法出自哪里. 问题里的说法 (预设) 和原文不完全一致的 (如问 "谁是总指挥", 原文是两人分任左右督), 照原文说明, 不要顺着问题的说法下结论.
10. 只回答三国 (东汉末到西晋初) 的历史问题. 与此无关的请求 (闲聊, 写代码, 翻译, 其他朝代等), 不调用工具, 一句话说明你只讲三国历史. 问题只描述了一件事, 没写人名, 地名, 年号, 看不出是不是三国的事时 (如 "有个人在船上称大象" "哪个皇帝被刺死了"), 不要直接当成无关请求: 先用问题里的事件词 search 一次, 查到相关记载就照常回答, 查不到再说明你只讲三国历史.
11. 每条资料都有《三国志》原文 (文言繁体) 和【白话译文】. 先看白话弄懂意思, 再对照原文核实; 两者有出入时以原文为准 (译文可能有错).
12. 问 "有哪些/有几个" 的汇总题, 每个可能相关的人或传分别搜一次, 不要搜一次就作答. 找到主要的那段正文后, 对它 read 一次 (会附上它的全部裴注, 裴注里常有正文没写的人和事, 如正文只列四个儿子, 裴注里还有第五个). 问儿子, 子女时, 养子 (原文 "養子" "以為子") 也要列出并注明, 并另外搜一次 "某某 养子". 问 "关系如何" "经过如何" 的综合题, 按阶段分别搜 (早年交往, 合作, 交恶, 结局), 每个阶段都要有原文.
13. 作答前自查: 问题要的每一项都在资料里找到原文了吗? 没找到的继续搜或 read; 确实找不到的明说没找到, 不要用记忆补事实. 查到的事实能不能回答问题? 能用时间先后或对比得出结论的, 不要停在 "资料里没有直接记载", 要把结论推出来并标明是推断.
14. 无论查到多少, 最后一定要给出文字回答, 不能空着.
15. 用户用英文提问时: search 仍然用中文, 把英文里的人名, 地名, 称号, 官职译回中文再搜 (Cao Cao → 曹操, Red Cliffs → 赤壁, Emperor Zhaolie → 昭烈皇帝, Grand Administrator → 太守); 人名也可以直接用拼音 who (如 who("Zhou Tai")), 它会列出拼音相同的所有人, 不止一位时按问题里的线索分清是哪一位, 拿不准就在回答里说明有几位; 拼法不标准的 (旧式威妥玛拼音, 如 Chang Ho, Ts'ao Ts'ao) 先想清楚对应哪个中文名. 回答用英文, 人名用汉语拼音, 官职用通行英译 (Chancellor, Grand Commandant, Inspector, Administrator, Colonel); 引文照样引原文 (文言) 并在后面附英文大意.
中文提问用简体中文回答, 英文提问用英文回答; 先给结论 (包括推断出的结论), 再给依据, 背景段放在最后."""

# 小模型 (gemma) 用的 system prompt: 不分事实 / 推断 / 背景, 只依据原文; 要求回答短 (只答问的那一点,
# 多讲一件事就多一处出错), 并加了读原文要点 (庙号/谥号, 追尊, 年份取事件本身那一段, 祖父辈).
SYSTEM_SMALL = """你是一位讲解三国历史的老师, 依据《三国志》(陈寿正文和裴松之注) 回答提问. 你手边有四个工具: search 检索原文, read 读某一块原文的前后文, era 把年号换算成公元年, who 查人 (一个名字或称号指谁, 有几个同名的人).

做法:
1. 回答任何三国历史问题之前, 必须先用 search 查资料, 至少换两种说法搜两次. 资料库按白话译文检索: 检索词用现代汉语, 人名写全名 (曹操, 刘备, 孙权), 加上事件关键词, 如 "曹操 去世 年龄" "荀彧 儿子 官职". 一次没找到就换说法再搜: 换成相关人物, 地名, 官职, 或人物的字; 译文里的说法常和问题不同 (洛阳 → 京城, 京都; 逃离 → 逃回, 东归, 改换姓名; 被杀 → 遇害, 被刺), 要换着词搜. 问题没给人名的 (如 "哪个皇帝被刺死了"), 先用问题里的事件词搜 ("皇帝 被刺 杀死"), 不要先猜一个人名去搜.
2. 问题里的人只给了称号 (先主, 宣王, 魏王, 明帝), 或问 "有几个X" "X是谁" 时, 先用 who 查: 它告诉你这个称呼指谁, 有几个同名的人, 各自的传在哪一段 (id 可以直接 read), 以及这个人在书里的其他称呼 (译文里常用称呼而不是名字, 如 魏太祖, 先主, 拿称呼去 search 更容易命中). 同名的人要分别查.
3. 原文常用简称和省略. 遇到 "帝" "公" "太祖" "先主" 等不确定指谁, 或 "七年" "明年" 这种没写年号的年份, 用 read 往前读, 直到弄清楚, 不要猜.
4. 资料标题里写 "裴注" 的块 (id 带 n, 如 46-0016n0) 是给正文 46-0016 作的注. 搜到裴注, 却没看到它所注的那段正文时, 对这块裴注调用 read, 会附上它注的正文: 事情的主干 (谁, 哪一年, 结果) 常常在正文里, 不要只凭注文作答.
5. 公元年只能来自 era 工具: 没有调用 era 就不写公元年; 白话译文里也没有公元年. 不要凭记忆写公元年, 也不要补资料里没有的背景 (身份, 情节, 出处).
6. 只依据你查到的原文回答, 不用自己的记忆补充事实. 资料是你自己检索到的, 不是用户提供的. 至少换 3 种说法搜过, 并 read 过最相关的那一段, 都找不到, 才说《三国志》里没有找到相关记载; 这时不要再给出 "通常认为" "一般指" 之类资料以外的说法或名单.
   - 只写资料里写了的: 不补地理, 世系, 别称, 治所之类的背景; 不自己推断因果和先后 ("所以", "此前" 要有原文依据); 原文里 "初" 开头的是追述以前的事.
   - 转述 "谁做了什么" 要和原文的主语一致; 拿不准的地方直接引原文, 不要改写.
   - 提到书名和作者, 照资料里写的 (如 "袁暐《献帝春秋》"), 不要换成别的书或别的作者.
7. 引用时只引原文 (文言), 不要引白话译文; 一字不改放在「」里, 后面标资料编号, 如「太祖乃變易姓名，間行東歸」[3]. 不要在「」里用省略号跳过内容, 不要把卷名, 人名, 书名单独放进「」.
8. 裴注要说明引自哪本书 (如 "裴注引《魏書》"); "臣松之以為" "臣松之案" 开头的是裴松之自己的按语, 说成 "裴松之按", 不要算到他引的书上. 几种说法不一致时并列说明, 不替读者裁决.
9. 问题出自《三国演义》而正史不同的, 照常回答, 说明 "这是演义的说法, 正史记载为……". 问题里的人物, 称号或情节在资料里查不到, 或者和原文对不上时, 想一想它是不是出自《三国演义》或戏曲 (如 "桃园结义" "三英战吕布"): 是的话要明说 "这是《三国演义》的说法, 《三国志》里没有这件事 (或: 记载为……)", 不要只说 "资料里没找到". 问题里的说法 (预设) 和原文不完全一致的 (如问 "谁是总指挥", 原文是两人分任左右督), 照原文说明, 不要顺着问题的说法下结论.
10. 只回答三国 (东汉末到西晋初) 的历史问题. 与此无关的请求 (闲聊, 写代码, 翻译, 其他朝代等), 不调用工具, 一句话说明你只讲三国历史.
11. 每条资料都有《三国志》原文 (文言繁体) 和【白话译文】. 先看白话弄懂意思, 再对照原文核实; 两者有出入时以原文为准 (译文可能有错).
12. 问 "有哪些/有几个" 的汇总题, 每个可能相关的人或传分别搜一次, 不要搜一次就作答. 找到主要的那段正文后, 对它 read 一次 (会附上它的全部裴注, 裴注里常有正文没写的人和事, 如正文只列四个儿子, 裴注里还有第五个). 问儿子, 子女时, 养子 (原文 "養子" "以為子") 也要列出并注明, 并另外搜一次 "某某 养子". 问 "关系如何" "经过如何" 的综合题, 按阶段分别搜 (早年交往, 合作, 交恶, 结局), 每个阶段都要有原文.
13. 作答前自查: 问题要的每一项都在资料里找到原文了吗? 没找到的继续搜或 read; 确实找不到的明说没找到, 不要用记忆补.
14. 无论查到多少, 最后一定要给出文字回答, 不能空着.
15. 读原文的几个要点 (常见的读错):
   - 庙号是 "某祖" "某宗" (太祖, 高祖, 世祖), 谥号是 "某帝" "某皇帝" (武帝, 文帝), 两者不同, 问哪个答哪个.
   - 原文写 "追尊" "追諡" 的人是死后追封的, 不是在位的皇帝; 数 "有几个皇帝" 时只数真正即位的.
   - 问 "哪一年" 时, 年份取事件本身所在的那一段 (往前找最近的年号), 不要用后面别的事件的年份.
   - 分清 "某某的祖父, 父亲, 兄弟" 和某某本人: 祖父的儿子 (如 "八龍") 不是本人的儿子.
16. 用户用英文提问时: 检索 (search, who) 仍然用中文, 把英文里的人名, 地名, 称号, 官职译回中文再搜 (Cao Cao → 曹操, Red Cliffs → 赤壁, Emperor Zhaolie → 昭烈皇帝, Grand Administrator → 太守); 拼法不标准的 (旧式威妥玛拼音, 如 Chang Ho, Ts'ao Ts'ao) 先想清楚对应哪个中文名. 回答用英文, 人名用汉语拼音, 官职用通行英译 (Chancellor, Grand Commandant, Inspector, Administrator, Colonel); 引文照样引原文 (文言) 并在后面附英文大意.
中文提问用简体中文回答, 英文提问用英文回答; 要短: 先用一两句话给结论, 再给 1~3 处原文依据 (引文 + 编号). 只回答问题问的那一点, 不要展开讲其他相关的事件和人物 (多讲一件事就多一处出错的可能); 拿不准的细节宁可不写. 汇总题 (有哪些/有几个) 把每一项列出来, 每项一句."""
SMALL_MODELS = ("gemma",)  # 模型名里含这些字样时, 默认用 SYSTEM_SMALL


def system_for(model: str | None) -> str:
    """按模型选默认 system prompt: 小模型 (SMALL_MODELS) 用 SYSTEM_SMALL, 其他用 SYSTEM."""
    return SYSTEM_SMALL if model and any(s in model.lower() for s in SMALL_MODELS) else SYSTEM


# 工具声明 (OpenAI tools 格式), 作为 tools 参数交给模型; description 文字靠评测迭代. 参数名和 run_tool 取的键一致:
#     search(query, book=None, part=None)   read(chunk_id, before=0, after=0)   era(era, year, regime=None)   who(name)
TOOLS = [
    {"type": "function", "function": {
        "name": "search",
        "description": "在《三国志》(含裴注) 里检索, 按白话译文匹配, 返回最相关的几块: 每块有原文 (文言繁体) 和白话译文. 检索词用现代汉语, 人名写全名, 加事件关键词; 一次只搜一个人物或一件事. 汇总题 (有哪些/有几个) 要对每个相关人物分别搜. 结果里标 \"裴注\" 的块 (id 带 n, 如 46-0016n0) 是给正文 46-0016 作的注; 只搜到注, 没看到它注的正文时, 用 read 读这块注, 会附上正文.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "检索词: 现代汉语, 人名全名 + 事件词, 如 \"曹操 去世 年龄\" \"刘备 儿子\""},
            "book": {"type": "string", "enum": ["魏书", "蜀书", "吴书"],
                     "description": "可选: 只在这一书里搜. 只在已经确定要找的内容在哪一书时才填; 不确定就不填 (搜全书)"},
            "part": {"type": "string", "enum": ["正文", "裴注"],
                     "description": "可选: 只搜陈寿正文或只搜裴注. 只在已经确定时才填; 不确定就不填, 很多事正文和裴注都有"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read",
        "description": f"读取某一块原文, 以及它在同一卷里前后相邻的块; 裴注块会附上它所注的正文 (搜到裴注而没看到正文时, 先 read 这块注). 用来弄清简称 (帝, 公, 太祖) 指谁, 没写年号的年份 (如 七年) 是哪个年号, 或补全上下文. chunk_id 用资料标题括号里的 id, 如 01-0053. before / after 各最多 {READ_MAX}. 找到相关块后, 读它前后各 1~2 块确认上下文 (谁, 哪一年) 再作答.",
        "parameters": {"type": "object", "properties": {
            "chunk_id": {"type": "string", "description": "块 id, 如 01-0053 或 01-0004n1"},
            "before": {"type": "integer", "description": f"往前读几块 (0~{READ_MAX})"},
            "after": {"type": "integer", "description": f"往后读几块 (0~{READ_MAX})"}},
            "required": ["chunk_id"]}}},
    {"type": "function", "function": {
        "name": "era",
        "description": "把年号的第几年换算成公元年, 如 era=建安, year=7 -> 202. 同名年号属于不同政权时 (如 建興 有蜀汉和孙吴的), 可以用 regime 指定政权 (东汉, 曹魏, 蜀汉, 孙吴, 西晋 等); 不指定就返回各政权的结果.",
        "parameters": {"type": "object", "properties": {
            "era": {"type": "string", "description": "年号, 两个字, 如 建安, 黃初"},
            "year": {"type": "integer", "description": "第几年, 元年写 1"},
            "regime": {"type": "string", "description": "政权 (可选)"}},
            "required": ["era", "year"]}}},
    {"type": "function", "function": {
        "name": "who",
        "description": "查人: 给一个姓名, 称号或字 (如 马忠, 先主, 魏王, 曹孟德), 返回别名表里对得上的人: 姓名, 字, 传的首段 id (可以直接 read), 书里的其他称呼 (译文里常用称呼, 如 魏太祖, 先主, 可以拿去 search), 以及已知的同名者. 也可以给拼音 (如 Zhou Tai, Chou T'ai, 英文提问时用): 返回拼音相同的所有人, 不止一位时要按问题里的其他线索分清是哪一位. 小说或其他史书里的叫法 (如 孙尚香, 北宫伯玉) 会说明《三国志》里的写法. 表里只有有传的人和少数常被只称称号的人, 查不到就直接 search.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "姓名, 称号, 字或拼音, 如 马忠, 先主, 曹孟德, Zhou Tai"}},
            "required": ["name"]}}},
]


def run_tool(lib: Library, seen: Seen, name: str, arguments: str | None) -> tuple[str, dict]:
    """执行一个工具调用, 返回 (交给模型的文字, 日志).

    name 和 arguments 原样来自模型的 tool_call; arguments 是 JSON 字符串, "" 或 None 当作 "{}".
    search / read 返回的块经 seen.add 编号后交给模型; era / who 直接返回 Library 给的文字, 不登记块.
        run_tool(lib, seen, "read", '{"chunk_id": "41-0000", "after": 1}')
        -> ("[1] ...", {"tool": "read", "args": {"chunk_id": "41-0000", "after": 1},
                        "chunks": ["41-0000", "41-0001", ...], "error": None})

    不抛异常: 参数写坏, 缺参数, 块不存在, 没有这个工具时返回 ("工具调用出错: <error>. ...", 日志),
    日志的 "error" 为 "类型名: 信息", 交给模型自己改; 这时 seen 不变."""
    log = {"tool": name, "args": None, "chunks": [], "error": None}
    try:
        args = json.loads(arguments or "{}")
        log["args"] = args
        if name == "search":
            got = lib.search(str(args["query"]), args.get("book"), args.get("part"))
        elif name == "read":
            got = lib.read(str(args["chunk_id"]), args.get("before", 0), args.get("after", 0))
        elif name == "era":
            return lib.era(str(args["era"]), args["year"], args.get("regime")), log
        elif name == "who":
            return lib.who(str(args["name"])), log
        else:
            raise ValueError(f"没有这个工具: {name}")
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        log["error"] = f"{type(e).__name__}: {e}"
        return f"工具调用出错: {log['error']}. 请检查参数后重试, 或直接回答.", log
    log["chunks"] = [c["id"] for c in got]
    return seen.add(got), log


# ---------- 第 2 层: 循环里的代码介入 ----------

def make_check(rounds: list[dict], seen: Seen, emit: Callable[[str, Any], None], question: str = "") -> Callable[[str], str | None]:
    """返回 check(回答) -> 提示文字或 None, 决定 agent_loop 返回后要不要追加提示再查.

    rounds 是 run_agent 的工具日志, 用来数模型查了几次. 按顺序判断, 每种提示最多一次, 名字在 OFF 里的跳过:
        nudge            调过 search 或 who, 但 search + read 不到 MIN_SEARCHES 次 -> nudge_text(...);
                         问题是汇总题 (AGGREGATE) 时门槛是 MIN_SEARCHES_AGG (agg_nudge, 默认开)
        nudge_notfound   调过 search, 回答里说没找到 (NOT_FOUND) -> NOT_FOUND_NUDGE
        nudge_scope      一次工具都没用, 回答是 "只讲三国 / 不在范围内" 的拒答 (SCOPE_REFUSAL) -> SCOPE_NUDGE (默认关, 见 OFF)
    一次工具都没用 (闲聊, 拒答) 时, 除了 nudge_scope 都不提示: 只描述事件, 不写人名的三国问题, 模型可能当成无关问题直接拒答. 给出提示时 emit("stage", 名字), 并在 rounds 里记一项 {"by": "code", "tool": 名字, ...}."""
    done: set[str] = set()

    def add(kind: str, text: str) -> str:
        done.add(kind)
        emit("stage", kind)
        rounds.append({"by": "code", "tool": kind, "args": {}, "chunks": [], "error": None})
        return text

    def check(reply: str) -> str | None:
        n_search = sum(1 for x in rounds if x["tool"] == "search")
        n_look = n_search + sum(1 for x in rounds if x["tool"] == "read")  # 搜过之后又 read 过, 算在找, 不提示
        n_who = sum(1 for x in rounds if x["tool"] == "who")
        # 只调了 who 没 search 也要提示: 小模型常常 who 之后直接作答
        need = MIN_SEARCHES_AGG if "agg_nudge" not in OFF and AGGREGATE.search(question) else MIN_SEARCHES
        if "nudge" not in done and "nudge" not in OFF and (n_search or n_who) and n_look < need:
            return add("nudge", nudge_text(n_look, seen.chunks))
        if "nudge_notfound" not in done and "nudge_notfound" not in OFF and n_search and NOT_FOUND.search(reply):
            return add("nudge_notfound", NOT_FOUND_NUDGE)
        n_tools = sum(1 for x in rounds if x.get("by") != "code")
        if "nudge_scope" not in done and "nudge_scope" not in OFF and not n_tools and SCOPE_REFUSAL.search(reply):
            return add("nudge_scope", SCOPE_NUDGE)
        return None
    return check


# ---------- 第 3 层: 作答后的处理 ----------

def after_answer(reply: str, messages: list[dict], complete: Callable[..., Any], state: dict, seen: Seen,
                 rounds: list[dict], emit: Callable[[str, Any], None], review: bool) -> tuple[str, list[tuple[str, str | None]]]:
    """agent_loop 返回之后的处理, 返回 (回答, quotes); quotes 同 verify_quotes: [("引文", "chunk id" | None), ...].

    complete(messages) -> message, 不传 tools; state 记请求次数 ("calls") 和最近的 finish_reason ("finish").
    依次: retry (bad_reply 且 state["calls"] < MAX_ROUNDS + 3 时重答一次), 去掉工具调用标记, review (按 REVIEW_MODE 选自查指令),
    引文核对和 quote_fix (核对不过的回给模型改一次), 换掉照抄的 chunk id 编号, cite_fix (代码改正能确定的错编号).
    review 为假时跳过 review 和 quote_fix, 名字在 OFF 里的步骤也跳过; 模型没调过工具时不自查. 自查 / 改引文的结果是拒答模板, 坏回答, 或比原回答短一半以上时, 保留原回答,
    因为服务商偶尔返回拒答模板, 不能覆盖本来对的回答.
    每一步在 rounds 里记一项, 如 {"by": "code", "tool": "quote_fix", "args": {"bad": ["公軍譙縣"]}, "chunks": [], "error": None}."""
    def usable(new: str, old: str) -> bool:
        return (not bad_reply(new, state["finish"]) and not _REFUSAL_TEMPLATE.search(new)
                and len(strip_markup(new)) >= len(old) // 2)

    def log(kind: str, args: dict | None = None) -> None:
        rounds.append({"by": "code", "tool": kind, "args": args or {}, "chunks": [], "error": None})

    if bad_reply(reply, state["finish"]) and state["calls"] < MAX_ROUNDS + 3:
        # 截断 ("**是董") / 只有工具调用标记 / 空回答: 去掉坏回答, 要求重新作答一次 (不传 tools)
        emit("stage", "retry")
        messages.append({"role": "developer", "content": FINAL_TURN})
        reply = complete(messages).content or ""
        log("retry")
    reply = strip_markup(reply)
    if review and "review" not in OFF and reply and any(x["by"] == "model" for x in rounds):
        # 作答后自查一遍: 针对补了资料外的背景/名单, 推错因果先后, 年份张冠李戴. 代价是每题多一次请求
        emit("draft", reply)  # 先把初稿给界面看, 自查结果出来后再换
        state["draft"] = reply
        emit("stage", "review")
        messages.append({"role": "assistant", "content": reply})
        text = {"short": REVIEW_TEXT_SHORT, "layered": REVIEW_TEXT_LAYERED, "patch": REVIEW_TEXT_PATCH}.get(REVIEW_MODE, REVIEW_TEXT)
        messages.append({"role": "user", "content": text + ("" if "lang_note" in OFF else " " + LANG_NOTE)})
        checked = complete(messages).content or ""
        state["review_raw"] = checked
        ok = REVIEW_MODE in ("short", "patch") and bool(_REVIEW_OK_RE.match(checked.strip()))
        if REVIEW_MODE == "patch":
            new, done, missed = (reply, 0, 0) if ok else apply_patch(reply, strip_markup(checked))
            log("review", {"ok": ok, "edits": done, "missed": missed})
            if done and usable(new, reply):
                reply = new
        else:
            log("review", {"ok": True} if ok else None)
            if not ok and usable(checked, reply):  # 自查时偶尔返回服务商的拒答模板, 不能覆盖原本对的回答
                reply = strip_markup(checked)
    quotes = verify_quotes(extract_quotes(reply), seen.chunks) if reply else []
    bad_q = [q for q, cid in quotes if cid is None]
    if bad_q and review and "quote_fix" not in OFF:
        # 引文核对不过的 (记错字, 用省略号跳过, 引了白话) 回给模型改一次
        emit("stage", "quote_fix")
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": QUOTE_FIX.format(quotes="\n".join(f"「{q}」" for q in bad_q))})
        fixed = complete(messages).content or ""
        log("quote_fix", {"bad": bad_q})
        if usable(fixed, reply):
            reply = strip_markup(fixed)
            quotes = verify_quotes(extract_quotes(reply), seen.chunks)
    # 模型有时照抄出处行里的 chunk id 当编号 ([41-0000]): 统一换成 [n], 后面的编号核对才认得
    reply = renumber_cites(reply, seen.chunks)
    if reply and "cite_fix" not in OFF:
        # 引文出自另一块却标了这一块的编号: 文字只在一块里时由代码直接改 (不再请求模型)
        reply, n = fix_cites(reply, seen.chunks)
        if n:
            log("cite_fix", {"fixed": n})
    return reply, quotes


# ---------- 接线 ----------

HISTORY_TURNS = 3  # 多轮对话时带上最近几轮问答 (每轮 = 用户问题 + 最终回答, 不含检索过程)


def run_agent(question: str, lib: Library, model: str | None = None, complete: Callable[..., Any] | None = None,
              system: str | None = None, effort: str | None = None, tools: list[dict] | None = None,
              review: bool | None = None, history: list[dict] | None = None,
              on_event: Callable[[str, Any], None] | None = None,
              handlers: dict[str, Callable[[dict], str]] | None = None) -> dict:
    """回答一个问题: 组 messages -> agent_loop -> after_answer.

    参数:
        lib       load_library() 的结果
        model     模型名; None 时用 .env 的 LLM_MODEL
        complete  complete(messages, model=None, **kw) -> ChatCompletion 形状的对象; 默认 rag.llm.complete,
                  测试换成假模型, 评测和网页聊天换成 rag/calls.make_complete 的结果
        system / tools / review   不给时用 system_for(model) / TOOLS / REVIEW, 实验对照时替换
        effort    给了就作为 reasoning_effort 传给模型
        history   之前几轮的 user / assistant 消息 (只含最终回答), 取最后 HISTORY_TURNS 轮放在本轮问题之前,
                  让 "他儿子呢" 这类追问接得上; 资料编号 [n] 每轮从 1 重新开始
        语言提醒: OFF 里没有 "lang_note" (默认) 时, 问题后面追加 {"role": "developer", "content": LANG_NOTE},
                  自查指令末尾也追加一次 (after_answer); 不认 developer 角色的模型会忽略它
        on_event  on_event(kind, info) 报进度, kind 为 "call" (第几次请求) / "tool" (工具日志) / "stage" (代码介入的名字)
                  / "draft" (自查前的初稿, 界面可以先显示);
                  回调里的异常被吞掉
        handlers  TOOLS 以外的工具: {工具名: handler(参数 dict) -> 交给模型的文字}, schema 由调用方放进 tools.
                  网页聊天用它接游戏 skill 的 start_game / end_game (rag/skills); handler 抛的异常不吞, 会结束这一问
                  (rag/skills.SkillReply 就是这样把游戏的回复直接交给界面)

    返回 (列表有删节):
        {"question": "刘备从葭萌南下打刘璋时留霍峻守城，……", "reply": "**结论**：……",
         "draft": "自查前的初稿 (没自查时为 None)", "review_raw": "自查那次请求的原始输出 (没自查时为 None)",
         "chunks": [<41-0000 的块>, ...],                 模型看过的块, [n] 指 chunks[n-1]
         "first": ["41-0000", "41-0017", ...],            第一次成功的 search 返回的块 id; 没有时为 []
         "quotes": [("小人頭可得，城不可得", "41-0000"), ...],   引文和所在块; 原文里找不到时为 None
         "cites": [{"quote": "...", "cites": [3], "status": "ok", "found": [3]}, ...],   rag.quotes.check_cites
         "rounds": [{"tool": "search", "args": {...}, "chunks": [...], "error": None, "by": "model"}, ...],   工具调用和代码介入
         "calls": 4,                                      请求模型的次数
         "prompt_tokens": [2630, 10882, 14246, 13752],    每次请求的 prompt token 数; 接口没给时为 None
         "messages": [...]}                               完整对话

    请求报 "no user query found in messages" 时补一条用户消息重申问题再试一次; 其他异常直接抛出."""
    tools = tools or TOOLS
    base = complete or llm.complete
    extra = {"reasoning_effort": effort} if effort else {}

    def emit(kind: str, info: Any) -> None:
        if on_event:
            try:
                on_event(kind, info)
            except Exception:
                pass

    seen, rounds, usage = Seen(lib.notes), [], []
    state = {"calls": 0, "finish": None}
    past = [m for m in (history or []) if m.get("role") in ("user", "assistant") and m.get("content")]
    messages = ([{"role": "system", "content": system or system_for(model)}] + past[-2 * HISTORY_TURNS:]
                + [{"role": "user", "content": question}])
    notes = []  # 紧跟在问题后面的 developer 消息 (同 rag/loop.py 的两条提示); 合成一条, 有的接口不许两条系统消息相连
    if "pinyin_who" not in OFF:  # 英文问题里的拼音人名, 代码先 who 一次, 拼音相同的人一起列出 (有的模型会自己把名字译错, 也不用拼音查);
        # 引导语用英文: 中文引导语 + 中文的 who 结果会把 Claude Haiku 带成用中文作答 (英文题 18 题里 1 题)
        for name in find_pinyin_names(question):
            notes.append(f"Code already looked up the pinyin name {name} in the question with who (the result is in Chinese "
                         f"only because the source text is):\n{lib.who(name)}")
            rounds.append({"tool": "pinyin_who", "args": {"name": name}, "chunks": [], "error": None, "by": "code"})
    if "lang_note" not in OFF:  # 提醒一次作答语言, 自查指令末尾再提醒一次
        notes.append(LANG_NOTE)
    if notes:
        messages.append({"role": "developer", "content": "\n\n".join(notes)})

    def call(msgs: list[dict], tools: list[dict] | None = None, effort_: str | None = None) -> Any:
        """请求一次模型, 返回 message (agent_loop 和 after_answer 用): 带上 model / effort (effort_ 给了就换成它),
        tools 为 None 时不传; 计数, 记 prompt tokens 和 finish_reason, 报进度."""
        emit("call", state["calls"] + 1)
        kw: dict[str, Any] = {"tools": tools} if tools else {}
        if effort_:
            kw["reasoning_effort"] = effort_
        try:
            resp = base(msgs, model=model, **{**extra, **kw})
        except Exception as e:
            # 有的对话模板 (qwen3.8) 偶尔报 "no user query found in messages", 原因不明: 补一条用户消息重申问题, 再试一次
            if "no user query" not in str(e):
                raise
            msgs.append({"role": "user", "content": f"请根据以上检索结果回答原问题: {question}"})
            resp = base(msgs, model=model, **{**extra, **kw})
        state["calls"] += 1
        state["finish"] = getattr(resp.choices[0], "finish_reason", None)
        usage.append(getattr(getattr(resp, "usage", None), "prompt_tokens", None))
        return resp.choices[0].message

    def call_tool(name: str, arguments: str) -> str:
        if handlers and name in handlers:
            try:
                args = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            log = {"tool": name, "args": args, "chunks": [], "error": None, "by": "model"}
            rounds.append(log)
            emit("tool", log)
            return handlers[name](args if isinstance(args, dict) else {})
        text, log = run_tool(lib, seen, name, arguments)
        log["by"] = "model"
        rounds.append(log)
        emit("tool", log)
        return text

    reply = agent_loop(messages, tools, call, call_tool)
    # 查得太少就作答等情况: 追加提示, 用剩下的请求次数再跑一次循环. 和把提示放在循环里等价: 提示那次回答也占一次请求,
    # 最后一次 (第 MAX_ROUNDS + 1 次请求) 的回答不再检查; 每种提示最多一次 (make_check 自己记).
    check = make_check(rounds, seen, emit, question)
    while state["calls"] <= MAX_ROUNDS and (hint := check(reply)):
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": hint})
        reply = agent_loop(messages, tools, call, call_tool, max_rounds=MAX_ROUNDS - state["calls"])
    after = (lambda m, tools=None: call(m, tools, REVIEW_EFFORT)) if REVIEW_EFFORT else call
    reply, quotes = after_answer(reply, messages, after, state, seen, rounds, emit, REVIEW if review is None else review)
    first = next((r["chunks"] for r in rounds if r["tool"] == "search" and not r["error"]), [])
    return {"question": question, "reply": reply, "draft": state.get("draft"), "review_raw": state.get("review_raw"),
            "chunks": seen.chunks, "first": first,
            "quotes": quotes, "cites": check_cites(reply, seen.chunks), "rounds": rounds, "calls": state["calls"], "prompt_tokens": usage, "messages": messages}


def main() -> None:
    ap = argparse.ArgumentParser(description="agent 问答: 模型调用工具查《三国志》后作答")
    ap.add_argument("question")
    ap.add_argument("--model", default=None)
    ap.add_argument("--index", default="trans", choices=["trans", "orig"], help="检索哪套切块 (默认白话)")
    ap.add_argument("--effort", default=None, help="reasoning_effort (none / low / medium / high), 覆盖 .env")
    args = ap.parse_args()
    import threading

    from rag import calls
    args.model = calls.setup(args.model)  # 后端 (DeepSeek 经 OpenRouter / Claude 原生接口) 由 .env 的 LLM_BACKEND 决定
    complete = calls.make_complete(args.model, threading.Semaphore(1))
    res = run_agent(args.question, load_library("" if args.index == "orig" else args.index), args.model,
                    complete=complete, effort=args.effort)
    for r in res["rounds"]:
        print(f"[{r['by']}] {r['tool']}({json.dumps(r['args'], ensure_ascii=False)}) -> {r['chunks']}"
              + (f"  出错: {r['error']}" if r["error"] else ""))
    print(f"\n请求 {res['calls']} 次, prompt tokens {res['prompt_tokens']}, 看过 {len(res['chunks'])} 块\n")
    print(res["reply"])
    for q, cid in res["quotes"]:
        print(f"  {'✓ ' + cid if cid else '✗ 原文里找不到'}  「{q}」")


if __name__ == "__main__":
    main()
