"""猜人物, 两种玩法 (GuessPerson 开局时让玩家选):
    1. Agent 出题, 玩家来猜 (GuessGame): Agent 心里定一个《三国志》人物, 玩家问是非问题, 最后说出名字.
    2. 玩家出题, Agent 来猜 (AgentGuessGame): 玩家心里想一个人, Agent 问是非问题 (GUESS_SYS, 可以查原文), 最后猜名字.

    uv run python -m rag.skills --openrouter   # 命令行里玩 (输入 "猜人物" 开局; 不带 --openrouter 用 .env 的模型), 见 rag/skills.py
    uv run python -m tests.test_skills         # 假模型测试, 不连网

玩法 1 的谜底由 random_person 从别名表 (data/aliases.json, 有传的人和原文多处记载的无传人物) 里随机出:
先统计每个人在全书正文块里出现的块数 (person_forms 去掉多人共用的称呼, 如 "都亭侯"), 按 块数 ** POP_WEIGHT 加权挑人
(名人稍多, 但不独占), 再从他出现的块里随机取一块给裁判预读. 抽到的谜底 (Person) 例如:
    {"id": "32-0000",                       别名表的人物 id, 也是本传首段 (没有传的人是提到他的一处) 的源 id
     "person": "刘备",
     "aliases": ["先主", "刘玄德", "刘豫州", "汉中王", "蜀汉昭烈皇帝"],   别名表里的称呼, 猜这些也算对
     "bio": ["32-0000", "32-0028"],         裁判预先读的原文: 本传首段 + 抽到的那一块的源 id
     "note": "字玄德"}                       揭晓时显示的一句说明
    notrans (可选) 为 true 时, 这个人在《三国志》里没有自己的传, bio 第一个 id 只是提到他的一处.

玩法 1 的裁判 (judge_question): 一局是同一段对话 (GuessGame.msgs), 谜底和它的出处原文在 system prompt 里, 需要时再用工具查;
前面的问答和玩家猜错的名字都留在对话里, 裁判据此保持前后一致. 它只写结论 (是 / 不是, 二选一的答选项, 原文没有的答
"没有记载", 问题有两种以上理解时在括号里注明按哪种理解), 不提供问题以外的信息; 页面只显示这一句和计数.
代码只检查这句话里有没有谜底的名字或称呼 (rag/soup.leaks), 有就换成 LEAK_REPLY. 玩家猜名字时由代码比对人名和称呼 (matches),
不交给模型判断.
"""
import random
import re
import threading
import time
from collections import Counter
from collections.abc import Mapping
from typing import Any, Callable, TypedDict

from rag import agent
from rag.library import Library, Seen
from rag.loop import agent_loop
from rag.normalize import to_simplified
from rag.soup import LEAK_REPLY, _parse_json, leaks

VERDICTS = ("是", "不是", "选择", "没有记载", "没法回答", "非游戏")  # 判词类别; 给玩家看的是 answer 那一句话
NOT_GAME = "非游戏"  # 这句话不是游戏里的问题: GuessGame.ask 返回 None, 交给问答 agent
NOT_COUNTED = "没法回答"  # 这一类不算一问
MAX_GUESSES = 3  # 两种玩法都是: 问问题不限次数, 猜名字最多几次 (猜错这么多次就结束)
JUDGE_ROUNDS = 3  # 裁判每个问题最多几轮工具调用
TURN_BUDGET = 90.0  # 一问 (一次 play) 最多花几秒: 超过后不再发新请求, 抛 TimeoutError, 由游戏提示玩家再说一次
OnEvent = Callable[[str, Any], None]  # on_event(kind, info): 进度事件, kind 是 "tool" / "stage" / "think" (见 play)

JUDGE_SYS = """你是 "猜人物" 游戏的裁判. 谜底是《三国志》里的 {person} (原文里也称 {aliases}; 本传或出处 {bio}). 玩家不知道谜底,
会向你提问. 这个人的出处原文已经附在下面 (编号 [n]), 先看这些资料; 问到资料里没有的事 (如和别人的交往, 别处的记载)
才用工具查《三国志》原文, 不要把下面已有的段落再 read 一遍.

{bio_text}

一局里玩家会一个一个地提问, 也会猜名字 (猜名字由程序判对错, 结果会告诉你). 怎么答 (只答结论, 不介绍, 不解释):
- 是非题答 "是" 或 "不是"; 二选一的问题只答选项.
- 原文没有记载、推不出来的, 答 "没有记载".
- 句末是 "吗" 的问题是是非题, 句子里有 "或者" 也一样: 符合其中任何一项就答 "是". 用 "还是" 连接选项的才是二选一.
- 不提供问题以外的任何信息 (官职、时期、经历、事件都不说).
- 问题有两种以上理解、按不同理解答案不同时, 在结论后用括号注明按哪种理解, 括号里只写理解, 不附带别的信息.
- 这是一局里的连续问答, 前面的问题和你的回答都在对话里; 回答要和前面一致.
- 没法用是 / 不是或一个选项回答的问题, 请玩家换成是非题或二选一的问题.
- 不是在问这个人的话 (问玩法、闲聊、想结束或换游戏、问和谜底无关的历史问题) -> verdict 填 "非游戏", answer 留空; 这句话会交给别人处理.
- 只依据查到的原文, 不用自己的历史知识. 注意同名的人, 不要查成别人.
- 不能写出谜底的名字、字、称号, 也不能直接说出他的姓.
每个问题最后只输出一行 JSON: {{"verdict": "是 | 不是 | 选择 | 没有记载 | 没法回答 | 非游戏", "answer": "结论 (必要时加括号注明按哪种理解)", "evidence": [依据的资料编号 n]}}"""

GUESS_SYS = """你在玩 "猜人物": 对方心里想了一个三国人物 (多半是《三国志》里的人, 也可能是只在《三国演义》里出名的人物),
你要用是非问题猜出是谁. 问问题不限次数, 但名字最多只能猜 {g} 次, 猜错 {g} 次就输了, 所以有把握再猜.

怎么问: 像一个熟悉三国的真人玩家那样, 问这个人本身, 让对方好答、有意思:
  - 身份: 君主, 将领, 谋士, 文人, 宦官, 后妃或女性, 外族首领……
  - 阵营和经历: 属于哪一方, 换过主公吗, 投降过吗 (魏、蜀、吴不是全部, 还有汉臣、汉末群雄、外族等; 不是魏也不是蜀不等于是吴);
  - 有记忆点的事: 打过哪场有名的仗, 和哪个名人关系密切或结仇, 做过什么出名的事, 有什么典故或外号;
  - 结局: 怎么死的, 被谁所杀, 寿终还是战死.
不要问书的结构 (在魏书、蜀书还是吴书, 正文还是裴注, 有没有自己的传, 在谁的传里): 对方心里想的是一个人, 多半答不上来.
实在没有别的线索时, 最多问一次 "他在史书里有自己的传吗".
一个问题只问一件事, 不要把两个条件放进同一问.

对方的回答: 多半凭印象, 可能是 "是" "不是" "不知道", 也可能是 "算是吧" "差不多" "不太记得了", 甚至记错.
按意思理解, 含糊的回答别当成铁证, 不要因为一句回答就排除一大片人; 人物可能先后属于不同阵营 (投降, 归附).

查原文: 你手边有检索《三国志》原文的工具. 前几问范围还大, 只问不查. 回答里出现能落到原文的线索 (事件, 关系, 官职, 结局) 后再查,
搜索词从这些线索来; search 的 book / part 参数可以自己用来缩小范围 (这是你查资料的手段, 不拿去问对方).
别在心里逐个列举候选人的生平: 心里有了一串候选人时, 用工具代替回忆 —— 只查两三个最像的人的细节,
或者把候选人的名字一股脑放进一次 search (如 "黄权 王平 张嶷"), 看原文里有什么能把他们分开的线索.
演义里的人物 (如貂蝉) 在《三国志》里可能没有这个名字, 只有相近的记载; 查不到时想想这种可能.

猜名字: 有把握时直接猜, 不要用 "就是某某吗" 这种是非问题去确认 (那会多用一次机会); 候选人还很多时不猜.
每一轮只做一件事, 最后只输出一行 JSON:
  问问题: {{"ask": "一个是非问题"}}
  猜名字: {{"guess": "人名", "desc": "一句话说明是哪一位 (同名的人很多, 如几个马忠)"}}"""

# 明确的猜名字: "/猜 刘备" "猜刘备" "我猜是刘备" "答案是刘备吗"
_GUESS = re.compile(r"^(?:/猜|猜|我猜|答案是)\s*(?:是|他是|她是)?\s*(.+?)\s*(?:吗|吧)?\s*[？?。.!！]*$")


class Person(TypedDict, total=False):
    """一局 "Agent 出题" 的谜底, 字段见模块 docstring."""
    id: str
    person: str
    aliases: list[str]
    bio: list[str]
    note: str
    notrans: bool  # 三国志里没有自己的传 (bio 第一个 id 只是提到他的一处), 揭晓时不说 "他的传在……"


def field(reply: str, key: str) -> str | None:
    """从模型回答里取 JSON 的某个字符串字段; JSON 不合法时 (如编号写成 [32-0000] 没加引号) 用正则直接找 "key": "值"."""
    v = _parse_json(reply).get(key)
    if v:
        return str(v)
    m = re.search(rf'"{key}"\s*:\s*"([^"]+)"', reply or "")
    return m.group(1) if m else None


def matches(name: str, person: Mapping[str, Any]) -> bool:
    """猜的名字是不是谜底: 和人名或任一称呼 (繁简不论) 一致, 或互相包含 (如 "刘玄德" 和 "玄德"); 单字不算."""
    n = to_simplified(name).strip()
    forms = [to_simplified(f) for f in [person["person"], *person.get("aliases", [])]]
    return len(n) >= 2 and any(n == f or (len(f) >= 2 and (f in n or n in f)) for f in forms)


def tool_runner(lib: Library, seen: Seen, log: list[dict]) -> Callable[[str, str], str]:
    """run_tool(name, arguments) -> 给模型的文字 (rag/agent.run_tool); 每次调用记进 log."""
    def run(name: str, arguments: str) -> str:
        text, entry = agent.run_tool(lib, seen, name, arguments)
        log.append({"tool": name, "args": entry.get("args"), "chunks": entry.get("chunks"), "error": entry.get("error")})
        return text
    return run


def play(msgs: list[dict], lib: Library, seen: Seen, log: list[dict], complete: Callable[..., Any], rounds: int,
         on_event: OnEvent | None = None, show: bool = True, budget: float = TURN_BUDGET) -> str:
    """跑一轮 agent_loop, 再删掉它追加的 developer 提醒 ("工具调用次数已经用完" 等).
    同一段对话要接着用时, 这些提醒留在历史里会让模型以为再也不能查原文.

    on_event 不为 None 时, 每次模型回复后发 ("think", 推理文字) (OpenRouter 在 message.reasoning 里返回, 没有就不发),
    每次工具调用后发 ("tool", {"tool", "args", "chunks", "error"}) (格式同 rag/agent.run_agent 的事件, app/chat.describe 能显示).
    show=False 时 (裁判知道谜底, 推理和检索词会泄底) 不发推理, 工具调用只发 ("stage", "查原文").
    budget 秒后不再发新请求, 抛 TimeoutError; msgs 此时可能停在半截, 调用方要自己恢复."""
    start = time.time()

    def call(messages: list[dict], tools: list[dict] | None = None) -> Any:
        if time.time() - start > budget:
            raise TimeoutError(f"这一问超过 {budget:.0f} 秒")
        msg = complete(messages, tools)
        reasoning = getattr(msg, "reasoning", None)
        if on_event and show and reasoning:
            on_event("think", reasoning)
        return msg

    run = tool_runner(lib, seen, log)

    def run_tool(name: str, arguments: str) -> str:
        text = run(name, arguments)
        if on_event:
            on_event(*(("tool", log[-1]) if show else ("stage", "查原文")))
        return text

    reply = agent_loop(msgs, agent.TOOLS, call, run_tool, max_rounds=rounds)
    msgs[:] = [m for m in msgs if m.get("role") != "developer"]
    return reply


def bio_chunks(lib: Library, ids: list[str]) -> list[Any]:
    """谜底人物出处 (源 id) 对应的全部块, 按原文顺序去重 (一个源段切成几块时都取)."""
    out: list[Any] = []
    for sid in ids:
        for i in lib.by_source.get(sid, []):
            if lib.chunks[i] not in out:
                out.append(lib.chunks[i])
    return out


def judge_messages(lib: Library, person: Person, seen: Seen) -> list[dict]:
    """一局裁判对话的开头: system prompt 里放谜底和它的出处原文 (bio_chunks, 经 seen 编号, 之后工具查到的段落接着编号)."""
    sys = JUDGE_SYS.format(person=person["person"], aliases="、".join(person.get("aliases", [])) or "无",
                           bio=", ".join(person["bio"]), bio_text=seen.add(bio_chunks(lib, person["bio"])))
    return [{"role": "system", "content": sys}]


def judge_question(lib: Library, person: Person, question: str, complete: Callable[..., Any],
                   on_event: OnEvent | None = None, msgs: list[dict] | None = None, seen: Seen | None = None) -> dict:
    """裁判玩家的一个问题. 返回 {"verdict": VERDICTS 之一 (类别), "answer": 给玩家看的一句话, "by": "model",
    "tools": [工具调用记录], "raw": 模型回答末尾}.

    complete(messages, tools=None) -> 模型的一条回复 message (rag/loop.agent_loop 的约定). 是非题、二选一、问法有歧义的问题
    都交给模型按意思回答, 代码不预先拦; 唯一的代码检查是 leaks: 回答里出现谜底的名字或称呼就换成 LEAK_REPLY.
    msgs 是这一局的裁判对话 (judge_messages 开头, 之前的问答和工具调用都在里面), 这一问和回答原地追加, 超时时恢复原样;
    不传时单独开一段 (seen 同时新建). msgs 和 seen 要成对传 (GuessGame 的 self.msgs / self.seen): 只传 msgs 时工具查到的段落会从 [1] 重新编号, 和 system prompt 里已编好的号撞上. 谜底的出处原文预先放在 system prompt 里, 大多数问题不用再调工具, 资料以外的事才查. 模型给出的判词类别不在 VERDICTS 里 (或解析不了) 时记为
    "没有记载"; answer 为空时用判词本身. on_event 只收到 ("stage", "查原文") (裁判的推理和检索词会泄底, 不发). 超过 TURN_BUDGET 时抛 TimeoutError."""
    if msgs is None:
        seen = Seen(lib.notes)
        msgs = judge_messages(lib, person, seen)
    seen = seen if seen is not None else Seen(lib.notes)
    log: list[dict] = []
    snapshot = list(msgs)
    msgs.append({"role": "user", "content": question})
    try:
        reply = play(msgs, lib, seen, log, complete, JUDGE_ROUNDS, on_event, show=False)
    except BaseException:  # 超时, 用户停止 (rag/skills.Cancelled): 这一问作废, 对话回到问之前
        msgs[:] = snapshot
        raise
    msgs.append({"role": "assistant", "content": reply})
    verdict = field(reply, "verdict")
    verdict = verdict if verdict in VERDICTS else "没有记载"
    if verdict == NOT_GAME:
        msgs[:] = snapshot  # 不是游戏里的问题: 不留在裁判对话里
        return {"verdict": verdict, "answer": "", "by": "model", "tools": log, "raw": reply[-300:]}
    answer = (field(reply, "answer") or "").strip() or verdict
    if leaks(answer, [person["person"], *person.get("aliases", [])]):
        answer = LEAK_REPLY
    return {"verdict": verdict, "answer": answer, "by": "model", "tools": log, "raw": reply[-300:]}


def parse_guess(text: str, known_names: set[str] | None = None) -> str | None:
    """玩家这句话是不是在猜名字, 是就返回猜的名字. 认两种写法: 明确的 "/猜 X" "猜X" "我猜是X" "答案是X";
    或整句只是一个已知人名 (known_names, 如 "刘备" "刘备？"). "他是男的吗" 这类普通问题不算猜名字."""
    t = text.strip()
    m = _GUESS.match(t)
    if m:
        return m.group(1).strip() or None
    bare = re.sub(r"^(?:是|他是|她是)|(?:吗|吧)?[？?。.!！]*$", "", t)
    if known_names and to_simplified(bare) in known_names:
        return bare
    return None


class GuessGame:
    """一局猜人物: 谜底, 已问的问题数 (不限), 猜错名字的次数 (到 max_guesses 结束并揭晓), 是否结束. 不依赖界面, 测试和 rag/skills.py 直接调它.

    lib 是 rag.library.load_library() 的结果; complete 签名同 judge_question 的 complete."""

    def __init__(self, person: Person, lib: Library, complete: Callable[..., Any], max_guesses: int = MAX_GUESSES,
                 known_names: set[str] | None = None) -> None:
        self.p, self.lib, self.complete, self.max = person, lib, complete, max_guesses
        self.known_names = known_names or set()
        self.asked = 0    # 问了几个问题 (不限)
        self.guesses = 0  # 猜错了几次名字 (到 self.max 就结束)
        self.over = False
        self.seen = Seen(lib.notes)
        self.msgs = judge_messages(lib, person, self.seen)  # 整局的裁判对话, 每问接着往下问

    def intro(self, on_event: OnEvent | None = None) -> str:
        """开局说明."""
        return (f"我心里想好了一个《三国志》里出现过的人物. 除了魏蜀吴的人, 也可能是东汉末年 (如董卓部下、汉朝大臣)"
                " 或西晋 (如晋朝开国的将相) 的人."
                " 问题不限次数 (是非题或二选一, 我查原文回答), 名字最多猜"
                f" {self.max} 次: 说 \"猜 某某\" 或直接说名字. 输入 /放弃 揭晓答案, /退出 结束游戏.")

    def ask(self, text: str, on_event: OnEvent | None = None) -> str | None:
        """处理玩家的一句话: 猜名字就由代码比对, 否则当作是非问题交给裁判. 返回界面上显示的一行.
        问题不限次数; 猜错名字 self.max 次就结束并揭晓.
        裁判超时时这一问不计数, 请玩家再问一次. 裁判判断不是游戏里的问题 (NOT_GAME) 时返回 None, 不计数."""
        text = text.strip()
        if self.over:
            return "这一局已经结束了."
        if not text:
            return ""
        name = parse_guess(text, self.known_names)
        if name:
            if matches(name, self.p):
                self.over = True
                return "猜对了! " + self.reveal_text()
            self.guesses += 1
            if self.guesses >= self.max:
                self.over = True
                return f"不是 {name}. {self.max} 次机会用完了. " + self.reveal_text()
            self.msgs.append({"role": "user", "content": f"(玩家猜是 {name}, 程序判定猜错了. 这一条不用回答.)"})
            self.msgs.append({"role": "assistant", "content": "好的."})
            return f"不是 {name}. (还能猜 {self.max - self.guesses} 次)"
        try:
            r = judge_question(self.lib, self.p, text, self.complete, on_event, self.msgs, self.seen)
        except TimeoutError:
            return "这一问查原文太久了, 没判出来 (不计数). 请再问一次."
        if r["verdict"] == NOT_GAME:
            return None
        if r["verdict"] != NOT_COUNTED and r["answer"] != LEAK_REPLY:
            self.asked += 1
        return f"{r['answer']}  (第 {self.asked} 问)"

    def reveal(self) -> str:
        """放弃, 直接揭晓."""
        self.over = True
        return self.reveal_text()

    def reveal_text(self) -> str:
        """揭晓谜底, 给人看的写法 (不出现 chunk id):
            谜底是 曹操 (字孟德; 又称 魏太祖、曹公、魏武帝).
            他的传在《三国志·魏书·武帝纪》, 开头是:「太祖武皇帝，沛國譙人也，姓曹，諱操，字孟德，漢相國參之後。」
        说明取 note (同名时说清是哪一位, 如 "孙吴潘璋的司马"), 别称取 aliases 前三个; 出处取 bio 第一个源 id 所在块的卷名,
        和那一块里第一句提到他 (名字或别称) 的话, 没有就取首句; 太长截到 60 字. 找不到那一块时只说名字.
        notrans 的人 (三国志里没有传) 说 "提到他的一处在……"."""
        extra = [self.p["note"]] if self.p.get("note") else []
        if self.p.get("aliases"):
            extra.append("又称 " + "、".join(self.p["aliases"][:3]))
        out = f"谜底是 {self.p['person']}" + (f" ({'; '.join(extra)})" if extra else "") + "."
        idx = self.lib.by_source.get(self.p["bio"][0], []) if self.p.get("bio") else []
        if idx:
            c = self.lib.chunks[idx[0]]
            title = to_simplified(c.get("juan_title", "")).replace("•", "·").replace(" ", "·")
            sentences = [x for x in re.split(r"(?<=。)", c["text"]) if x.strip()] or [c["text"]]
            names = [to_simplified(n) for n in [self.p["person"], *self.p.get("aliases", [])] if len(n) >= 2]
            first = next((x for x in sentences if any(n in to_simplified(x) for n in names)), sentences[0])
            first = first if len(first) <= 60 else first[:60] + "……"
            where = "《三国志》里没有他的传, 提到他的一处在" if self.p.get("notrans") else "他的传在"
            out += f"\n{where}《三国志·{title}》:「{first}」" if c.get("tier") == 1 else ""
        return out


NOT_THREE_KINGDOMS = {"孔子", "公孙述"}  # 别名表里收了、但不是三国时代的人 (原文只是提到), 不出题


def person_forms(people: list[Any]) -> list[set[str]]:
    """每个人用来在原文里认人的写法 (简体): 姓名, 姓 + 字, 别名表记下的原文称呼 (orig).
    去掉三种容易误认的: 单字 (如 "權"); 官称 (kind 为 "官称", 如 "大将军", 同一官职很多人做过); 表里两个人以上共用的写法
    (如 "都亭侯" "太祖": 多人都封过 / 都被这样称呼, 认出来也不知道是谁)."""
    raw = []
    for p in people:
        forms = {p.get("name", "")} | {f.get("orig", "") for f in p.get("forms", []) if f.get("kind") != "官称"}
        if p.get("zi") and p.get("surname"):
            forms.add(p["surname"] + p["zi"])
        raw.append({to_simplified(f) for f in forms if len(f) >= 2})
    shared = {f for f, n in Counter(f for fs in raw for f in fs).items() if n > 1}
    return [fs - shared if fs - shared else fs for fs in raw]  # 全被去掉时 (名字和别人相同) 保留原样, 免得这个人认不出来


def people_in(text: str, people: list[Any], forms: list[set[str]] | None = None) -> list[Any]:
    """一段原文里出现的别名表人物 (rag/alias 的 people): person_forms 里的任一写法出现在原文 (转简体) 里就算.
    forms 是 person_forms(people) 的结果, 反复调用时传进来省得重算."""
    t = to_simplified(text)
    forms = forms if forms is not None else person_forms(people)
    return [p for p, fs in zip(people, forms) if any(f in t for f in fs)]


POP_WEIGHT = 0.5  # 挑人的权重 = 出现块数 ** POP_WEIGHT: 0 = 人人机会相同; 0.5 时模拟 2 万局曹操约 1.7%, 出现最多的 50 人合计约 28%
# 别名表人物 -> 在全书正文块里出现的块 (下标), 按 people 列表的位置存. random_person 第一次调用时算
# (rag/skills.make_router 启动时在后台线程先算); 统计完成前开局的那一局会等它算完 (加锁).
_APPEAR: list[list[int]] = []
_PEOPLE: list[Any] = []
_appear_lock = threading.Lock()


def _pool(lib: Library) -> tuple[list[Any], list[list[int]]]:
    """出题用的人和他们出现的正文块 (只算 tier 1 的正文块; 一次都没出现的人不出题)."""
    from rag.alias import load_aliases
    with _appear_lock:
        if not _PEOPLE:
            people = [p for p in load_aliases().get("people", [])
                      if len(p.get("name", "")) >= 2 and p.get("id") and p["name"] not in NOT_THREE_KINGDOMS]
            forms = person_forms(people)
            appear: list[list[int]] = [[] for _ in people]
            for i, c in enumerate(lib.chunks):
                if c.get("tier") == 1:
                    t = to_simplified(c["text"])
                    for k, fs in enumerate(forms):
                        if any(f in t for f in fs):
                            appear[k].append(i)
            _PEOPLE[:], _APPEAR[:] = people, appear
        return _PEOPLE, _APPEAR


def random_person(lib: Library, rng: random.Random | None = None) -> tuple[Person, str]:
    """随机出题: 在别名表里在正文出现过的人中挑一个, 权重 = 出现块数 ** POP_WEIGHT (名人稍多, 但不独占);
    再从他出现的正文块里随机取一块. 返回 (谜底, 这一块的 id). 谜底的 bio = [本传首段源 id, 这一块的源 id]
    (裁判预载这两处原文); aliases = 别名表里的称呼. 挑人用代码随机而不交给模型: 模型总偏向最有名的几个人."""
    r = rng or random
    people, appear = _pool(lib)
    ks = [k for k, a in enumerate(appear) if a]
    k = r.choices(ks, weights=[len(appear[k]) ** POP_WEIGHT for k in ks])[0]
    p, c = people[k], lib.chunks[r.choice(appear[k])]
    aliases = sorted({f["form"] for f in p.get("forms", []) if f.get("form")} - {p["name"]})
    bio = [p["id"]] + [s for s in c["sources"] if s != p["id"]][:1]
    note = f"字{p['zi']}" if p.get("zi") else ""
    person: Person = {"id": p["id"], "person": p["name"], "aliases": aliases, "bio": bio, "note": note}
    if p.get("notrans"):
        person["notrans"] = True
    return person, c["id"]


def new_game(lib: Library, complete: Callable[..., Any], rng: random.Random | None = None,
             pool: list[Person] | None = None) -> GuessGame:
    """开一局 "Agent 出题": 默认用 random_person 从全书随机出题; 给定谜底列表 pool 时从中抽 (测试用).
    known_names 取别名表里的人名, 用来认出 "刘备？" 这种只说名字的猜测."""
    from rag.alias import load_aliases
    person = (rng or random).choice(pool) if pool is not None else random_person(lib, rng)[0]
    known = {to_simplified(p["name"]) for p in load_aliases().get("people", []) if len(p.get("name", "")) >= 2}
    return GuessGame(person, lib, complete, known_names=known)


AGENT_SAFETY_MAX = 50  # Agent 来猜时问问题不限次数, 只在问到这么多次还没猜中时认输 (防止模型一直问下去, 不告诉玩家和模型)
AGENT_ROUNDS = 4       # Agent 每一轮最多几轮工具调用
_YES = {"是", "是的", "对", "对的", "对了", "嗯", "有", "属于", "算", "算是", "没错", "正确", "猜对了", "是他", "是她",
        "yes", "y", "yep", "correct"}
_NO = {"不是", "否", "不对", "错", "错了", "没有", "不属于", "不算", "不是他", "不是她", "猜错了", "no", "n", "nope", "wrong"}
_UNKNOWN = {"不知道", "不清楚", "不确定", "无法判断", "没写", "说不准", "unknown"}
FREE_MAX = 40  # 认不出的回答在这个字数以内就原话交给 Agent (如 "都待过"); 更长的多半跑题了, 请玩家就问题作答
_REMIND = "请只输出一行 JSON: {\"ask\": ...} 或 {\"guess\": ..., \"desc\": ...}."


def _answer_kind(text: str) -> str | None:
    """玩家的回答归成 是 / 不是 / 不知道; 认不出返回 None."""
    t = re.sub(r"[\s。.!！?？~～,，]+", "", text).casefold()
    return "是" if t in _YES else "不是" if t in _NO else "不知道" if t in _UNKNOWN else None


class AgentGuessGame:
    """Agent 来猜: 玩家心里想一个人, Agent 问是非问题 (可以用检索工具查原文), 最后猜名字, 玩家答 对 / 错.

    整局是同一段对话 (GUESS_SYS 是 system prompt); 每一轮用 play() 跑 agent 循环, 并删掉循环追加的 developer 提醒.
    问问题不限次数, 名字最多猜 max_guesses 次: 猜错时告诉 Agent 还能猜几次, 猜错 max_guesses 次就认输.
    问和猜合计到 safety_max 轮还没猜中也认输 (只防模型一直问下去). Agent 没给出问题或名字时提醒一次.
    complete 签名同 judge_question 的 complete; turns 记录每一轮 {"n", "ask" 或 "guess" + "desc", "tools"}."""

    def __init__(self, lib: Library, complete: Callable[..., Any], max_guesses: int = MAX_GUESSES,
                 safety_max: int = AGENT_SAFETY_MAX) -> None:
        self.lib, self.complete, self.max, self.safety_max = lib, complete, max_guesses, safety_max
        self.guesses = 0  # 猜错了几次名字
        self.msgs: list[dict] = [{"role": "system", "content": GUESS_SYS.format(g=max_guesses)},
                                 {"role": "user", "content": "我想好了一个人, 开始吧."}]
        self.turns: list[dict] = []
        self.seen = Seen(lib.notes)
        self.over = False
        self.pending = False  # 上一次想下一问时超时了: 玩家下一句话不当回答, 直接重试

    def intro(self, on_event: OnEvent | None = None) -> str:
        """开局: 说明玩法, 并给出第一个问题."""
        return ("你心里想好一个《三国志》里的人物, 我来问. 请回答 是 / 不是 / 不知道, 也可以简短说明 (如 \"都待过\");"
                " 我猜名字时回答 对 / 错."
                f" 我问问题不限次数, 名字最多猜 {self.max} 次. /放弃 我认输, /退出 结束游戏.\n\n" + self._next(on_event))

    def _next(self, on_event: OnEvent | None = None) -> str:
        """让 Agent 想下一问 (或猜名字). 超时时恢复对话历史, 标记 pending, 请玩家随便说一句再试."""
        snapshot = list(self.msgs)
        try:
            return self._think(on_event)
        except TimeoutError:
            self.msgs[:] = snapshot
            self.pending = True
            return "这一问我想得太久了, 没想出来. 请随便回复一句 (如 \"继续\") 让我再想一次."
        except BaseException:  # 用户停止: 对话回到这一轮之前
            self.msgs[:] = snapshot
            raise

    def _think(self, on_event: OnEvent | None) -> str:
        n = len(self.turns) + 1
        if n > self.safety_max:
            self.over = True
            return f"问了 {self.safety_max} 次还没猜出来, 我认输. 你想的是谁?"
        log: list[dict] = []
        reply = play(self.msgs, self.lib, self.seen, log, self.complete, AGENT_ROUNDS, on_event)
        if not field(reply, "guess") and not field(reply, "ask"):
            self.msgs += [{"role": "assistant", "content": reply}, {"role": "user", "content": _REMIND}]
            reply = play(self.msgs, self.lib, self.seen, log, self.complete, 1, on_event)
        self.msgs.append({"role": "assistant", "content": reply})
        name = field(reply, "guess")
        turn: dict[str, Any] = ({"n": n, "guess": name, "desc": field(reply, "desc") or ""} if name
                                else {"n": n, "ask": field(reply, "ask") or ""})
        turn["tools"] = log
        self.turns.append(turn)
        if name:
            desc = f" ({turn['desc']})" if turn["desc"] else ""
            return f"第 {n} 次: 我猜是 {name}{desc}. 对吗?"
        return f"第 {n} 问: {turn['ask'] or '(我没想好问题, 请随便回答一句让我继续)'}"

    def ask(self, text: str, on_event: OnEvent | None = None) -> str | None:
        """玩家的回答, 返回 Agent 的下一问 (或结局). 上一轮是问题时: 认得出的归成 是 / 不是 / 不知道 (_YES / _NO / _UNKNOWN),
        认不出但不超过 FREE_MAX 字的原话交给 Agent (如 "都待过", GUESS_SYS 让它按意思理解); 更长的返回 None, 交给问答 agent.
        上一轮是猜名字时只认 对 / 错."""
        if self.over:
            return "这一局已经结束了."
        if self.pending:
            self.pending = False
            return self._next(on_event)
        kind = _answer_kind(text)
        last = self.turns[-1]
        text = text.strip()
        if "guess" in last:
            if kind is None or kind == "不知道":
                return "请回答 对 / 错 (猜名字时只认对错)."
            if kind == "是":
                self.over = True
                return f"太好了, 是 {last['guess']}! 我用了 {last['n']} 次."
            self.guesses += 1
            if self.guesses >= self.max:
                self.over = True
                return f"{self.max} 次都没猜中, 我认输. 你想的是谁?"
            self.msgs.append({"role": "user", "content": f"不是 {last['guess']}. (你还能猜 {self.max - self.guesses} 次.)"})
        else:
            if kind is None and (not text or len(text) > FREE_MAX):
                return None  # 太长的多半不是在回答这个问题 (问玩法, 想结束……), 交给问答 agent
            answer = kind or text  # 认得出的归成三种之一, 认不出的短回答原话交给 Agent
            last["answer"] = answer
            self.msgs.append({"role": "user", "content": answer})
        return self._next(on_event)

    def reveal(self) -> str:
        """玩家让 Agent 认输."""
        self.over = True
        return "好吧, 我认输. 你想的是谁?"


class GuessPerson:
    """猜人物 skill 的一局: 先让玩家选玩法 (或 start_game 带 mode 时由 choose 直接定) (1 = Agent 出题玩家猜, GuessGame; 2 = 玩家出题 Agent 猜, AgentGuessGame), 之后转交."""
    _USER = {"1", "我猜", "你出题", "我来猜"}
    _AGENT = {"2", "你猜", "我出题", "你来猜"}

    def __init__(self, user_game: Callable[[], Any], agent_game: Callable[[], Any]) -> None:
        self.user_game, self.agent_game = user_game, agent_game
        self.game: Any = None
        self._over = False

    @property
    def over(self) -> bool:
        return self._over or bool(self.game and self.game.over)

    def intro(self, on_event: OnEvent | None = None) -> str:
        return "两种玩法: 回复 1 = 我想一个人, 你来猜; 回复 2 = 你想一个人, 我来猜."

    MODES = {"player_guesses": "1", "agent_guesses": "2"}  # SKILL.md 的 modes -> 开局时回复的编号

    def ask(self, text: str, on_event: OnEvent | None = None) -> str | None:
        """还没选玩法时认 1 / 2 (和 _USER / _AGENT 里的说法), 认不出返回 None (交给问答 agent); 选了之后转交."""
        if self.game is not None:
            return self.game.ask(text, on_event)
        t = re.sub(r"[\s。.!！?？]+", "", text)
        if t in self._USER:
            self.game = self.user_game()
        elif t in self._AGENT:
            self.game = self.agent_game()
        else:
            return None
        return self.game.intro(on_event)

    def choose(self, mode: str, on_event: OnEvent | None = None) -> str:
        """按 mode (player_guesses = 玩家猜, agent_guesses = Agent 猜) 直接开局, 不再问选哪种 (start_game 带了 mode 时)."""
        return self.ask(self.MODES[mode], on_event) or self.intro(on_event)

    def reveal(self) -> str:
        if self.game is None:
            self._over = True
            return "还没开始, 已结束."
        return self.game.reveal()
