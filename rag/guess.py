"""猜人物, 两种玩法 (GuessPerson 开局时让玩家选):
    1. Agent 出题, 玩家来猜 (GuessGame): Agent 心里定一个《三国志》人物, 玩家问是非问题, 最后说出名字.
    2. 玩家出题, Agent 来猜 (AgentGuessGame): 玩家心里想一个人, Agent 问是非问题 (GUESS_SYS, 可以查原文), 最后猜名字.

    uv run python -m rag.skills          # 命令行里玩 (输入 "猜人物" 开局), 见 rag/skills.py
    uv run python -m tests.test_skills   # 假模型测试, 不连网

玩法 1 的谜底池 data/guess/checks.jsonl, 一行一个人物, 只有 "playable": true 的人会被抽到 (身份明确, 原文有足够记载):
    {"id": "g-06", "tier": "rare", "person": "马忠", "aliases": ["狐笃", "马德信"], "bio": ["43-0011"],
     "homonym": true, "note": "...", "playable": true, "questions": [...]}
    bio      这个人本传或出处的源 id, 裁判先读这些段落
    homonym  有同名的人 (如蜀汉马忠 / 孙吴马忠): 揭晓时说明是哪一位
    note     一句话说明是哪一位, 揭晓时显示; 没有就只显示名字和出处

玩法 1 的裁判 (judge_question): 每个问题单独开一段对话, 谜底的出处原文预先给它, 需要时再用工具查; 它回一句话
(是 / 不是 + 简短说明, 二选一的答一个选项, 原文没有的说 "历史上没有记载"), 页面只显示这一句和计数.
代码只检查这句话里有没有谜底的名字或称呼 (rag/soup.leaks), 有就换成 LEAK_REPLY. 玩家猜名字时由代码比对人名和称呼 (matches),
不交给模型判断.
"""
import json
import random
import re
import time
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable, TypedDict

from rag import agent, config
from rag.library import Library, Seen
from rag.loop import agent_loop
from rag.normalize import to_simplified
from rag.soup import LEAK_REPLY, _parse_json, leaks

CHECKS = config.DATA_DIR / "guess" / "checks.jsonl"
VERDICTS = ("是", "不是", "选择", "没有记载", "没法回答")  # 判词类别; 给玩家看的是 answer 那一句话
NOT_COUNTED = "没法回答"  # 这一类不算一问
MAX_QUESTIONS = 20
JUDGE_ROUNDS = 3  # 裁判每个问题最多几轮工具调用
TURN_BUDGET = 90.0  # 一问 (一次 play) 最多花几秒: 超过后不再发新请求, 抛 TimeoutError, 由游戏提示玩家再说一次
OnEvent = Callable[[str, Any], None]  # on_event(kind, info): 进度事件, kind 是 "tool" / "stage" / "think" (见 play)

JUDGE_SYS = """你是 "猜人物" 游戏的裁判. 谜底是《三国志》里的 {person} (原文里也称 {aliases}; 本传或出处 {bio}). 玩家不知道谜底,
会向你提问. 这个人的出处原文已经附在下面 (编号 [n]), 先看这些资料; 问到资料里没有的事 (如和别人的交往, 别处的记载)
才用工具查《三国志》原文, 不要把下面已有的段落再 read 一遍.

{bio_text}

怎么答 (玩家多半凭印象提问, 按他最可能的意思答, 不抠字眼):
- 是非题答 是 / 不是, 可以带一句很短的说明 (如 "是, 但不是蜀国的" "不是, 他以谋士为主").
- 二选一的问题 (如 "是南方人还是北方人" "主要在魏国还是吴国") 直接答其中一个选项.
- 问法有歧义时按最可能的意思答: "是蜀国人或者官员吗" 是问 "是不是蜀国的人或官员", 不是问 "是蜀国人, 或者是官员".
- 原文写明或能直接推出的, 答是或不是; 查过原文确实没有记载的, 答 "历史上没有记载".
- 实在没法用是 / 不是或一个选项回答的 (如 "他是谁" "讲讲他的生平") -> 请玩家换成是非题或选择题.
- 只依据查到的原文, 不用自己的历史知识. 注意同名的人, 不要查成别人.
- 回答不超过一句话, 不能写出谜底的名字、字、称号, 也不能直接说出他的姓.
最后只输出一行 JSON: {{"verdict": "是 | 不是 | 选择 | 没有记载 | 没法回答", "answer": "给玩家看的那一句话", "evidence": [依据的资料编号 n]}}"""

GUESS_SYS = """你在玩 "猜人物": 对方心里想了一个三国人物 (多半是《三国志》里的人, 也可能是只在《三国演义》里出名的人物),
你要用是非问题猜出是谁. 你最多问 {n} 次 (猜名字也算一次).

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
    """谜底池的一行, 字段见模块 docstring."""
    id: str
    tier: str
    person: str
    aliases: list[str]
    bio: list[str]
    homonym: bool
    note: str
    playable: bool
    notrans: bool  # 三国志里没有自己的传 (bio 第一个 id 只是提到他的一处), 揭晓时不说 "他的传在……"


def load_pool(path: Path = CHECKS) -> list[Person]:
    """谜底池里可以出题的人 (playable 为 true)."""
    with open(path, encoding="utf-8") as f:
        return [p for p in map(json.loads, (line for line in f if line.strip())) if p.get("playable")]


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


def judge_question(lib: Library, person: Person, question: str, complete: Callable[..., Any],
                   on_event: OnEvent | None = None) -> dict:
    """裁判玩家的一个问题. 返回 {"verdict": VERDICTS 之一 (类别), "answer": 给玩家看的一句话, "by": "model",
    "tools": [工具调用记录], "raw": 模型回答末尾}.

    complete(messages, tools=None) -> 模型的一条回复 message (rag/loop.agent_loop 的约定). 是非题、二选一、问法有歧义的问题
    都交给模型按意思回答, 代码不预先拦; 唯一的代码检查是 leaks: 回答里出现谜底的名字或称呼就换成 LEAK_REPLY.
    每个问题单独开一段对话 (裁判互不记忆); 谜底的出处原文 (bio_chunks, 经 Seen 编号) 预先放进
    system prompt, 大多数问题不用再调工具, 资料以外的事才查; 模型输出解析不了或判词不在 VERDICTS 里时判 "无法判断".
    on_event 只收到 ("stage", "查原文") (裁判的推理和检索词会泄底, 不发). 超过 TURN_BUDGET 时抛 TimeoutError."""
    seen, log = Seen(lib.notes), []
    sys = JUDGE_SYS.format(person=person["person"], aliases="、".join(person.get("aliases", [])) or "无",
                           bio=", ".join(person["bio"]), bio_text=seen.add(bio_chunks(lib, person["bio"])))
    reply = play([{"role": "system", "content": sys}, {"role": "user", "content": question}], lib, seen, log,
                 complete, JUDGE_ROUNDS, on_event, show=False)
    verdict = field(reply, "verdict")
    verdict = verdict if verdict in VERDICTS else "没有记载"
    answer = (field(reply, "answer") or "").strip() or {"没有记载": "历史上没有记载"}.get(verdict, verdict)
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
    """一局猜人物: 谜底, 已问的问题数, 是否结束. 不依赖界面, 测试和 rag/skills.py 直接调它.

    lib 是 rag.library.load_library() 的结果; complete 签名同 judge_question 的 complete."""

    def __init__(self, person: Person, lib: Library, complete: Callable[..., Any], max_questions: int = MAX_QUESTIONS,
                 known_names: set[str] | None = None) -> None:
        self.p, self.lib, self.complete, self.max = person, lib, complete, max_questions
        self.known_names = known_names or set()
        self.asked = 0
        self.over = False

    def intro(self, on_event: OnEvent | None = None) -> str:
        """开局说明."""
        return (f"我心里想好了一个《三国志》里的人物. 你可以问 {self.max} 个问题 (是非题或二选一, 我查原文回答),"
                " 猜名字时说 \"猜 某某\" 或直接说名字. 输入 /放弃 揭晓答案, /退出 结束游戏.")

    def ask(self, text: str, on_event: OnEvent | None = None) -> str:
        """处理玩家的一句话: 猜名字就由代码比对, 否则当作是非问题交给裁判. 返回界面上显示的一行.
        裁判超时时这一问不计数, 请玩家再问一次."""
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
            self.asked += 1
            return f"不是 {name}. ({self.asked}/{self.max})" + self._left()
        if self.asked >= self.max:
            return f"{self.max} 个问题用完了, 猜一个名字吧, 或者输入 /放弃 揭晓."
        try:
            r = judge_question(self.lib, self.p, text, self.complete, on_event)
        except TimeoutError:
            return "这一问查原文太久了, 没判出来 (不计数). 请再问一次."
        if r["verdict"] != NOT_COUNTED and r["answer"] != LEAK_REPLY:
            self.asked += 1
        return f"{r['answer']}  ({self.asked}/{self.max})" + self._left()

    def _left(self) -> str:
        return " 问题用完了, 猜一个名字吧." if self.asked >= self.max and not self.over else ""

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


def people_in(text: str, people: list[Any]) -> list[Any]:
    """一段原文里出现的别名表人物 (rag/alias 的 people): 姓名, 字, 或别名表记下的原文称呼 (orig) 出现在原文里就算.
    原文和名字都转简体再比; 单字的称呼 (如 "權") 太容易误认, 不算."""
    t = to_simplified(text)
    out = []
    for p in people:
        forms = {p.get("name", "")} | {f.get("orig", "") for f in p.get("forms", [])}
        if p.get("zi") and p.get("surname"):
            forms.add(p["surname"] + p["zi"])
        if any(len(f) >= 2 and to_simplified(f) in t for f in forms):
            out.append(p)
    return out


_COUNTS: dict[str, int] = {}  # 别名表人物 id -> 在全书正文块里出现的块数 (random_person 第一次调用时算)


def random_person(lib: Library, rng: random.Random | None = None, tries: int = 50) -> tuple[Person, str]:
    """真随机出题: 从全书正文块里随机抽一块, 找出其中出现的别名表人物 (people_in), 再随机挑一个; 一个都没有就重抽.
    挑人的权重 = 1 / 这个人出现的块数: 不然曹操 (魏书大量段落有 "太祖") 会占约 15%, 加权后各人被选中的机会接近.
    返回 (谜底, 抽到的块 id). 谜底的 bio = [本传首段源 id, 抽到的块的源 id] (裁判预载这两处原文);
    aliases = 别名表里的称呼. 挑人用代码随机而不交给模型: 模型总偏向一块里最有名的人 (魏书大量段落都有曹操)."""
    from rag.alias import load_aliases
    r = rng or random
    people = [p for p in load_aliases().get("people", [])
              if len(p.get("name", "")) >= 2 and p.get("id") and p["name"] not in NOT_THREE_KINGDOMS]
    body = [c for c in lib.chunks if c.get("tier") == 1]
    if not _COUNTS:
        for c in body:
            for p in people_in(c["text"], people):
                _COUNTS[p["id"]] = _COUNTS.get(p["id"], 0) + 1
    for _ in range(tries):
        c = r.choice(body)
        found = people_in(c["text"], people)
        if found:
            p = r.choices(found, weights=[1 / _COUNTS.get(x["id"], 1) for x in found])[0]
            aliases = sorted({f["form"] for f in p.get("forms", []) if f.get("form")} - {p["name"]})
            bio = [p["id"]] + [s for s in c["sources"] if s != p["id"]][:1]
            note = f"字{p['zi']}" if p.get("zi") else ""
            person: Person = {"id": p["id"], "person": p["name"], "aliases": aliases, "bio": bio, "note": note}
            if p.get("notrans"):
                person["notrans"] = True
            return person, c["id"]
    raise RuntimeError(f"抽了 {tries} 块都没找到别名表里的人")


def new_game(lib: Library, complete: Callable[..., Any], rng: random.Random | None = None,
             pool: list[Person] | None = None) -> GuessGame:
    """开一局 "Agent 出题": 默认用 random_person 从全书随机出题; 给了 pool 时从 pool 里抽 (测试用).
    known_names 取别名表里的人名, 用来认出 "刘备？" 这种只说名字的猜测."""
    from rag.alias import load_aliases
    person = (rng or random).choice(pool) if pool is not None else random_person(lib, rng)[0]
    known = {to_simplified(p["name"]) for p in load_aliases().get("people", []) if len(p.get("name", "")) >= 2}
    return GuessGame(person, lib, complete, known_names=known)


AGENT_MAX = 25        # Agent 来猜时最多几次 (问问题和猜名字都算)
AGENT_HINT_AFTER = 20  # 问过这么多次以后, 每轮告诉模型还剩几次 (模型自己不数)
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
    问满 AGENT_HINT_AFTER 次后每轮附上 "还剩 N 次". Agent 没给出问题或名字时提醒一次.
    complete 签名同 judge_question 的 complete; turns 记录每一轮 {"n", "ask" 或 "guess" + "desc", "tools"}."""

    def __init__(self, lib: Library, complete: Callable[..., Any], max_turns: int = AGENT_MAX,
                 hint_after: int = AGENT_HINT_AFTER) -> None:
        self.lib, self.complete, self.max, self.hint_after = lib, complete, max_turns, hint_after
        self.msgs: list[dict] = [{"role": "system", "content": GUESS_SYS.format(n=max_turns)},
                                 {"role": "user", "content": "我想好了一个人, 开始吧."}]
        self.turns: list[dict] = []
        self.seen = Seen(lib.notes)
        self.over = False
        self.pending = False  # 上一次想下一问时超时了: 玩家下一句话不当回答, 直接重试

    def intro(self, on_event: OnEvent | None = None) -> str:
        """开局: 说明玩法, 并给出第一个问题."""
        return ("你心里想好一个《三国志》里的人物, 我来问. 请回答 是 / 不是 / 不知道, 也可以简短说明 (如 \"都待过\");"
                " 我猜名字时回答 对 / 错."
                f" 我最多问 {self.max} 次. /放弃 我认输, /退出 结束游戏.\n\n" + self._next(on_event))

    def _next(self, on_event: OnEvent | None = None) -> str:
        """让 Agent 想下一问 (或猜名字). 超时时恢复对话历史, 标记 pending, 请玩家随便说一句再试."""
        snapshot = list(self.msgs)
        try:
            return self._think(on_event)
        except TimeoutError:
            self.msgs[:] = snapshot
            self.pending = True
            return "这一问我想得太久了, 没想出来. 请随便回复一句 (如 \"继续\") 让我再想一次."

    def _think(self, on_event: OnEvent | None) -> str:
        n = len(self.turns) + 1
        if n > self.max:
            self.over = True
            return f"{self.max} 次用完了, 我没猜出来. 你想的是谁?"
        if n > self.hint_after:
            self.msgs[-1]["content"] += f"\n(还剩 {self.max - n + 1} 次.)"
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

    def ask(self, text: str, on_event: OnEvent | None = None) -> str:
        """玩家的回答, 返回 Agent 的下一问 (或结局). 上一轮是问题时: 认得出的归成 是 / 不是 / 不知道 (_YES / _NO / _UNKNOWN),
        认不出但不超过 FREE_MAX 字的原话交给 Agent (如 "都待过", GUESS_SYS 让它按意思理解), 更长的请玩家就问题作答.
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
            self.msgs.append({"role": "user", "content": f"不是 {last['guess']}."})
        else:
            if kind is None and (not text or len(text) > FREE_MAX):
                return f"请就这个问题回答: 是 / 不是 / 不知道, 或者简短说明 ({FREE_MAX} 字以内, 如 \"都待过\" \"前期是\")."
            answer = kind or text  # 认得出的归成三种之一, 认不出的短回答原话交给 Agent
            last["answer"] = answer
            self.msgs.append({"role": "user", "content": answer})
        return self._next(on_event)

    def reveal(self) -> str:
        """玩家让 Agent 认输."""
        self.over = True
        return "好吧, 我认输. 你想的是谁?"


class GuessPerson:
    """猜人物 skill 的一局: 先让玩家选玩法 (1 = Agent 出题玩家猜, GuessGame; 2 = 玩家出题 Agent 猜, AgentGuessGame), 之后转交."""
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

    def ask(self, text: str, on_event: OnEvent | None = None) -> str:
        if self.game is not None:
            return self.game.ask(text, on_event)
        t = re.sub(r"[\s。.!！?？]+", "", text)
        if t in self._USER:
            self.game = self.user_game()
        elif t in self._AGENT:
            self.game = self.agent_game()
        else:
            return "请回复 1 (我出题, 你来猜) 或 2 (你出题, 我来猜)."
        return self.game.intro(on_event)

    def again_like(self, old: Any, on_event: OnEvent | None = None) -> str:
        """"再来一局": 上一局选过玩法就沿用 (直接开局), 没选过就照常问."""
        prev = getattr(old, "game", None)
        if isinstance(prev, GuessGame):
            self.game = self.user_game()
        elif isinstance(prev, AgentGuessGame):
            self.game = self.agent_game()
        else:
            return self.intro(on_event)
        return self.game.intro(on_event)

    def reveal(self) -> str:
        if self.game is None:
            self._over = True
            return "还没开始, 已结束."
        return self.game.reveal()
