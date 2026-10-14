"""猜人物 (Agent 出题, 玩家来猜): Agent 心里定一个《三国志》人物, 玩家问是非问题, 最后说出名字.

    uv run python -m rag.skills          # 命令行里玩 (输入 "猜人物" 开局), 见 rag/skills.py
    uv run python -m tests.test_skills   # 假模型测试, 不连网

谜底池 data/guess/checks.jsonl, 一行一个人物, 只有 "playable": true 的人会被抽到 (身份明确, 原文有足够记载):
    {"id": "g-06", "tier": "rare", "person": "马忠", "aliases": ["狐笃", "马德信"], "bio": ["43-0011"],
     "homonym": true, "note": "...", "playable": true, "questions": [...]}
    bio      这个人本传或出处的源 id, 裁判先读这些段落
    homonym  有同名的人 (如蜀汉马忠 / 孙吴马忠): 揭晓时说明是哪一位
    note     一句话说明是哪一位, 揭晓时显示; 没有就只显示名字和出处

裁判 (judge_question): 每个问题单独开一段对话, 模型用 search / read / who 等工具查原文后给判词 (VERDICTS).
页面只显示判词和计数, 不显示模型写的其他文字, 所以就算模型被诱导写出名字也不会展示出去; 直接打听谜底本身的问题
(如 "他名字第一个字是什么") 由代码先拦下 (rag/soup.is_meta 加上 _META_NAME). 玩家猜名字时由代码比对人名和称呼 (matches),
不交给模型判断.
"""
import json
import random
import re
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable, TypedDict

from rag import agent, config
from rag.library import Library, Seen
from rag.loop import agent_loop
from rag.normalize import to_simplified
from rag.soup import _parse_json, is_meta

CHECKS = config.DATA_DIR / "guess" / "checks.jsonl"
VERDICTS = ("是", "不是", "无法判断", "请问是非题")
MAX_QUESTIONS = 20
JUDGE_ROUNDS = 3  # 裁判每个问题最多几轮工具调用

JUDGE_SYS = """你是 "猜人物" 游戏的裁判. 谜底是《三国志》里的 {person} (原文里也称 {aliases}; 本传或出处 {bio}). 玩家不知道谜底,
会问是非问题. 你手边有检索《三国志》原文的工具: 先用工具查这个人的原文 (可以直接 read 上面的出处 id), 再判断.
判断规则: 原文写明或能直接推出 -> 是 / 不是; 原文确实没写, 推不出来 -> 无法判断 (没写不等于 "不是"); 问题不是能用是 / 不是回答的 -> 请问是非题.
只依据查到的原文, 不用自己的历史知识. 注意同名的人, 不要查成别人.
最后只输出一行 JSON: {{"verdict": "是 | 不是 | 无法判断 | 请问是非题", "evidence": [依据的资料编号 n]}}"""

# 打听名字本身的问题 (猜人物特有; 汤底 / 答案 / 谜底等由 rag/soup.is_meta 拦)
_META_NAME = re.compile(r"名字|姓什么|叫什么|哪个字|第[一二三四五六七八九十\d]+个字|几个字|拼音|首字母")
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


def play(msgs: list[dict], lib: Library, seen: Seen, log: list[dict], complete: Callable[..., Any], rounds: int) -> str:
    """跑一轮 agent_loop, 再删掉它追加的 developer 提醒 ("工具调用次数已经用完" 等).
    同一段对话要接着用时, 这些提醒留在历史里会让模型以为再也不能查原文."""
    reply = agent_loop(msgs, agent.TOOLS, complete, tool_runner(lib, seen, log), max_rounds=rounds)
    msgs[:] = [m for m in msgs if m.get("role") != "developer"]
    return reply


def judge_question(lib: Library, person: Person, question: str, complete: Callable[..., Any]) -> dict:
    """裁判一个是非问题. 返回 {"verdict": VERDICTS 之一, "by": "code" | "model", "tools": [工具调用记录], "raw": 模型回答末尾}.

    complete(messages, tools=None) -> 模型的一条回复 message (rag/loop.agent_loop 的约定). 打听谜底本身的问题由代码判
    "请问是非题", 不请求模型. 每个问题单独开一段对话 (裁判互不记忆); 模型输出解析不了或判词不在 VERDICTS 里时判 "无法判断"."""
    if is_meta(question) or _META_NAME.search(question):
        return {"verdict": "请问是非题", "by": "code", "tools": [], "raw": None}
    seen, log = Seen(lib.notes), []
    sys = JUDGE_SYS.format(person=person["person"], aliases="、".join(person.get("aliases", [])) or "无",
                           bio=", ".join(person["bio"]))
    reply = play([{"role": "system", "content": sys}, {"role": "user", "content": question}], lib, seen, log,
                 complete, JUDGE_ROUNDS)
    verdict = field(reply, "verdict")
    return {"verdict": verdict if verdict in VERDICTS else "无法判断", "by": "model", "tools": log, "raw": reply[-300:]}


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

    def intro(self) -> str:
        """开局说明."""
        return (f"我心里想好了一个《三国志》里的人物. 你可以问 {self.max} 个是非问题 (我只答 是 / 不是 / 无法判断),"
                " 猜名字时说 \"猜 某某\" 或直接说名字. 输入 /放弃 揭晓答案, /退出 结束游戏.")

    def ask(self, text: str) -> str:
        """处理玩家的一句话: 猜名字就由代码比对, 否则当作是非问题交给裁判. 返回界面上显示的一行."""
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
        r = judge_question(self.lib, self.p, text, self.complete)
        if r["verdict"] != "请问是非题":
            self.asked += 1
        return f"{r['verdict']}  ({self.asked}/{self.max})" + self._left()

    def _left(self) -> str:
        return " 问题用完了, 猜一个名字吧." if self.asked >= self.max and not self.over else ""

    def reveal(self) -> str:
        """放弃, 直接揭晓."""
        self.over = True
        return self.reveal_text()

    def reveal_text(self) -> str:
        """谜底: 人名, 说明 (同名时说清是哪一位), 原文出处."""
        note = f" ({self.p['note']})" if self.p.get("note") else ""
        return f"谜底是 {self.p['person']}{note}. 出处: {', '.join(self.p['bio'])}."


def new_game(lib: Library, complete: Callable[..., Any], rng: random.Random | None = None,
             pool: list[Person] | None = None) -> GuessGame:
    """从谜底池随机抽一个人开一局. known_names 取别名表里的人名, 用来认出 "刘备？" 这种只说名字的猜测."""
    from rag.alias import load_aliases
    people = pool if pool is not None else load_pool()
    known = {to_simplified(p["name"]) for p in load_aliases().get("people", []) if len(p.get("name", "")) >= 2}
    return GuessGame((rng or random).choice(people), lib, complete, known_names=known)
