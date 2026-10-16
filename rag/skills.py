"""三国志 Agent 的 skill: 问答 agent 可以和玩家玩的游戏 (猜人物, 海龟汤).

    uv run python -m rag.skills --openrouter     # 命令行里玩: 输入 "/猜人物" 或 "/海龟汤" 开局 (命令行不接问答 agent)
    uv run python -m tests.test_skills           # 假模型测试, 不连网

每个 skill 是 skills/<name>/SKILL.md: 开头的 YAML frontmatter 给名字、说明和快捷命令, 正文是玩法说明.

    skills/guess_person/SKILL.md
        ---
        name: guess_person                                     和 make_router 里的游戏工厂对应
        title: 猜人物
        description: 两种玩法, 1 = Agent 想一个人玩家来猜, ……    写进问答 agent 的 system prompt (skill 列表)
        modes: {player_guesses: ……, agent_guesses: ……}         可选: start_game 的 mode 参数可以取的值
        triggers: ["/猜人物", "/guess_person"]                  快捷命令 (网页按钮发的就是它), 不经过模型直接开局
        ---
        玩法: ……                                               开局后给问答 agent 看, /帮助 时也显示

开局 (在对话里): 问答 agent 的 system prompt 里有 skill 列表 (skill_prompt), tools 里多两个工具 (Router.tools):
    start_game(game, mode?)   开一局; 玩家说 "来玩猜人物吧" "你来猜我想的人" 时由 agent 决定调用
    end_game(reveal)          结束正在玩的一局 (玩家说 "不玩了")
handler (Router.handlers) 执行后抛 SkillReply, 把游戏的回复 (开局说明 / 揭晓) 原样交给界面, 这一问不再请求模型.
整句是快捷命令 (match_skill, 不分大小写, 忽略句末标点) 时直接开局, 不经过模型.

游戏进行中: 玩家的消息先交给这局游戏 (Router.handle). 游戏判断这句话不是游戏里的话 (问玩法、闲聊、想结束、问别的历史问题;
裁判判词 "非游戏") 时返回 None, 由界面交给问答 agent, 并在 system prompt 里附上正在玩的游戏和玩法 (Router.agent_note);
问答 agent 不知道谜底. 游戏中的命令 (整句匹配, 不经过模型):
    /退出 /quit /exit          结束这一局
    /放弃 /揭晓 /giveup        揭晓答案并结束
    /提示 /hint                要一条提示 (游戏有提示时)
    /答 <解释> /answer <解释>   提交完整解释 (海龟汤); 猜人物里等同于猜名字
    /帮助 /help                再看一遍 SKILL.md 的玩法

Router 按会话 id 保存每个会话正在玩的游戏, 只放在服务端内存里; 界面 (app/chat.py, 命令行) 只需要把会话 id 和用户消息交进来.
停止: handle 可以带一个 threading.Event, 它被 set 后, 这一问不再发新的模型请求 (抛 Cancelled), 游戏状态退回问之前.
"""
import argparse
import json
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import yaml

from rag import config

SKILLS_DIR = config.ROOT / "skills"
_TAIL = re.compile(r"[\s。.!！?？~～]+$")
COMMANDS = {
    "quit": ("/退出", "/quit", "/exit"),
    "reveal": ("/放弃", "/揭晓", "/giveup"),
    "hint": ("/提示", "/hint"),
    "help": ("/帮助", "/help"),
}
ANSWER = re.compile(r"^/(?:答|answer)\s*(.*)$", re.S | re.I)
REQUEST_TIMEOUT = 60  # 游戏里单次模型请求的超时 (秒); 问答 agent 用 rag/openrouter 的默认 300 秒
TRACE_DIR = config.DATA_DIR / "processed" / "game_logs"  # 游戏和网页问答的逐步记录, 按天一个 jsonl (不进 git)
_ctx = threading.local()  # 当前线程正在处理的那一问: sid, skill, turn, cancel; 模型请求的记录靠它知道属于哪一问
_trace_lock = threading.Lock()
OnEvent = Callable[[str, Any], None]  # on_event(kind, info): 游戏一问的进度事件, 见 rag/guess.play


class SkillReply(Exception):
    """start_game / end_game 的 handler 抛出: 这一问的回复就是 text (开局说明或揭晓), 问答 agent 不再继续."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.text = text


class Cancelled(Exception):
    """用户点了停止或发了新消息: 这一问不再发新的模型请求."""


def set_context(sid: str | None, skill: str | None, turn: int | None, cancel: threading.Event | None = None) -> None:
    """记下当前线程在处理哪一问 (trace 的 sid / skill / turn, 模型请求前查的 cancel). 问答 agent 的线程也用它."""
    _ctx.sid, _ctx.skill, _ctx.turn, _ctx.cancel = sid, skill, turn, cancel


def check_cancel() -> None:
    """当前这一问被取消了就抛 Cancelled. 每次模型请求前调用."""
    c = getattr(_ctx, "cancel", None)
    if c is not None and c.is_set():
        raise Cancelled()


def trace(record: dict) -> None:
    """往 TRACE_DIR/<日期>.jsonl 追加一行记录, 自动带上时间和当前这一问的 sid / skill / turn (见 Router.handle).

    记录类型 (type):
        user     玩家这一句: {"text"} (skill 为 "qa" 的是问答 agent 的一问, 记录格式相同)
        request  一次模型请求: {"seconds", "n_messages", "tools", "last" (最后一条消息, 截断), "content", "reasoning",
                 "tool_calls": [{"name", "arguments"}], "usage": {"prompt", "completion", "reasoning"}, "error"}
        tool     一次工具调用: {"tool", "args", "chunks", "error"}
        reply    这一问的回复: {"text", "seconds"}
    写失败 (磁盘等) 时静默跳过, 不影响游戏."""
    rec = {"ts": round(time.time(), 3), **{k: getattr(_ctx, k, None) for k in ("sid", "skill", "turn")}, **record}
    try:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        with _trace_lock, open(TRACE_DIR / f"{time.strftime('%Y-%m-%d')}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass


class GameLike(Protocol):
    """一局游戏要提供的接口 (rag/guess.GuessPerson, rag/soup.Game 都符合)."""

    @property
    def over(self) -> bool: ...

    def intro(self, on_event: OnEvent | None = None) -> str: ...
    def ask(self, text: str, /, on_event: OnEvent | None = None) -> str | None: ...  # None: 这句话不是游戏里的话
    def reveal(self) -> str: ...


@dataclass
class Skill:
    """skills/<name>/SKILL.md 读出来的一个 skill."""
    name: str
    title: str
    description: str
    triggers: list[str]
    body: str
    modes: dict[str, str]


def load_skills(root: Path = SKILLS_DIR) -> dict[str, Skill]:
    """读 root 下每个子目录的 SKILL.md, 返回 {name: Skill}. frontmatter 是两行 "---" 之间的 YAML."""
    out = {}
    for path in sorted(root.glob("*/SKILL.md")):
        text = path.read_text(encoding="utf-8")
        m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
        if not m:
            raise ValueError(f"{path}: 开头没有 --- 包起来的 frontmatter")
        meta = yaml.safe_load(m.group(1)) or {}
        out[meta["name"]] = Skill(meta["name"], meta.get("title", meta["name"]), meta.get("description", ""),
                                  [str(t) for t in meta.get("triggers", [])], m.group(2).strip(),
                                  {str(k): str(v) for k, v in (meta.get("modes") or {}).items()})
    return out


def _norm(text: str) -> str:
    return _TAIL.sub("", text.strip()).casefold()


def match_skill(text: str, skills: dict[str, Skill]) -> Skill | None:
    """整句是某个 skill 的快捷命令就返回它 (规则见模块 docstring), 否则 None."""
    t = _norm(text)
    return next((s for s in skills.values() if any(t == _norm(x) for x in s.triggers)), None)


def _command(text: str) -> str | None:
    t = _norm(text)
    return next((k for k, forms in COMMANDS.items() if t in forms), None)


def skill_prompt(skills: dict[str, Skill]) -> str:
    """问答 agent 的 system prompt 末尾附上的 skill 列表."""
    lines = []
    for sk in skills.values():
        modes = "; ".join(f"{k}: {v}" for k, v in sk.modes.items())
        lines.append(f"- {sk.name} ({sk.title}): {sk.description}" + (f" mode 可选 {modes}." if modes else ""))
    return ("\n\n除了回答问题, 你还能和对方玩下面的游戏 (skill):\n" + "\n".join(lines) +
            "\n对方想玩时调用 start_game 开局 (谜底由程序出, 你不要自己出题, 也不要自己当裁判); 不确定对方想不想玩时先问一句.")


class Router:
    """按会话 id 管理正在进行的游戏. factories: {skill 名: 无参函数, 返回新的一局游戏 (GameLike)}.

    只有 factories 里有的 skill 才可用; SKILL.md 存在但没有对应工厂的, 不出现在 skill 列表和 start_game 里."""

    def __init__(self, factories: dict[str, Callable[[], GameLike]], skills: dict[str, Skill] | None = None) -> None:
        all_skills = skills if skills is not None else load_skills()
        self.skills = {k: v for k, v in all_skills.items() if k in factories}
        self.factories = factories
        self.sessions: dict[str, tuple[Skill, GameLike]] = {}
        self.turns: dict[str, int] = {}  # 每个会话说了几句 (记录里的 turn)
        self.lock = threading.Lock()

    def active(self, sid: str) -> Skill | None:
        """这个会话正在玩的 skill, 没有在玩时 None."""
        s = self.sessions.get(sid)
        return s[0] if s else None

    def claims(self, sid: str, text: str) -> bool:
        """这句话先交给游戏 (正在玩, 或是快捷命令). 界面用它决定先走游戏还是直接走问答 agent, 不改状态."""
        return match_skill(text, self.skills) is not None or self.active(sid) is not None

    # ---------- 问答 agent 用的工具 ----------

    def tools(self) -> list[dict]:
        """start_game / end_game 的 schema, 加在问答 agent 的 TOOLS 后面."""
        names = list(self.skills)
        modes = sorted({m for sk in self.skills.values() for m in sk.modes})
        start: dict = {"type": "object", "properties": {"game": {"type": "string", "enum": names, "description": "游戏名"}},
                       "required": ["game"]}
        if modes:
            start["properties"]["mode"] = {"type": "string", "enum": modes,
                                           "description": "玩法 (只有部分游戏有, 见 system prompt 的 skill 列表); 对方没说清楚时不填, 开局时会问"}
        return [
            {"type": "function", "function": {"name": "start_game", "description": "开一局游戏 (skill). 开局说明由程序直接发给对方.",
                                              "parameters": start}},
            {"type": "function", "function": {
                "name": "end_game", "description": "结束正在玩的一局 (对方说不玩了). reveal 为 true 时同时揭晓答案.",
                "parameters": {"type": "object", "properties": {"reveal": {"type": "boolean"}}, "required": []}}},
        ]

    def handlers(self, sid: str, on_event: OnEvent | None = None) -> dict[str, Callable[[dict], str]]:
        """{工具名: handler}, 交给 rag/agent.run_agent. 执行成功时抛 SkillReply (回复交给界面); 参数不对时返回错误文字给模型."""
        def start_game(args: dict) -> str:
            name, mode = args.get("game"), args.get("mode")
            if name not in self.skills:
                return f"没有这个游戏: {name}. 可选: {', '.join(self.skills)}."
            raise SkillReply(self.start(sid, name, mode if mode in self.skills[name].modes else None, on_event))

        def end_game(args: dict) -> str:
            if self.active(sid) is None:
                return "现在没有在玩游戏."
            raise SkillReply(self.end(sid, bool(args.get("reveal"))))
        return {"start_game": start_game, "end_game": end_game}

    def agent_note(self, sid: str) -> str:
        """交给问答 agent 时附在 system prompt 末尾的文字: skill 列表; 正在玩时加上游戏名和玩法."""
        note = skill_prompt(self.skills)
        sk = self.active(sid)
        if sk:
            note += (f"\n\n对方正在玩 {sk.title}, 玩法:\n{sk.body}\n"
                     "这句话游戏判断不是游戏里的问题, 交给你回答. 你不知道谜底, 不要猜谜底, 也不要替裁判回答游戏里的问题;"
                     " 对方想结束就调用 end_game, 想换游戏就调用 start_game; 回答完可以提醒对方游戏还在进行.")
        return note

    # ---------- 开局, 结束, 处理一句话 ----------

    def start(self, sid: str, name: str, mode: str | None = None, on_event: OnEvent | None = None) -> str:
        """给会话 sid 开一局 name (正在玩的那局直接丢掉), 返回开局说明. mode 是 SKILL.md 的 modes 之一时直接按这个玩法开局."""
        sk = self.skills[name]
        game = self.factories[name]()
        choose = getattr(game, "choose", None)
        intro = choose(mode, on_event) if mode and choose else game.intro(on_event)
        with self.lock:
            self.sessions[sid] = (sk, game)
        return f"【{sk.title}】{intro}"

    def end(self, sid: str, reveal: bool = False) -> str:
        """结束会话 sid 正在玩的一局; reveal 时返回揭晓的文字."""
        with self.lock:
            cur = self.sessions.pop(sid, None)
        if not cur:
            return "现在没有在玩游戏."
        sk, game = cur
        return game.reveal() if reveal else f"已退出{sk.title}. 可以继续问三国历史问题."

    def handle(self, sid: str, text: str, on_event: OnEvent | None = None,
               cancel: threading.Event | None = None) -> str | None:
        """处理会话 sid 的一句话, 返回要显示的文字; None 表示不归游戏管 (没在玩, 或游戏判断这句话不是游戏里的话).
        on_event(kind, info) 收游戏这一问的进度 (Agent 的推理, 工具调用), 见 rag/guess.play.
        cancel 被 set 后不再发新的模型请求, 抛 Cancelled (游戏状态退回这一问之前)."""
        sk0 = match_skill(text, self.skills) or self.active(sid)
        if sk0 is None:
            return None
        self.turns[sid] = self.turns.get(sid, 0) + 1
        set_context(sid, sk0.name, self.turns[sid], cancel)
        trace({"type": "user", "text": text})
        start = time.time()

        def logged(kind: str, info: Any) -> None:
            if kind == "tool" and isinstance(info, dict):
                trace({"type": "tool", **{k: info.get(k) for k in ("tool", "args", "chunks", "error")}})
            if on_event:
                on_event(kind, info)
        reply = self._handle(sid, text, logged)
        trace({"type": "reply", "text": reply, "seconds": round(time.time() - start, 2)})
        return reply

    def _handle(self, sid: str, text: str, on_event: OnEvent | None) -> str | None:
        sk = match_skill(text, self.skills)
        if sk:
            return self.start(sid, sk.name, None, on_event)
        with self.lock:
            cur = self.sessions.get(sid)
        if not cur:
            return None
        sk, game = cur
        cmd = _command(text)
        if cmd == "quit":
            return self.end(sid)
        if cmd == "reveal":
            return self.end(sid, reveal=True)
        if cmd == "help":
            return sk.body
        if cmd == "hint":
            hint = getattr(game, "hint", None)
            return hint() if hint else "这个游戏没有提示."
        m = ANSWER.match(text.strip())
        if m:
            submit = getattr(game, "guess", None)
            reply = submit(m.group(1).strip()) if submit else game.ask("猜 " + m.group(1).strip(), on_event)
        else:
            reply = game.ask(text, on_event)
        if game.over:
            with self.lock:
                self.sessions.pop(sid, None)
        return reply

    def reset(self, sid: str) -> None:
        """结束这个会话正在玩的游戏 (界面上 "清除对话" 时调用), 不揭晓."""
        with self.lock:
            self.sessions.pop(sid, None)


def logged_request(complete: Callable[..., Any], messages: list[dict], **kw: Any) -> Any:
    """发一次请求 complete(messages, **kw) (返回 ChatCompletion), 把耗时, token, 推理, 工具调用记进 trace, 失败也记一行再抛.
    请求前先 check_cancel: 这一问被取消 (停止 / 新消息) 时抛 Cancelled, 不再花钱. 游戏和网页问答都经过这里."""
    check_cancel()
    label = getattr(_ctx, "skill", None) or "?"
    t = time.time()
    base = {"n_messages": len(messages), "tools": bool(kw.get("tools")),
            "last": str(messages[-1].get("content"))[:300] if messages else None}
    try:
        resp = complete(messages, **kw)
    except Exception as e:
        trace({"type": "request", "seconds": round(time.time() - t, 2), **base, "error": f"{type(e).__name__}: {e}"})
        print(f"[{label}] 模型请求失败 {time.time() - t:.1f}s: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        raise
    try:
        msg, u = resp.choices[0].message, getattr(resp, "usage", None)
        details = getattr(u, "completion_tokens_details", None)
        calls = [{"name": c.function.name, "arguments": c.function.arguments} for c in getattr(msg, "tool_calls", None) or []]
        trace({"type": "request", "seconds": round(time.time() - t, 2), **base, "content": msg.content,
               "reasoning": getattr(msg, "reasoning", None), "tool_calls": calls,
               "usage": {"prompt": getattr(u, "prompt_tokens", None), "completion": getattr(u, "completion_tokens", None),
                         "reasoning": getattr(details, "reasoning_tokens", None)}, "error": None})
        print(f"[{label}] 模型请求 {time.time() - t:.1f}s, 消息 {len(messages)} 条, 工具调用 {len(calls)} 个",
              file=sys.stderr, flush=True)
    except Exception:  # 返回的形状不对 (假模型等): 不记, 照样返回
        pass
    return resp


def make_router(lib: Any, complete: Callable[..., Any], model: str | None = None, effort: str | None = None,
                rng: random.Random | None = None) -> Router:
    """建好两个游戏的 Router. complete 签名同 rag.llm.complete (complete(messages, model=..., **kw) -> ChatCompletion), 每次请求经 logged_request;
    猜人物的裁判要走 agent 循环 (要的是返回 message 的 complete), 这里包一层. lib 是 rag.library.load_library() 的结果."""
    from rag import guess, soup

    def traced(messages: list[dict], **kw: Any) -> Any:
        return logged_request(complete, messages, **kw)

    def message_complete(messages: list[dict], tools: list[dict] | None = None) -> Any:
        kw: dict[str, Any] = {"model": model, "timeout": REQUEST_TIMEOUT}
        if effort:
            kw["reasoning_effort"] = effort
        if tools:
            kw["tools"] = tools
        return traced(messages, **kw).choices[0].message

    puzzles = soup.load_puzzles()
    r = rng or random.Random()
    # 猜人物随机出题要先统计各人在全书出现的块数 (需要几秒), 启动时在后台先算好, 开局时不用等
    threading.Thread(target=lambda: guess.random_person(lib, random.Random()), daemon=True).start()
    return Router({
        "guess_person": lambda: guess.GuessPerson(lambda: guess.new_game(lib, message_complete, rng=r),
                                                  lambda: guess.AgentGuessGame(lib, message_complete)),
        "turtle_soup": lambda: soup.Game(r.choice(puzzles), traced, model, effort),
    })


def main() -> None:
    """命令行里玩. 没在游戏里时, 提示怎么开局 (命令行不接问答 agent, 问答用 rag.agent 或网页)."""
    ap = argparse.ArgumentParser(description="三国志 Agent 的游戏 skill (命令行)")
    ap.add_argument("--model", help="不给就用 .env 的 LLM_MODEL")
    ap.add_argument("--effort", help="reasoning_effort, 覆盖 .env 的 LLM_REASONING_EFFORT")
    ap.add_argument("--openrouter", action="store_true", help="走 OpenRouter (key 见 rag/openrouter.use_openrouter)")
    args = ap.parse_args()
    from rag import llm
    from rag.library import load_library

    model, complete = args.model or config.LLM_MODEL, llm.complete
    if args.openrouter:
        from rag.openrouter import PER_MODEL, full_name, make_complete, use_openrouter
        use_openrouter()
        model = full_name(model)
        sem = threading.Semaphore(PER_MODEL)

        def complete(*a: Any, **kw: Any) -> Any:  # 每次请求新包一层: make_complete 按创建时刻计时
            return make_complete(model, True, sem)(*a, **kw)
    router = make_router(load_library(), complete, model, args.effort or config.LLM_REASONING_EFFORT or None)
    names = " / ".join(t for s in router.skills.values() for t in s.triggers[:1])
    print(f"输入 {names} 开局; Ctrl-D 退出.")
    while True:
        try:
            text = input("> ")
        except EOFError:
            break
        reply = router.handle("cli", text)
        print(reply if reply is not None else f"现在没有在玩游戏. 输入 {names} 开局.")


if __name__ == "__main__":
    main()
