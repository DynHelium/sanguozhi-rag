"""三国志 Agent 的 skill: 聊天里说出触发词就开一局游戏 (猜人物, 海龟汤), 之后的消息都交给这局游戏, 不走问答 agent.

    uv run python -m rag.skills --openrouter     # 命令行里玩: 输入 "猜人物" 或 "海龟汤" 开局
    uv run python -m tests.test_skills           # 假模型测试, 不连网

每个 skill 是 skills/<name>/SKILL.md: 开头的 YAML frontmatter 给名字和触发词, 正文是玩法说明 (/帮助 时显示).

    skills/guess_person/SKILL.md
        ---
        name: guess_person                                     和 make_router 里的游戏工厂对应
        title: 猜人物
        description: Agent 心里想一个《三国志》人物, 玩家问是非问题, 最后说出名字.
        triggers: ["猜人物", "/猜人物", "/guess_person"]
        ---
        玩法: ……

触发 (match_skill): 整句去掉首尾空白和句末标点后和某个触发词完全相同才算, 不分大小写 ("/Turtle_soup" 也行).
所以 "猜人物怎么玩" "海龟汤是什么" 这类普通提问不会误触发, 照常交给问答 agent.

游戏进行中 (Router.handle) 的命令, 也是整句匹配:
    /退出 /quit /exit          结束这一局
    /放弃 /揭晓 /giveup        揭晓答案并结束
    /提示 /hint                要一条提示 (游戏有提示时)
    /答 <解释> /answer <解释>   提交完整解释 (海龟汤); 猜人物里等同于猜名字
    /帮助 /help                再看一遍 SKILL.md 的玩法
其他消息都交给游戏的 ask(). 再说一次触发词就换一局 (同一个或另一个游戏).

Router 按会话 id 保存每个会话正在玩的游戏, 只放在服务端内存里; 界面 (app/chat.py, 命令行) 只需要把会话 id 和用户消息交进来,
拿回要显示的文字. handle 返回 None 表示这句话不归游戏管, 由调用方照常交给问答 agent.
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
TRACE_DIR = config.DATA_DIR / "processed" / "game_logs"  # 每局游戏的逐步记录, 按天一个 jsonl (不进 git)
_ctx = threading.local()  # 当前线程正在处理的那一问: sid, skill, turn; 模型请求的记录靠它知道属于哪一问
_trace_lock = threading.Lock()


def trace(record: dict) -> None:
    """往 TRACE_DIR/<日期>.jsonl 追加一行记录, 自动带上时间和当前这一问的 sid / skill / turn (见 Router.handle).

    记录类型 (type):
        user     玩家这一句: {"text"}
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
OnEvent = Callable[[str, Any], None]


class GameLike(Protocol):
    """一局游戏要提供的接口 (rag/guess.GuessPerson, rag/soup.Game 都符合)."""

    @property
    def over(self) -> bool: ...

    def intro(self, on_event: OnEvent | None = None) -> str: ...
    def ask(self, text: str, /, on_event: OnEvent | None = None) -> str: ...
    def reveal(self) -> str: ...


@dataclass
class Skill:
    """skills/<name>/SKILL.md 读出来的一个 skill."""
    name: str
    title: str
    description: str
    triggers: list[str]
    body: str


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
                                  [str(t) for t in meta.get("triggers", [])], m.group(2).strip())
    return out


def _norm(text: str) -> str:
    return _TAIL.sub("", text.strip()).casefold()


def match_skill(text: str, skills: dict[str, Skill]) -> Skill | None:
    """整句是某个 skill 的触发词就返回它 (规则见模块 docstring), 否则 None."""
    t = _norm(text)
    return next((s for s in skills.values() if any(t == _norm(x) for x in s.triggers)), None)


def _command(text: str) -> str | None:
    t = _norm(text)
    return next((k for k, forms in COMMANDS.items() if t in forms), None)


class Router:
    """按会话 id 管理正在进行的游戏. factories: {skill 名: 无参函数, 返回新的一局游戏 (GameLike)}.

    只有 factories 里有的 skill 才会被触发; SKILL.md 存在但没有对应工厂的, 触发词照常交给问答 agent."""

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
        """这句话归不归游戏管 (是触发词, 或这个会话正在玩). 界面用它决定走游戏还是问答 agent, 不改状态."""
        return match_skill(text, self.skills) is not None or self.active(sid) is not None

    def handle(self, sid: str, text: str, on_event: OnEvent | None = None) -> str | None:
        """处理会话 sid 的一句话, 返回要显示的文字; None 表示不归游戏管 (没在玩, 也不是触发词).
        on_event(kind, info) 收游戏这一问的进度 (Agent 的推理, 工具调用), 见 rag/guess.play."""
        cur0 = self.sessions.get(sid)
        sk0 = match_skill(text, self.skills) or (cur0[0] if cur0 else None)
        if sk0 is None:
            return None
        self.turns[sid] = self.turns.get(sid, 0) + 1
        _ctx.sid, _ctx.skill, _ctx.turn = sid, sk0.name, self.turns[sid]
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
            game = self.factories[sk.name]()
            with self.lock:
                self.sessions[sid] = (sk, game)
            return f"【{sk.title}】{game.intro(on_event)}"
        with self.lock:
            cur = self.sessions.get(sid)
        if not cur:
            return None
        sk, game = cur
        cmd = _command(text)
        if cmd == "quit":
            self._end(sid)
            return f"已退出{sk.title}. 可以继续问三国历史问题."
        if cmd == "reveal":
            self._end(sid)
            return game.reveal()
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
            self._end(sid)
        return reply

    def reset(self, sid: str) -> None:
        """结束这个会话正在玩的游戏 (界面上 "清除对话" 时调用), 不揭晓."""
        self._end(sid)

    def _end(self, sid: str) -> None:
        with self.lock:
            self.sessions.pop(sid, None)


def make_router(lib: Any, complete: Callable[..., Any], model: str | None = None, effort: str | None = None,
                rng: random.Random | None = None) -> Router:
    """建好两个游戏的 Router. complete 签名同 rag.llm.complete (complete(messages, model=..., **kw) -> ChatCompletion);
    猜人物的裁判要走 agent 循环 (要的是返回 message 的 complete), 这里包一层. lib 是 rag.library.load_library() 的结果."""
    from rag import guess, soup

    def traced(messages: list[dict], **kw: Any) -> Any:
        """发一次请求 (ChatCompletion), 把耗时, token, 推理, 工具调用记进 trace, 失败也记一行再抛."""
        t = time.time()
        base = {"n_messages": len(messages), "tools": bool(kw.get("tools")),
                "last": str(messages[-1].get("content"))[:300] if messages else None}
        try:
            resp = complete(messages, **kw)
        except Exception as e:
            trace({"type": "request", "seconds": round(time.time() - t, 2), **base, "error": f"{type(e).__name__}: {e}"})
            print(f"[skill] 模型请求失败 {time.time() - t:.1f}s: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            raise
        msg, u = resp.choices[0].message, getattr(resp, "usage", None)
        details = getattr(u, "completion_tokens_details", None)
        trace({"type": "request", "seconds": round(time.time() - t, 2), **base, "content": msg.content,
               "reasoning": getattr(msg, "reasoning", None),
               "tool_calls": [{"name": c.function.name, "arguments": c.function.arguments} for c in msg.tool_calls or []],
               "usage": {"prompt": getattr(u, "prompt_tokens", None), "completion": getattr(u, "completion_tokens", None),
                         "reasoning": getattr(details, "reasoning_tokens", None)}, "error": None})
        print(f"[skill] 模型请求 {time.time() - t:.1f}s, 消息 {len(messages)} 条, 工具调用 {len(msg.tool_calls or [])} 个",
              file=sys.stderr, flush=True)
        return resp

    def message_complete(messages: list[dict], tools: list[dict] | None = None) -> Any:
        kw: dict[str, Any] = {"model": model, "timeout": REQUEST_TIMEOUT}
        if effort:
            kw["reasoning_effort"] = effort
        if tools:
            kw["tools"] = tools
        return traced(messages, **kw).choices[0].message

    puzzles = soup.load_puzzles()
    r = rng or random.Random()
    # 猜人物随机出题要先统计各人在全书出现的块数 (约 8 秒), 启动时在后台先算好, 开局时不用等
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
    names = " / ".join(t for s in router.skills.values() for t in s.triggers)
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
