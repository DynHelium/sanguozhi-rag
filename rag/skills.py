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
import random
import re
import threading
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


class GameLike(Protocol):
    """一局游戏要提供的接口 (rag/guess.GuessGame, rag/soup.Game 都符合)."""
    over: bool

    def intro(self) -> str: ...
    def ask(self, text: str, /) -> str: ...
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
        self.lock = threading.Lock()

    def active(self, sid: str) -> Skill | None:
        """这个会话正在玩的 skill, 没有在玩时 None."""
        s = self.sessions.get(sid)
        return s[0] if s else None

    def handle(self, sid: str, text: str) -> str | None:
        """处理会话 sid 的一句话, 返回要显示的文字; None 表示不归游戏管 (没在玩, 也不是触发词)."""
        sk = match_skill(text, self.skills)
        if sk:
            game = self.factories[sk.name]()
            with self.lock:
                self.sessions[sid] = (sk, game)
            return f"【{sk.title}】{game.intro()}"
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
            reply = submit(m.group(1).strip()) if submit else game.ask("猜 " + m.group(1).strip())
        else:
            reply = game.ask(text)
        if game.over:
            self._end(sid)
        return reply

    def _end(self, sid: str) -> None:
        with self.lock:
            self.sessions.pop(sid, None)


def make_router(lib: Any, complete: Callable[..., Any], model: str | None = None, effort: str | None = None,
                rng: random.Random | None = None) -> Router:
    """建好两个游戏的 Router. complete 签名同 rag.llm.complete (complete(messages, model=..., **kw) -> ChatCompletion);
    猜人物的裁判要走 agent 循环 (要的是返回 message 的 complete), 这里包一层. lib 是 rag.library.load_library() 的结果."""
    from rag import guess, soup

    def message_complete(messages: list[dict], tools: list[dict] | None = None) -> Any:
        kw: dict[str, Any] = {"model": model}
        if effort:
            kw["reasoning_effort"] = effort
        if tools:
            kw["tools"] = tools
        return complete(messages, **kw).choices[0].message

    puzzles = soup.load_puzzles()
    r = rng or random.Random()
    return Router({
        "guess_person": lambda: guess.new_game(lib, message_complete, rng=r),
        "turtle_soup": lambda: soup.Game(r.choice(puzzles), complete, model, effort),
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
