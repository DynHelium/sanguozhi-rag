"""rag/skills.py (触发和会话路由) 与 rag/guess.py (猜人物) 的测试: 假模型, 假资料, 不连网.

    uv run python -m tests.test_skills
"""
import json
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace

from rag import guess, skills, soup
from tests.test_agent import LIB

skills.TRACE_DIR = Path(tempfile.mkdtemp())  # 测试的逐步记录写到临时目录, 不进 data/

PERSON: guess.Person = {"id": "g-01", "person": "刘备", "aliases": ["先主", "刘玄德"], "bio": ["32-0000"],
                        "note": "蜀汉昭烈皇帝"}
PUZZLE = {"id": "soup-x", "tier": "common", "person": "曹冲", "title": "t", "surface": "一个孩子把衣服戳破了.",
          "truth": "为救库吏.", "facts": [{"id": "f1", "text": "救库吏", "evidence": [0]}],
          "evidence": [{"source_id": "20-0003", "quote": "庫吏懼必死"}], "hints": ["和老鼠有关"], "max_questions": 20}


def judge_says(*verdicts: str):
    """假的 message 版 complete: 依次返回给定判词的 JSON, 不调工具; calls 记录被调了几次."""
    queue = list(verdicts)

    def complete(messages, tools=None):
        complete.calls += 1
        return SimpleNamespace(content=f'{{"verdict": "{queue.pop(0)}"}}', tool_calls=None)
    complete.calls = 0
    return complete


def chat_says(*texts: str):
    """假的 ChatCompletion 版 complete (rag.llm.complete 的形状), 海龟汤裁判用."""
    queue = list(texts)

    def complete(messages, **kw):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=queue.pop(0)))])
    return complete


def test_load_and_match_triggers():
    """仓库里的两个 SKILL.md 能读; 快捷命令整句匹配, 不分大小写, 句末标点不论; 普通的话 (包括 "猜人物") 交给问答 agent 判断."""
    sk = skills.load_skills()
    assert {"guess_person", "turtle_soup"} <= set(sk) and set(sk["guess_person"].modes) == {"player_guesses", "agent_guesses"}
    for text in (" /猜人物！", "/guess_person", "/GUESS_PERSON"):
        assert skills.match_skill(text, sk).name == "guess_person", text
    for text in ("/海龟汤", "/Turtle_soup", "/海龟汤?"):
        assert skills.match_skill(text, sk).name == "turtle_soup", text
    for text in ("猜人物", "海龟汤", "猜人物怎么玩", "曹操因为什么逃离洛阳？", "猜"):
        assert skills.match_skill(text, sk) is None, text


def test_parse_guess():
    """明确的猜名字写法和只说一个已知人名算猜名字; 普通是非问题不算."""
    known = {"刘备", "曹操"}
    assert guess.parse_guess("猜 刘备", known) == "刘备"
    assert guess.parse_guess("/猜 先主", known) == "先主"
    assert guess.parse_guess("我猜是曹操吧", known) == "曹操"
    assert guess.parse_guess("刘备？", known) == "刘备"
    assert guess.parse_guess("是曹操吗", known) == "曹操"
    assert guess.parse_guess("他是男的吗？", known) is None
    assert guess.parse_guess("他和刘备是兄弟吗", known) is None


def test_guess_game_flow():
    """问一题 (模型判词), 没法回答的问题不计数, 猜错计一问, 用称呼猜中结束."""
    c = judge_says("是", "没法回答")
    g = guess.GuessGame(PERSON, LIB, c, known_names={"刘备", "曹操"})
    assert g.ask("他是男性吗？").startswith("是  (1/20)")
    assert g.ask("讲讲他的生平").startswith("没法回答  (1/20)") and c.calls == 2
    assert g.ask("猜 曹操").startswith("不是 曹操. (2/20)")
    out = g.ask("猜 刘玄德")
    assert out.startswith("猜对了") and "刘备" in out and "蜀汉昭烈皇帝" in out and g.over


def test_guess_bad_verdict_and_limit():
    """判词不认识且没有 answer 时算 "没有记载"; 问题用完后不再请求模型, 只提示猜名字."""
    c = judge_says("大概是吧")
    g = guess.GuessGame(PERSON, LIB, c, max_questions=1)
    assert g.ask("他打过仗吗").startswith("没有记载  (1/1)")
    assert "用完了" in g.ask("他当过皇帝吗") and c.calls == 1


def test_router_guess_session():
    """触发词开局, 之后的消息归游戏; /放弃 揭晓并结束; 结束后的消息交回问答 agent (返回 None)."""
    r = skills.Router({"guess_person": lambda: guess.GuessGame(PERSON, LIB, judge_says("不是"))})
    assert r.handle("s1", "曹操是怎么死的") is None
    assert r.handle("s1", "/猜人物").startswith("【猜人物】")
    assert r.handle("s2", "他是男性吗") is None, "别的会话不受影响"
    assert r.handle("s1", "他是魏国人吗").startswith("不是")
    assert "玩法" in r.handle("s1", "/帮助")
    assert r.handle("s1", "/放弃").startswith("谜底是 刘备")
    assert r.active("s1") is None and r.handle("s1", "他是魏国人吗") is None


def test_router_soup_session():
    """海龟汤: 是非问题, /提示, /答 提交解释 (说到全部要点即猜中并结束); /退出 结束不揭晓."""
    c = chat_says('{"verdict": "是", "evidence": [0]}', '{"covered": ["f1"], "contradiction": false}')
    r = skills.Router({"turtle_soup": lambda: soup.Game(PUZZLE, c)})
    assert "汤面" in r.handle("a", "/Turtle_soup")
    assert r.handle("a", "和库房有关吗").startswith("是")
    assert r.handle("a", "/提示").startswith("提示 1")
    assert r.handle("a", "/答 他是为了救库吏").startswith("猜中了") and r.active("a") is None
    r.handle("b", "/海龟汤")
    assert r.handle("b", "/退出").startswith("已退出") and r.active("b") is None


def test_router_restart_and_unknown_skill():
    """游戏中再发快捷命令就换一局; 没有游戏工厂的 skill 不触发, 也不在 start_game 的选项里."""
    made = []
    r = skills.Router({"guess_person": lambda: made.append(1) or guess.GuessGame(PERSON, LIB, judge_says())})
    r.handle("s", "/猜人物")
    r.handle("s", "/猜人物")
    assert len(made) == 2
    assert r.handle("x", "/海龟汤") is None, "turtle_soup 没有工厂, 交回问答 agent"
    assert r.tools()[0]["function"]["parameters"]["properties"]["game"]["enum"] == ["guess_person"]


def test_new_game_draws_from_pool():
    """new_game 从给定的谜底池抽人 (随机数可复现)."""
    g = guess.new_game(LIB, judge_says(), rng=random.Random(0), pool=[PERSON])
    assert g.p["person"] == "刘备" and "刘备" in g.known_names


def agent_says(*replies: str):
    """假的 message 版 complete: 依次返回给定的 JSON 文字 (Agent 来猜时的问题或猜测), 不调工具."""
    queue = list(replies)

    def complete(messages, tools=None):
        complete.last = messages
        return SimpleNamespace(content=queue.pop(0), tool_calls=None)
    return complete


def test_agent_guess_flow():
    """Agent 来猜: 开局给第一问; 太长的回答交给问答 agent (None); 猜错接着问; 猜对结束. 历史里没有 developer 提醒."""
    c = agent_says('{"ask": "这个人是男性吗？"}', '{"guess": "曹操", "desc": "魏武帝"}', '{"guess": "刘备", "desc": "先主"}')
    g = guess.AgentGuessGame(LIB, c)
    assert g.intro().endswith("第 1 问: 这个人是男性吗？")
    assert g.ask("x" * 41) is None
    assert g.ask("是的").startswith("第 2 次: 我猜是 曹操 (魏武帝)")
    assert g.ask("嗯哼").startswith("请回答 对 / 错")
    assert g.ask("错").startswith("第 3 次: 我猜是 刘备")
    assert g.ask("对").startswith("太好了") and g.over
    assert all(m["role"] != "developer" for m in c.last)


def test_agent_guess_runs_out():
    """次数用完就结束, 问玩家想的是谁; /放弃 让 Agent 认输."""
    c = agent_says('{"ask": "q1"}', '{"ask": "q2"}')
    g = guess.AgentGuessGame(LIB, c, max_turns=1)
    g.intro()
    assert "用完了" in g.ask("不是") and g.over
    g2 = guess.AgentGuessGame(LIB, agent_says('{"ask": "q"}'))
    g2.intro()
    assert "认输" in g2.reveal() and g2.over


def test_guess_person_mode_choice():
    """猜人物先选玩法: 1 = 玩家猜, 2 = Agent 猜; 别的话交给问答 agent (None)."""
    r = skills.Router({"guess_person": lambda: guess.GuessPerson(
        lambda: guess.GuessGame(PERSON, LIB, judge_says("是")), lambda: guess.AgentGuessGame(LIB, agent_says('{"ask": "q?"}')))})
    assert "两种玩法" in r.handle("u", "/猜人物")
    assert r.handle("u", "这游戏怎么玩") is None and r.active("u") is not None
    assert "你可以问" in r.handle("u", "1")
    assert r.handle("u", "他是男的吗").startswith("是")
    r.handle("v", "/guess_person")
    assert r.handle("v", "2").endswith("第 1 问: q?")


def test_play_reports_thinking_and_tools():
    """Agent 来猜: 推理文字和工具调用 (带检索词) 作为进度事件发出; 裁判只发 "查原文", 不发推理和检索词."""
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="search", arguments='{"query": "浚儀"}'))
    replies = [SimpleNamespace(content="", tool_calls=[call], reasoning="先查一下"),
               SimpleNamespace(content='{"ask": "他是魏国人吗？"}', tool_calls=None, reasoning="再问阵营")]
    events = []
    g = guess.AgentGuessGame(LIB, lambda m, tools=None: replies.pop(0))
    assert g.intro(on_event=lambda k, i: events.append((k, i))).endswith("第 1 问: 他是魏国人吗？")
    assert [k for k, _ in events] == ["think", "tool", "think"] and events[1][1]["args"] == {"query": "浚儀"}
    judge = [SimpleNamespace(content="", tool_calls=[call], reasoning="谜底是刘备"),
             SimpleNamespace(content='{"verdict": "是"}', tool_calls=None, reasoning="刘备是男的")]
    seen = []
    r = guess.judge_question(LIB, PERSON, "他是男的吗", lambda m, tools=None: judge.pop(0), lambda k, i: seen.append((k, i)))
    assert r["verdict"] == "是" and seen == [("stage", "查原文")]


def test_agent_guess_timeout_recovers():
    """想下一问超时: 对话历史恢复原样, 玩家下一句话不当回答, 直接重试."""
    def slow(m, tools=None):
        raise TimeoutError("慢")
    g = guess.AgentGuessGame(LIB, agent_says('{"ask": "q1"}', '{"ask": "q2"}'))
    g.intro()
    g.complete, before = slow, list(g.msgs)
    assert "想得太久" in g.ask("是") and g.pending and g.msgs[:len(before)] == before
    g.complete = agent_says('{"ask": "q2"}')
    assert g.ask("继续").endswith("第 2 问: q2") and not g.pending


def test_router_writes_trace():
    """每句话记 user / tool / reply 三类记录, 带 sid, skill, turn; 不归游戏管的话不记."""
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="search", arguments='{"query": "浚儀"}'))
    replies = [SimpleNamespace(content="", tool_calls=[call]), SimpleNamespace(content='{"ask": "q?"}', tool_calls=None)]
    for f in skills.TRACE_DIR.glob("*.jsonl"):
        f.unlink()
    r = skills.Router({"guess_person": lambda: guess.AgentGuessGame(LIB, lambda m, tools=None: replies.pop(0))})
    assert r.handle("t", "曹操是谁") is None
    r.handle("t", "/猜人物")
    rows = [json.loads(line) for f in skills.TRACE_DIR.glob("*.jsonl") for line in open(f, encoding="utf-8")]
    assert [x["type"] for x in rows] == ["user", "tool", "reply"]
    assert {(x["sid"], x["skill"], x["turn"]) for x in rows} == {("t", "guess_person", 1)}
    assert rows[1]["args"] == {"query": "浚儀"} and rows[2]["text"].endswith("第 1 问: q?")


def test_agent_guess_free_answer():
    """认不出的短回答 (如 "都待过") 原话交给 Agent; "属于" 归成 "是"."""
    c = agent_says('{"ask": "他属于蜀汉吗？"}', '{"ask": "他降过魏吗？"}', '{"ask": "q3"}')
    g = guess.AgentGuessGame(LIB, c)
    g.intro()
    g.ask("都待过, 先蜀后魏")
    assert g.msgs[-2] == {"role": "user", "content": "都待过, 先蜀后魏"}
    g.ask("属于")
    assert g.msgs[-2] == {"role": "user", "content": "是"}


def test_people_in_finds_names_and_titles():
    """原文里出现姓名、原文称呼 (如 "太祖") 或 "姓 + 字" 都算; 单字称呼不算."""
    people = [{"id": "01-0000", "name": "曹操", "surname": "曹", "zi": "孟德", "forms": [{"form": "魏太祖", "orig": "太祖"}]},
              {"id": "47-0000", "name": "孙权", "surname": "孙", "zi": "仲谋", "forms": [{"form": "孙权", "orig": "權"}]},
              {"id": "32-0000", "name": "刘备", "surname": "刘", "zi": "玄德", "forms": []}]
    assert [p["name"] for p in guess.people_in("太祖軍譙，權遣使", people)] == ["曹操"]
    assert [p["name"] for p in guess.people_in("劉玄德在新野", people)] == ["刘备"]


def test_shared_titles_and_offices_not_used():
    """多人共用的称呼 (如两人都封过 "都亭侯") 和官称不用来认人, 否则一提 "都亭侯" 就算两人都出现."""
    people = [{"id": "a", "name": "苏则", "forms": [{"orig": "都亭侯", "kind": "爵号"}]},
              {"id": "b", "name": "任峻", "forms": [{"orig": "都亭侯", "kind": "爵号"}, {"orig": "大將軍", "kind": "官称"}]}]
    assert guess.people_in("封都亭侯，拜大將軍", people) == []
    assert [p["name"] for p in guess.people_in("任峻為都亭侯", people)] == ["任峻"]


def test_agent_tools_start_and_end_game():
    """问答 agent 的 start_game (带 mode 时直接按玩法开局) / end_game: handler 抛 SkillReply, 回复交给界面; 参数不对时返回错误文字."""
    r = skills.Router({"guess_person": lambda: guess.GuessPerson(
        lambda: guess.GuessGame(PERSON, LIB, judge_says()), lambda: guess.AgentGuessGame(LIB, agent_says('{"ask": "q?"}')))})
    h = r.handlers("a")
    assert "没有这个游戏" in h["start_game"]({"game": "chess"}) and "没有在玩" in h["end_game"]({})
    try:
        h["start_game"]({"game": "guess_person", "mode": "agent_guesses"})
        raise AssertionError("应该抛 SkillReply")
    except skills.SkillReply as e:
        assert e.text.startswith("【猜人物】") and e.text.endswith("第 1 问: q?")
    assert "正在玩 猜人物" in r.agent_note("a") and "guess_person" in r.agent_note("b")
    try:
        h["end_game"]({"reveal": True})
    except skills.SkillReply as e:
        assert "认输" in e.text and r.active("a") is None


def test_non_game_message_goes_to_agent():
    """裁判判 "非游戏" 时 ask 返回 None, 不计数, 也不留在裁判对话里."""
    g = guess.GuessGame(PERSON, LIB, judge_says("非游戏"))
    n = len(g.msgs)
    assert g.ask("这游戏怎么玩") is None and g.asked == 0 and len(g.msgs) == n


def test_cancel_stops_before_request():
    """取消标记 set 后, 游戏这一问不再请求模型 (抛 Cancelled), 裁判对话退回问之前."""
    import threading
    calls = []

    def complete(messages, **kw):
        calls.append(1)
        raise AssertionError("不该请求")
    r = skills.Router({"guess_person": lambda: guess.GuessGame(
        PERSON, LIB, lambda m, tools=None: skills.logged_request(complete, m).choices[0].message)})
    r.handle("c", "/猜人物")
    stop = threading.Event()
    stop.set()
    g = r.sessions["c"][1]
    n = len(g.msgs)
    try:
        r.handle("c", "他是男的吗", cancel=stop)
        raise AssertionError("应该抛 Cancelled")
    except skills.Cancelled:
        pass
    assert calls == [] and len(g.msgs) == n and g.asked == 0


def test_judge_free_answer_and_leak():
    """裁判可以回一句话 (二选一答选项); 回答里出现谜底名字或称呼就换成 LEAK_REPLY, 这一问不计数."""
    replies = [SimpleNamespace(content='{"verdict": "选择", "answer": "北方人"}', tool_calls=None),
               SimpleNamespace(content='{"verdict": "是", "answer": "是, 他就是先主"}', tool_calls=None)]
    g = guess.GuessGame(PERSON, LIB, lambda m, tools=None: replies.pop(0))
    assert g.ask("他是南方人还是北方人？").startswith("北方人  (1/20)")
    assert g.ask("他是不是开国皇帝").startswith(guess.LEAK_REPLY) and g.asked == 1


def test_judge_keeps_one_conversation():
    """一局的裁判是同一段对话: 第二问时模型能看到第一问的问题和回答, 以及玩家猜错的名字."""
    seen_msgs = []

    def complete(messages, tools=None):
        seen_msgs.append([m.get("content") for m in messages])
        return SimpleNamespace(content='{"verdict": "是", "answer": "是"}', tool_calls=None)
    g = guess.GuessGame(PERSON, LIB, complete, known_names={"曹操"})
    g.ask("他是皇帝吗")
    g.ask("曹操")
    g.ask("他是蜀国人吗")
    last = "\n".join(c or "" for c in seen_msgs[-1])
    assert "他是皇帝吗" in last and "曹操" in last and len(seen_msgs) == 2

if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
