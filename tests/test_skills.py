"""rag/skills.py (触发和会话路由) 与 rag/guess.py (猜人物) 的测试: 假模型, 假资料, 不连网.

    uv run python -m tests.test_skills
"""
import random
from types import SimpleNamespace

from rag import guess, skills, soup
from tests.test_agent import LIB

PERSON: guess.Person = {"id": "g-01", "tier": "common", "person": "刘备", "aliases": ["先主", "刘玄德"],
                        "bio": ["32-0000"], "note": "蜀汉昭烈皇帝", "playable": True}
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
    """仓库里的两个 SKILL.md 能读; 触发词整句匹配, 不分大小写, 句末标点不论; 普通提问不触发."""
    sk = skills.load_skills()
    assert {"guess_person", "turtle_soup"} <= set(sk)
    for text in ("猜人物", " 猜人物！", "/猜人物", "/guess_person", "/GUESS_PERSON"):
        assert skills.match_skill(text, sk).name == "guess_person", text
    for text in ("海龟汤", "/海龟汤", "/Turtle_soup", "海龟汤?"):
        assert skills.match_skill(text, sk).name == "turtle_soup", text
    for text in ("猜人物怎么玩", "海龟汤是什么", "曹操因为什么逃离洛阳？", "猜"):
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
    """问一题 (模型判词), 打听名字被代码拦下不计数, 猜错计一问, 用称呼猜中结束."""
    c = judge_says("是")
    g = guess.GuessGame(PERSON, LIB, c, known_names={"刘备", "曹操"})
    assert g.ask("他是男性吗？").startswith("是  (1/20)")
    assert g.ask("他名字第一个字是什么？").startswith("请问是非题  (1/20)") and c.calls == 1
    assert g.ask("猜 曹操").startswith("不是 曹操. (2/20)")
    out = g.ask("猜 刘玄德")
    assert out.startswith("猜对了") and "刘备" in out and "蜀汉昭烈皇帝" in out and g.over


def test_guess_bad_verdict_and_limit():
    """判词不认识时判 "无法判断"; 问题用完后不再请求模型, 只提示猜名字."""
    c = judge_says("大概是吧")
    g = guess.GuessGame(PERSON, LIB, c, max_questions=1)
    assert g.ask("他打过仗吗").startswith("无法判断  (1/1)")
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
    r.handle("b", "海龟汤")
    assert r.handle("b", "/退出").startswith("已退出") and r.active("b") is None


def test_router_restart_and_unknown_skill():
    """游戏中再说触发词就换一局; 没有游戏工厂的 skill 不触发."""
    made = []
    r = skills.Router({"guess_person": lambda: made.append(1) or guess.GuessGame(PERSON, LIB, judge_says())})
    r.handle("s", "猜人物")
    r.handle("s", "猜人物")
    assert len(made) == 2
    assert r.handle("x", "海龟汤") is None, "turtle_soup 没有工厂, 交回问答 agent"


def test_new_game_draws_from_pool():
    """new_game 从给定的谜底池抽人 (随机数可复现)."""
    g = guess.new_game(LIB, judge_says(), rng=random.Random(0), pool=[PERSON])
    assert g.p["person"] == "刘备" and "刘备" in g.known_names


if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
