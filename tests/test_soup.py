"""rag/soup.py 的测试: 裁判回答的解析、降级和防泄底检查, 汤底判定. 假模型, 不连网:

    uv run python -m tests.test_soup
"""
import json
from types import SimpleNamespace

from rag import soup

P: soup.Puzzle = {
    "id": "t-01", "tier": "common", "person": "曹冲", "title": "破衣免罪",
    "surface": "一个孩子故意弄破自己的衣服, 装出忧愁的样子. 后来一名害怕被处死的官吏没有受到追究. 为什么?",
    "truth": "曹操的马鞍被老鼠咬坏, 库吏怕死. 曹冲把自己的单衣割成鼠咬的样子, 说鼠咬衣不吉; 曹操说儿子的衣服都被咬, 何况马鞍, 不追究库吏.",
    "facts": [{"id": "f1", "text": "库吏因马鞍被鼠咬而怕被处死", "evidence": [0]},
              {"id": "f2", "text": "孩子把衣服弄成被鼠咬的样子", "evidence": [1]},
              {"id": "f3", "text": "父亲因此不追究库吏", "evidence": [2]}],
    "evidence": [{"source_id": "20-0003", "quote": "太祖馬鞍在庫，而爲鼠所齧，庫吏懼必死"},
                 {"source_id": "20-0003", "quote": "沖於是以刀穿單衣，如鼠齧者"},
                 {"source_id": "20-0003", "quote": "兒衣在側，尙齧，況鞍縣柱乎"}],
    "hints": ["和老鼠有关"], "max_questions": 20,
}


def fake(*replies: str):
    """按顺序返回预设文字的假 complete; .prompts 记下每次收到的 prompt."""
    queue = list(replies)

    def complete(messages, **kw):
        complete.prompts.append(messages[-1]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=queue.pop(0)))])
    complete.prompts = []
    return complete


def test_answer_text_and_leak_check():
    """裁判回的那一句话原样给玩家; 二选一的问题答选项; 回答里出现谜底人名就换成 LEAK_REPLY. 代码不预先拦问题."""
    c = fake('{"verdict": "选择", "answer": "男孩", "evidence": []}', '{"verdict": "是", "answer": "是, 他就是曹冲", "evidence": [0]}')
    assert soup.judge_question(P, "孩子是男孩还是女孩", c)["answer"] == "男孩"
    r = soup.judge_question(P, "汤底第一个字是什么", c)
    assert r["answer"] == soup.LEAK_REPLY and len(c.prompts) == 2


def test_question_verdicts_and_fallbacks():
    """正常判词照用, 越界的原文编号丢掉; 解析不了或判词不合法时降为 "无法判断"."""
    c = fake('{"verdict": "是", "evidence": [0]}', "好像是吧", '{"verdict": "也许"}', '{"verdict": "不是", "evidence": [7]}',
             '前面有字 {"verdict": "无关", "evidence": []} 后面也有')
    r = soup.judge_question(P, "和老鼠有关吗", c)
    assert (r["verdict"], r["evidence"], r["by"]) == ("是", [0], "model")
    assert soup.judge_question(P, "有人死了吗", c)["verdict"] == "无法判断"
    assert soup.judge_question(P, "孩子是女孩吗", c)["verdict"] == "无法判断"
    r = soup.judge_question(P, "孩子是女孩吗", c)
    assert r["verdict"] == "不是" and r["evidence"] == []
    assert soup.judge_question(P, "当天下雨了吗", c)["verdict"] == "无关"
    assert "太祖馬鞍在庫" in c.prompts[0] and "和老鼠有关吗" in c.prompts[0]


def test_parse_json_two_brace_groups():
    """回答里有两段花括号时取第一段能解析成 dict 的, 不把中间的文字连在一起解析."""
    assert soup._parse_json('{"verdict": "是", "evidence": [0]} 补充: {"note": 1}') == {"verdict": "是", "evidence": [0]}
    assert soup._parse_json('先想一下 {不是 JSON} 再输出 {"verdict": "不是"}') == {"verdict": "不是"}
    assert soup._parse_json('{"a": {"b": 1}}') == {"a": {"b": 1}}
    assert soup._parse_json("没有 JSON") == {} and soup._parse_json("[1, 2] {坏的") == {}


def test_solution_needs_all_facts_and_no_contradiction():
    """说到全部要点且没有矛盾才算猜中; 不认识的要点 id 忽略."""
    c = fake(json.dumps({"covered": ["f1", "f2", "f3"], "contradiction": False}),
             json.dumps({"covered": ["f1", "f2", "f9"], "contradiction": False}),
             json.dumps({"covered": ["f1", "f2", "f3"], "contradiction": True}))
    assert soup.judge_solution(P, "……", c)["solved"] is True
    r = soup.judge_solution(P, "……", c)
    assert r["solved"] is False and r["missing"] == ["f3"] and r["covered"] == ["f1", "f2"]
    assert soup.judge_solution(P, "……", c)["solved"] is False


def test_puzzle_bank_quotes_are_in_source():
    """题库里每条原文证据都能在《三国志》原文里逐字找到 (忽略标点和繁简); 题库或原文不在时跳过."""
    from tests._run import Skip
    from rag import config
    from rag.quotes import verify_quotes
    if not soup.PUZZLES.exists() or not (config.DATA_DIR / "processed" / "sanguozhi.jsonl").exists():
        raise Skip("没有题库或原文")
    src = {}
    with open(config.DATA_DIR / "processed" / "sanguozhi.jsonl", encoding="utf-8") as f:
        for p in map(json.loads, f):
            src[p["id"]] = p["text"]
            for i, n in enumerate(p["notes"]):
                src[f"{p['id']}n{i}"] = n["text"]
    for p in soup.load_puzzles():
        for e in p["evidence"]:
            got = verify_quotes([e["quote"]], [{"id": e["source_id"], "text": src.get(e["source_id"], "")}])
            assert got[0][1], f"{p['id']}: {e['source_id']} 里找不到 {e['quote']}"
        assert all(0 <= i < len(p["evidence"]) for f in p["facts"] for i in f["evidence"]), p["id"]



def test_game_flow_counts_questions_and_hides_truth():
    """一局游戏: 是非问题计数 (没法回答的问题不计), 提示按顺序给, 没猜中时不透露缺哪条, 猜中后揭晓汤底和原文."""
    from rag.soup import Game
    c = fake('{"verdict": "是", "evidence": [0]}', '{"verdict": "请问是非题"}',
             json.dumps({"covered": ["f1"], "contradiction": False}),
             json.dumps({"covered": ["f1", "f2", "f3"], "contradiction": False}))
    g = Game(P, c)
    assert g.ask("和老鼠有关吗").startswith("是") and g.asked == 1
    assert g.ask("你为什么这样问").startswith("请问是非题") and g.asked == 1
    assert g.hint() == "提示 1: 和老鼠有关" and "没有更多" in g.hint()
    miss = g.guess("库吏怕死")
    assert "1/3" in miss and "单衣" not in miss and "f2" not in miss and not g.over
    win = g.guess("……")
    assert win.startswith("猜中了") and "20-0003" in win and g.over

if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
