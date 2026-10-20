"""rag/agent.py 的测试: TOOLS / SYSTEM, run_tool, run_agent (代码介入, 自查, 引文修改), who.
循环本身见 tests/test_loop.py, 工具背后的数据查询见 tests/test_library.py.
不依赖 pytest, 不连网, 不加载 bge-m3:

    uv run python -m tests.test_agent

假模型: 一个按脚本依次返回 "调用工具" 或 "文字回答" 的 complete 函数, 记下每次收到的 messages 和参数.
"""
import copy
import json
from types import SimpleNamespace
from typing import Any, Callable

from rag import agent, library

CHUNKS = [
    {"id": "01-0001", "tier": 1, "juan": 1, "juan_title": "魏書•武帝紀", "headings": [], "parent": None, "cites": [],
     "sources": ["01-0001"], "text": "六年夏四月，揚兵河上。"},
    {"id": "01-0002", "tier": 1, "juan": 1, "juan_title": "魏書•武帝紀", "headings": [], "parent": None, "cites": [],
     "sources": ["01-0002"], "text": "七年春正月，公軍譙。遂至浚儀，治睢陽渠。"},
    {"id": "01-0002n0", "tier": 2, "juan": 1, "juan_title": "魏書•武帝紀", "headings": [], "parent": "01-0002",
     "cites": ["魏書"], "sources": ["01-0002n0"], "text": "《魏書》曰：太祖至浚儀。"},
    {"id": "01-0003", "tier": 1, "juan": 1, "juan_title": "魏書•武帝紀", "headings": [], "parent": None, "cites": [],
     "sources": ["01-0003"], "text": "進軍官渡。"},
    {"id": "02-0000", "tier": 1, "juan": 2, "juan_title": "魏書•文帝紀", "headings": [], "parent": None, "cites": [],
     "sources": ["02-0000"], "text": "文皇帝諱丕。"},
]
ERAS = {"建安": {"status": "ref", "start": 196, "by_regime": {"东汉": 196}},
        "建興": {"status": "多政权", "start": None, "by_regime": {"蜀汉": 223, "孙吴": 252}},
        "黃初": {"status": "ref", "start": 220, "by_regime": {"曹魏": 220}},
        "某某": {"status": "待核", "start": None, "by_regime": {}}}
TABLE = {"浚儀": ["01-0002"], "官渡": ["01-0003"]}


def fake_search(query: str, k: int, where=None) -> list[dict]:
    """假检索: 按 TABLE 查词返回块, 最多 k 个."""
    return [c for c in CHUNKS if c["id"] in TABLE.get(query, [])][:k]


LIB = agent.Library(CHUNKS, fake_search, ERAS)
agent.REVIEW = False  # 默认关自查, 让假模型脚本不必多备一次回答; 要测自查的用例显式传 review=True


def tc(cid: str, name: str, **args: Any) -> SimpleNamespace:
    """造一个 tool_call: tc.id, tc.function.name, tc.function.arguments (JSON 字符串)."""
    return SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)))


def resp(content: str = "", calls: list[SimpleNamespace] | None = None, tokens: int = 100) -> SimpleNamespace:
    """造一个 ChatCompletion 形状的响应: resp.choices[0].message 有 content 和 tool_calls, resp.usage.prompt_tokens."""
    msg = SimpleNamespace(content=content, tool_calls=calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=SimpleNamespace(prompt_tokens=tokens))


class Script:
    """假模型: 按顺序返回预先给定的响应; seen 记下每次请求的 messages (深拷贝) 和其他关键字参数 (如 tools)."""

    def __init__(self, *replies: SimpleNamespace) -> None:
        self.replies: list[SimpleNamespace] = list(replies)
        self.seen: list[dict] = []

    def __call__(self, messages: list[dict], model: str | None = None, **kw: Any) -> SimpleNamespace:
        self.seen.append({"messages": copy.deepcopy(messages), "kw": kw})
        return self.replies.pop(0)


# ---------- TOOLS / SYSTEM ----------

def test_tools_declared():
    """TOOLS 声明了 search / read / era / who, 每项符合 OpenAI tools 格式, required 的参数都在 properties 里."""
    names = {t["function"]["name"] for t in agent.TOOLS}
    assert {"search", "read", "era", "who"} <= names
    for t in agent.TOOLS:
        f = t["function"]
        assert t["type"] == "function" and f["description"] and f["parameters"]["type"] == "object"
        assert set(f["parameters"]["required"]) <= set(f["parameters"]["properties"])


def test_system_prompt_is_str():
    assert isinstance(agent.SYSTEM, str) and agent.SYSTEM.strip()


# ---------- run_tool ----------

def test_run_tool_errors_go_back_to_model():
    """参数不是 JSON, 块不存在, 工具名不存在: 不抛异常, 返回 "出错" 的文字交给模型, 日志记下错误, 不登记块."""
    seen = agent.Seen()
    for name, raw in [("search", "不是 JSON"), ("read", '{"chunk_id": "99-9999"}'), ("fly", "{}")]:
        text, log = agent.run_tool(LIB, seen, name, raw)
        assert log["error"] and "出错" in text, (name, text)
    assert seen.chunks == []


# ---------- run_agent ----------

def test_agent_no_presearch_then_tools_then_answer():
    """代码不预先检索; 并行的工具调用各回一条 tool 消息且 id 对应; 块编号沿用, read 正文时附带裴注; 引文核对到块."""
    script = Script(
        resp(calls=[tc("a", "search", query="浚儀")]),
        resp(calls=[tc("b", "read", chunk_id="01-0002", before=1), tc("c", "era", era="建安", year=7)]),
        resp("建安七年「遂至浚儀，治睢陽渠」[1]"))
    r = agent.run_agent("曹操什么时候去过浚仪？", LIB, complete=script)
    first = script.seen[0]["messages"]
    assert [m["role"] for m in first] == ["system", "user", "developer"], "不再由代码先检索; developer 是 LANG_NOTE"
    assert script.seen[0]["kw"].get("tools") == agent.TOOLS
    second = script.seen[1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["id"] == "a"
    assert second[-1] == {"role": "tool", "tool_call_id": "a", "content": second[-1]["content"]}
    third = script.seen[2]["messages"]
    tools = [m for m in third if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tools] == ["a", "b", "c"], "并行调用各回一条, 顺序一致"
    assert "202" in tools[-1]["content"]
    assert r["first"] == ["01-0002"] and r["quotes"] == [("遂至浚儀，治睢陽渠", "01-0002")]
    assert [c["id"] for c in r["chunks"]] == ["01-0002", "01-0001", "01-0002n0"], \
        "已见过的块沿用编号, 不重复登记; read 正文时附带它的裴注"


def test_agent_cap_by_omitting_tools():
    """模型一直调工具: 第 MAX_ROUNDS + 1 次请求不传 tools (也不传 tool_choice), 拿到文字回答."""
    replies = [resp(calls=[tc(f"x{i}", "search", query="官渡")]) for i in range(agent.MAX_ROUNDS)] + [resp("答")]
    script = Script(*replies)
    r = agent.run_agent("问题", LIB, complete=script)
    assert r["reply"] == "答" and len(script.seen) == agent.MAX_ROUNDS + 1
    assert all("tools" in x["kw"] for x in script.seen[:-1]) and "tools" not in script.seen[-1]["kw"]
    assert "tool_choice" not in script.seen[-1]["kw"]


def test_agent_answer_without_tools():
    """不调工具直接回答 (闲聊): 一次请求, 没有块."""
    r = agent.run_agent("你好", LIB, complete=Script(resp("我只讲三国历史.")))
    assert r["reply"] == "我只讲三国历史." and r["chunks"] == [] and r["first"] == [] and r["calls"] == 1


def test_agent_nudges_once_after_single_search():
    """只搜 1 次就作答: 追加一条 NUDGE 提示 (列出看过的资料出自哪里), 模型再搜后作答."""
    script = Script(
        resp(calls=[tc("a", "search", query="浚儀")]),
        resp("草稿回答"),
        resp(calls=[tc("b", "search", query="官渡")]),
        resp("最终回答"))
    r = agent.run_agent("问题", LIB, complete=script)
    assert r["reply"] == "最终回答" and len(script.seen) == 4
    nudge = script.seen[2]["messages"][-1]
    assert nudge["role"] == "developer" and "1 次" in nudge["content"], "搜 1 次就想答: 追加一条提示"
    assert "魏書•武帝紀" in nudge["content"], "提示里列出看过的资料出自哪里"
    assert nudge["content"].startswith(agent.DRAFT.format(reply="草稿回答")), "提示里带着草稿"
    assert {"role": "assistant", "content": "草稿回答"} not in script.seen[2]["messages"], "草稿不作为 assistant 消息放回对话"
    assert [x["tool"] for x in r["rounds"]] == ["search", "nudge", "search"]


def who_call(cid: str) -> SimpleNamespace:
    """调 who 的 tool_call (tc() 的参数名 name 和 who 的参数撞名, 单独造)."""
    return SimpleNamespace(id=cid, function=SimpleNamespace(name="who", arguments='{"name": "曹操"}'))


def test_agent_nudge_on_second_to_last_request_then_final():
    """倒数第二次请求作答被提示: 下一次就是最后一次 (追加 FINAL_TURN, 不传 tools), 总请求数不超过 MAX_ROUNDS + 1."""
    n = agent.MAX_ROUNDS
    script = Script(*[resp(calls=[who_call(f"w{i}")]) for i in range(n - 1)], resp("草稿"), resp("终稿"))
    r = agent.run_agent("问题", LIB, complete=script)
    assert r["reply"] == "终稿" and len(script.seen) == n + 1
    last = script.seen[-1]
    assert "tools" not in last["kw"] and last["messages"][-1]["content"] == agent.FINAL_TURN
    assert last["messages"][-2]["role"] == "developer" and last["messages"][-2]["content"].startswith(agent.DRAFT.format(reply="草稿"))
    assert "nudge" in [x["tool"] for x in r["rounds"]]


def test_agent_no_nudge_after_last_request():
    """最后一次请求 (不带 tools) 的回答不再检查: 只搜过 1 次也不提示."""
    n = agent.MAX_ROUNDS
    script = Script(resp(calls=[tc("a", "search", query="浚儀")]),
                    *[resp(calls=[who_call(f"w{i}")]) for i in range(n - 1)], resp("最后的回答"))
    r = agent.run_agent("问题", LIB, complete=script)
    assert r["reply"] == "最后的回答" and len(script.seen) == n + 1
    assert "nudge" not in [x["tool"] for x in r["rounds"]]


def test_agent_nudge_only_once_and_not_without_search():
    """提示只追加一次, 模型坚持原答案就收下; 一次工具都没用时不提示."""
    script = Script(resp(calls=[tc("a", "search", query="浚儀")]), resp("草稿"), resp("还是这个答案"))
    r = agent.run_agent("问题", LIB, complete=script)
    assert r["reply"] == "还是这个答案" and len(script.seen) == 3, "只提示一次, 模型坚持就收下"
    r = agent.run_agent("你好", LIB, complete=Script(resp("我只讲三国历史.")))
    assert r["calls"] == 1, "一次都没搜 (闲聊) 不提示"


# ---------- 白话切块: 展示原文 + 白话, 引文只认原文 ----------

def test_seen_shows_translation_and_quotes_check_original_only():
    """白话切块: 交给模型的资料有原文和【白话译文】, chunk 的 text 仍只有原文; 引白话的引文判为找不到."""
    c = {**CHUNKS[1], "trans": "建安七年春正月，曹公驻军谯县。"}
    lib = agent.Library([CHUNKS[0], c] + CHUNKS[2:], lambda q, k, where=None: [c], ERAS)
    script = Script(resp(calls=[tc("c1", "search", query="谯")]), resp(calls=[tc("c2", "read", chunk_id="01-0002")]),
                    resp("「公軍譙」[1]「曹公驻军谯县」[1]"))
    r = agent.run_agent("曹操七年去了哪", lib, complete=script)
    shown = script.seen[1]["messages"][-1]["content"]
    assert "七年春正月，公軍譙" in shown and "【白话译文】建安七年春正月，曹公驻军谯县。" in shown
    assert "【白话译文】" not in r["chunks"][0]["text"], "译文不能拼进 chunk 的 text"
    assert dict(r["quotes"]) == {"公軍譙": "01-0002", "曹公驻军谯县": None}, "引白话判成找不到"


# ---------- who: 别名表查人 ----------

def _fake_aliases(data: dict) -> Callable[[], None]:
    """临时换掉 alias.load_aliases (清掉缓存), 返回恢复函数."""
    from rag import alias
    orig = alias.load_aliases
    alias.alias_index.cache_clear()
    alias.load_aliases = lambda: data
    library.load_aliases = alias.load_aliases

    def restore():
        alias.load_aliases = orig
        library.load_aliases = orig
        alias.alias_index.cache_clear()
    return restore


ALIASES = {"people": [
    {"id": "43-0011", "name": "马忠", "surname": "马", "zi": "德信", "forms": [{"form": "彭乡亭侯", "orig": "彭鄉亭侯", "src": "43-0011", "kind": "爵号"}]},
    {"id": "32-0000", "name": "刘备", "surname": "刘", "zi": "玄德", "forms": [{"form": "先主", "orig": "先主", "src": "32-0000", "kind": "称呼"}]}],
    "homonyms": {"马忠": [{"src": "47-0016", "desc": "孙权部将, 擒关羽"}]}, "variants": {}, "variant_surnames": {}}


def test_who_title_and_homonyms():
    """who 查称号 (先主 -> 刘备) 和同名 (马忠 2 人), 查不到时说没有; 只调 who 不 search 就作答时也会被提示再查."""
    restore = _fake_aliases(ALIASES)
    try:
        t = LIB.who("先主")
        assert "刘备" in t and "32-0000" in t and "刘玄德" in t
        t = LIB.who("马忠")
        assert "共 2 个人" in t and "43-0011" in t and "47-0016" in t and "擒关羽" in t
        assert "没有" in LIB.who("许贡")
        call = SimpleNamespace(id="c1", function=SimpleNamespace(name="who", arguments='{"name": "先主"}'))
        r = agent.run_agent("先主是谁", LIB, complete=Script(resp(calls=[call]), resp("刘备"), resp("刘备")))
        assert r["rounds"][0]["tool"] == "who" and r["rounds"][0]["error"] is None
    finally:
        restore()

    restore = _fake_aliases(ALIASES)
    try:
        call = SimpleNamespace(id="c1", function=SimpleNamespace(name="who", arguments='{"name": "先主"}'))
        script = Script(resp(calls=[call]), resp("刘备"), resp(calls=[tc("c2", "search", query="浚儀")]), resp("答"))
        r = agent.run_agent("先主是谁", LIB, complete=script)
        assert any(x["tool"] == "nudge" for x in r["rounds"]), "只调 who 不 search 就作答, 要提示"
    finally:
        restore()


def test_markup_reply_is_retried_and_stripped():
    """模型 (如 deepseek) 把工具调用写进文字 (<｜DSML｜tool_calls>) 而不是放进 tool_calls: 回答判为坏回答,
    以 developer 角色追加 FINAL_TURN 要求重答一次, 并去掉回答里残留的标记."""
    markup = "\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"search\">x</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
    script = Script(resp(calls=[tc("c1", "search", query="浚儀")]), resp(calls=[tc("c2", "search", query="官渡")]),
                    resp(markup), resp("魏太祖七年至浚仪。" + markup))
    r = agent.run_agent("曹操去过浚仪吗", LIB, complete=script)
    assert r["reply"] == "魏太祖七年至浚仪。", r["reply"]
    assert any(x["tool"] == "retry" for x in r["rounds"])
    assert script.seen[-1]["messages"][-1] == {"role": "developer", "content": agent.FINAL_TURN}, "重答前的提示用 developer 角色"


def test_review_replaces_reply_and_skips_without_tools():
    """自查 (默认 layered 模式) 的回答替换原回答; 自查返回拒答模板时保留原回答; 没用工具时不自查."""
    script = Script(resp(calls=[tc("c1", "search", query="浚儀")]), resp(calls=[tc("c2", "search", query="官渡")]),
                    resp("初稿, 带一句资料外的背景。"), resp("改后的回答, 去掉了资料外的背景。"))
    r = agent.run_agent("曹操去过浚仪吗", LIB, complete=script, review=True)
    assert r["reply"] == "改后的回答, 去掉了资料外的背景。" and r["rounds"][-1]["tool"] == "review"
    script = Script(resp(calls=[tc("c1", "search", query="浚儀")]), resp(calls=[tc("c2", "search", query="官渡")]),
                    resp("魏太祖建安七年至浚仪, 治睢阳渠。"), resp("作为一个人工智能语言模型，我还没学习如何回答这个问题"))
    r = agent.run_agent("曹操去过浚仪吗", LIB, complete=script, review=True)
    assert r["reply"] == "魏太祖建安七年至浚仪, 治睢阳渠。", "自查返回拒答模板时保留原回答"
    review_msg = script.seen[-1]["messages"][-1]
    assert review_msg == {"role": "developer", "content": agent.DRAFT.format(reply="魏太祖建安七年至浚仪, 治睢阳渠。") + agent.REVIEW_TEXT_LAYERED + " " + agent.LANG_NOTE}, "默认 REVIEW_MODE layered"
    assert script.seen[-1]["messages"][-2]["role"] != "assistant", "自查不把草稿作为 assistant 消息放回 (Claude 原生接口不许 system 跟在 assistant 文字后面)"
    assert all(not (m.get("content") or "").startswith(agent.DRAFT[:10]) for m in r["messages"]), "自查用的指令请求完就撤掉"
    r = agent.run_agent("你好", LIB, complete=Script(resp("我只讲三国历史.")), review=True)
    assert r["calls"] == 1, "没用工具 (闲聊) 不自查"


def test_review_short_keeps_reply_when_ok():
    """REVIEW_MODE "short": 自查回 "无误" 时保留原回答; 回了改后的回答时照常替换."""
    orig = agent.REVIEW_MODE
    agent.REVIEW_MODE = "short"
    try:
        script = Script(resp(calls=[tc("c1", "search", query="浚儀")]), resp(calls=[tc("c2", "search", query="官渡")]),
                        resp("魏太祖建安七年至浚仪。"), resp(" 无误。"))
        r = agent.run_agent("曹操去过浚仪吗", LIB, complete=script, review=True)
        assert r["reply"] == "魏太祖建安七年至浚仪。" and r["rounds"][-1]["args"] == {"ok": True}
        assert script.seen[-1]["messages"][-1]["content"] == agent.DRAFT.format(reply="魏太祖建安七年至浚仪。") + agent.REVIEW_TEXT_SHORT + " " + agent.LANG_NOTE
        script = Script(resp(calls=[tc("c1", "search", query="浚儀")]), resp(calls=[tc("c2", "search", query="官渡")]),
                        resp("初稿, 带一句资料外的背景。"), resp("改后的回答, 去掉了资料外的背景。"))
        r = agent.run_agent("曹操去过浚仪吗", LIB, complete=script, review=True)
        assert r["reply"] == "改后的回答, 去掉了资料外的背景。"
    finally:
        agent.REVIEW_MODE = orig


def test_not_found_answer_gets_one_more_try():
    """回答说没找到 / 没有记载, 而搜得少或没看过传末评时, 要求核实一次 (nudge_notfound, 默认开); 只追加一次."""
    assert "nudge_notfound" not in agent.OFF
    _not_found_once()


def test_not_found_pattern_covers_english_answers():
    """英文回答说 "does not mention / no record" 也算 "没有记载" (nudge_notfound 对英文题也起作用); 普通句子不算."""
    for t in ["The Sanguozhi does not mention a White Horse General.", "There is no record of this.", "原文没有记载"]:
        assert agent.NOT_FOUND.search(t), t
    assert not agent.NOT_FOUND.search("Zhou Tai was a general of Wu.")


def test_not_found_after_enough_searches_and_ping_is_left_alone():
    """已经 search 了 MIN_SEARCHES_NOTFOUND 次, 也看过传末评 ("評曰" 开头的块) 时, 说 "没有记载" 不再提示 (原文确实没有的正当回答)."""
    ping = {"id": "01-0090", "tier": 1, "juan": 1, "juan_title": "魏書•武帝紀", "headings": ["評"], "parent": None, "cites": [],
            "sources": ["01-0090"], "text": "評曰：漢末，天下大亂。"}
    chunks = CHUNKS + [ping]
    table = {**TABLE, "評": ["01-0090"]}
    lib = agent.Library(chunks, lambda q, k, where=None: [c for c in chunks if c["id"] in table.get(q, [])][:k], ERAS)
    queries = ["浚儀", "官渡", "評", "官渡"]
    script = Script(*[resp(calls=[tc(f"c{i}", "search", query=q)]) for i, q in enumerate(queries)],
                    resp("《三国志》里没有记载曹操去过许都以外的这个地方."))
    r = agent.run_agent("曹操去过浚仪吗", lib, complete=script)
    assert "nudge_notfound" not in [x["tool"] for x in r["rounds"]]


def test_scope_refusal_without_tools_gets_one_search():
    """一次工具没用就说 "只讲三国" 的拒答 (nudge_scope): 默认关 (agent.OFF 里有它), 不提示;
    打开后追加一次 SCOPE_NUDGE, 模型搜一次再答. 一次工具没用的普通回答 (不是拒答) 不提示."""
    assert "nudge_scope" in agent.OFF
    script = Script(resp("我只讲三国的历史, 这个问题不在我能回答的范围内."))
    r = agent.run_agent("有个富豪路上搭载了一位妇人, 富豪是谁", LIB, complete=script)
    assert "nudge_scope" not in [x["tool"] for x in r["rounds"]] and len(script.seen) == 1
    agent.OFF.discard("nudge_scope")
    try:
        script = Script(resp("我只讲三国的历史, 这个问题不在我能回答的范围内."),
                        resp(calls=[tc("c1", "search", query="富豪 妇人 天使")]), resp(calls=[tc("c2", "search", query="浚儀")]),
                        resp("找到了: 公軍譙."))
        r = agent.run_agent("有个富豪路上搭载了一位妇人, 富豪是谁", LIB, complete=script)
        assert [x["tool"] for x in r["rounds"]].count("nudge_scope") == 1 and r["reply"] == "找到了: 公軍譙."
        assert script.seen[1]["messages"][-1]["content"] == agent.DRAFT.format(reply="我只讲三国的历史, 这个问题不在我能回答的范围内.") + agent.SCOPE_NUDGE
        script = Script(resp("你好, 我是讲三国历史的助手."))
        r = agent.run_agent("你好", LIB, complete=script)
        assert "nudge_scope" not in [x["tool"] for x in r["rounds"]] and len(script.seen) == 1
    finally:
        agent.OFF.add("nudge_scope")


def test_lang_note_after_question_and_in_review():
    """lang_note (默认开): 问题后面紧跟一条 developer 消息 LANG_NOTE, 自查指令末尾也加上; 放进 OFF 后都没有."""
    assert "lang_note" not in agent.OFF
    script = Script(resp(calls=[tc("c1", "search", query="浚儀")]), resp(calls=[tc("c2", "search", query="官渡")]),
                    resp("draft answer"), resp("checked answer"))
    agent.run_agent("Did Cao Cao go to Junyi?", LIB, complete=script, review=True)
    first = script.seen[0]["messages"]
    assert first[-1]["role"] == "developer" and first[-1]["content"].endswith(agent.LANG_NOTE) and first[-2]["role"] == "user"
    assert "Cao Cao" in first[-1]["content"], "问题里的拼音人名 (pinyin_who) 和语言提醒合在一条 developer 消息里"
    assert script.seen[-1]["messages"][-1]["content"].endswith(agent.LANG_NOTE)
    agent.OFF.add("lang_note")
    try:
        script = Script(resp(calls=[tc("c1", "search", query="浚儀")]), resp(calls=[tc("c2", "search", query="官渡")]), resp("answer"))
        agent.run_agent("Did Cao Cao go to Junyi?", LIB, complete=script)
        assert all(agent.LANG_NOTE not in (m.get("content") or "") for m in script.seen[0]["messages"])
    finally:
        agent.OFF.discard("lang_note")


def test_aggregate_question_needs_three_searches():
    """汇总题 (有哪些, 有几个, how many ...) 搜两次就想作答时也提示再查 (门槛 MIN_SEARCHES_AGG); 普通题搜两次不提示."""
    script = Script(resp(calls=[tc("c1", "search", query="浚儀")]), resp(calls=[tc("c2", "search", query="官渡")]),
                    resp("初答"), resp(calls=[tc("c3", "search", query="浚儀")]), resp("再答"))
    r = agent.run_agent("曹操有几个儿子", LIB, complete=script)
    assert [x["tool"] for x in r["rounds"]].count("nudge") == 1 and r["reply"] == "再答"
    script = Script(resp(calls=[tc("c1", "search", query="浚儀")]), resp(calls=[tc("c2", "search", query="官渡")]), resp("答"))
    r = agent.run_agent("曹操去过浚仪吗", LIB, complete=script)
    assert "nudge" not in [x["tool"] for x in r["rounds"]]


def test_english_question_with_chinese_answer_is_rewritten_once():
    """英文问题得到中文回答 (去掉引文后大多是汉字) 时, 要求改写成英文一次 (lang_fix, 默认开); 中文问题不管."""
    script = Script(resp(calls=[tc("c1", "search", query="浚儀")]), resp(calls=[tc("c2", "search", query="官渡")]),
                    resp("去过: 「遂至浚儀」[1]。"), resp("Yes. 「遂至浚儀」[1] (he went on to Junyi)."))
    r = agent.run_agent("Did Cao Cao go to Junyi?", LIB, complete=script)
    assert r["reply"].startswith("Yes.") and [x["tool"] for x in r["rounds"]].count("lang_fix") == 1
    assert script.seen[-1]["messages"][-1]["content"] == agent.DRAFT.format(reply="去过: 「遂至浚儀」[1]。") + agent.LANG_FIX
    script = Script(resp(calls=[tc("c1", "search", query="浚儀")]), resp(calls=[tc("c2", "search", query="官渡")]), resp("去过 [1]"))
    r = agent.run_agent("曹操去过浚仪吗", LIB, complete=script)
    assert "lang_fix" not in [x["tool"] for x in r["rounds"]]


def _not_found_once():
    script = Script(resp(calls=[tc("c1", "search", query="浚儀")]), resp(calls=[tc("c2", "search", query="官渡")]),
                    resp("《三国志》里没有找到相关记载."), resp(calls=[tc("c3", "search", query="浚儀")]), resp("找到了: 公軍譙."))
    r = agent.run_agent("曹操去过浚仪吗", LIB, complete=script)
    assert [x["tool"] for x in r["rounds"]].count("nudge_notfound") == 1 and r["reply"] == "找到了: 公軍譙."
    assert script.seen[3]["messages"][-1]["content"] == agent.DRAFT.format(reply="《三国志》里没有找到相关记载.") + agent.NOT_FOUND_NUDGE


def test_unverified_quote_is_sent_back_once():
    """引文核对不过 (原文是 "公軍譙", 回答写成 "公軍譙縣") 的, 回给模型改一次; 改好后重新核对."""
    script = Script(resp(calls=[tc("c1", "search", query="浚儀")]), resp(calls=[tc("c2", "search", query="官渡")]),
                    resp("魏太祖建安七年至浚仪: 「公軍譙縣」[1]"),        # 初稿
                    resp("魏太祖建安七年至浚仪: 「公軍譙縣」[1]"),        # 自查后仍是错引文
                    resp("魏太祖建安七年至浚仪: 「公軍譙」[1]"))          # 引文修正
    r = agent.run_agent("曹操去过浚仪吗", LIB, complete=script, review=True)
    assert [x["tool"] for x in r["rounds"]][-2:] == ["review", "quote_fix"]
    assert "「公軍譙縣」" in script.seen[-1]["messages"][-1]["content"]
    assert r["reply"].endswith("「公軍譙」[1]") and r["quotes"] == [("公軍譙", "01-0002")]


def test_history_goes_between_system_and_question():
    """多轮对话: 只带最近 HISTORY_TURNS 轮的 user / assistant 消息, 放在 system 和本轮问题之间."""
    hist = [{"role": "user", "content": f"问{i}"} if i % 2 == 0 else {"role": "assistant", "content": f"答{i}"} for i in range(10)]
    hist.insert(3, {"role": "tool", "content": "不该带上"})
    script = Script(resp("我只讲三国历史."))
    agent.run_agent("他儿子呢", LIB, complete=script, history=hist)
    msgs = script.seen[0]["messages"]
    assert msgs[0]["role"] == "system" and msgs[-2] == {"role": "user", "content": "他儿子呢"}
    assert msgs[-1] == {"role": "developer", "content": agent.LANG_NOTE}
    middle = msgs[1:-2]
    assert len(middle) == 2 * agent.HISTORY_TURNS and middle[-1]["content"] == "答9", "只带最近几轮"
    assert all(m["role"] in ("user", "assistant") for m in middle)


def test_on_event_reports_tools_and_stages():
    """on_event 依次收到 call / tool / stage 事件; 回调抛异常不影响回答."""
    events = []
    script = Script(resp(calls=[tc("c1", "search", query="浚儀")]), resp(calls=[tc("c2", "search", query="官渡")]),
                    resp("魏太祖建安七年至浚仪。"), resp("魏太祖建安七年至浚仪, 治睢阳渠。"))
    agent.run_agent("曹操去过浚仪吗", LIB, complete=script, review=True, on_event=lambda k, i: events.append((k, i)))
    kinds = [k for k, _ in events]
    assert kinds.count("tool") == 2 and ("stage", "review") in events and kinds[0] == "call"
    assert [i["args"]["query"] for k, i in events if k == "tool"] == ["浚儀", "官渡"]
    agent.run_agent("你好", LIB, complete=Script(resp("我只讲三国历史.")), on_event=lambda k, i: 1 / 0)  # 回调出错不影响



def test_apply_patch():
    """patch 自查的输出逐条套用: 改句, 删句 (多余空行合并), 抄错的原文跳过并计数; "无误" 之类没有改动的输出不改回答."""
    reply = "甲做了A。乙做了B。\n\n以下不是《三国志》的记载: 丙在今天某地。"
    patch = "原文: 乙做了B。\n改为: 乙做了C [2]。\n原文: 以下不是《三国志》的记载: 丙在今天某地。\n改为: 删除（这是背景段, 不需要）\n原文: 丁做了D。\n改为: 丁没做。"
    new, done, missed = agent.apply_patch(reply, patch)
    assert new == "甲做了A。乙做了C [2]。" and (done, missed) == (2, 1)
    assert agent.apply_patch(reply, "无误") == (reply.strip(), 0, 0)
    noisy = "原文: 甲做了A。\n改为: 无误\n原文: 乙做了B。\n改为: 乙做了B [1]。（事实，依据 [1]，无误）\n原文: 以下不是《三国志》的记载: 丙在今天某地。\n改为: （保留）"
    assert agent.apply_patch(reply, noisy) == ("甲做了A。乙做了B [1]。\n\n以下不是《三国志》的记载: 丙在今天某地。", 1, 0)


def test_patch_review_and_draft_event():
    """REVIEW_MODE = "patch": 先发 draft 事件 (初稿), 自查只回改动, 代码替换后作为最终回答."""
    old_mode, agent.REVIEW_MODE = agent.REVIEW_MODE, "patch"
    try:
        events = []
        script = Script(resp(calls=[tc("c1", "search", query="浚儀")]), resp(calls=[tc("c2", "search", query="譙")]),
                        resp("太祖至浚仪 [1]。另有一句没依据的话。"), resp("原文: 另有一句没依据的话。\n改为: 删除"))
        res = agent.run_agent("曹操去过浚仪吗", LIB, complete=script, review=True, on_event=lambda k, i: events.append((k, i)))
        assert ("draft", "太祖至浚仪 [1]。另有一句没依据的话。") in events
        assert res["reply"] == "太祖至浚仪 [1]。"
        assert any(r["tool"] == "review" and r["args"].get("edits") == 1 for r in res["rounds"])
    finally:
        agent.REVIEW_MODE = old_mode

if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
