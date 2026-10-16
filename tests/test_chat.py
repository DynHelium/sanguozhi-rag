"""app/chat.py 的测试. 不依赖 pytest / gradio, 不连网, 不加载 bge-m3:

    uv run python -m tests.test_chat

假模型和假资料沿用 tests/test_agent.py (Script / resp / tc / LIB); 关掉 agent.REVIEW (回答后的自查),
让每题的模型请求数正好等于给 Script 的回复数.
"""
from types import SimpleNamespace

import tempfile
from pathlib import Path

from app import chat
from rag import agent, skills
from tests.test_agent import LIB, Script, resp, tc

agent.REVIEW = False
skills.TRACE_DIR = Path(tempfile.mkdtemp())  # 逐步记录写到临时目录, 不进 data/


def run(question: str, history: list[dict], *replies: SimpleNamespace) -> tuple[list[list[dict]], Script]:
    """跑一轮 respond, 返回 (每次 yield 的消息列表, 假模型)."""
    script = Script(*replies)
    respond = chat.make_respond(LIB, complete_factory=lambda: script)
    return list(respond(question, history)), script


# ---------- to_agent_history ----------

def test_history_keeps_only_questions_and_answers():
    """去掉折叠块 (metadata 带 title) 和空消息, 列表形式的 content 转成字符串; metadata 为 None 算普通消息."""
    page = [
        {"role": "user", "content": "曹操去过浚仪吗"},
        {"role": "assistant", "content": "· 搜索: 浚儀", "metadata": {"title": "检索过程 (1 步)", "status": "done"}},
        {"role": "assistant", "content": "去过 [1]"},
        {"role": "assistant", "content": "[1] ...", "metadata": {"title": "出处 (1 段)"}},
        {"role": "user", "content": [{"type": "text", "text": "后来呢"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "进军官渡"}], "metadata": None},
        {"role": "assistant", "content": ""},
    ]
    assert chat.to_agent_history(page) == [
        {"role": "user", "content": "曹操去过浚仪吗"}, {"role": "assistant", "content": "去过 [1]"},
        {"role": "user", "content": "后来呢"}, {"role": "assistant", "content": "进军官渡"}]


def test_history_empty():
    assert chat.to_agent_history([]) == []


# ---------- describe / render_sources ----------

def test_describe_lines():
    """工具调用显示工具和参数及取回段数, 出错标出错; stage 事件有对应文字; 其他事件 (call) 不显示."""
    assert chat.describe("tool", {"tool": "search", "args": {"query": "浚儀"}, "chunks": ["01-0002"], "error": None}) == "· 搜索: 浚儀 → 1 段"
    assert "出错" in chat.describe("tool", {"tool": "read", "args": {"chunk_id": "x"}, "chunks": [], "error": "没有这一块"})
    assert chat.describe("stage", "review").startswith("· 核对")
    assert chat.describe("call", 1) is None


def test_sources_only_cited_and_in_range():
    """出处只列回答引用到且在范围内的编号; 附引文核对结果, 越界编号单独标出."""
    chunks = [c for c in LIB.chunks if c["id"] in ("01-0002", "01-0003")]
    cites = [{"quote": "公軍譙", "cites": [1], "status": "ok", "found": [1]},
             {"quote": "别处的话", "cites": [1], "status": "wrong", "found": [2]}]
    md = chat.render_sources({"reply": "至浚仪 [1], 又 [9]", "chunks": chunks, "cites": cites})
    assert "01-0002" in md and "01-0003" not in md, "只列回答引用到的编号, 越界的 [9] 不列出处"
    assert "✓" in md and "出处标错, 文字在 [2]" in md
    assert "不存在的编号: [9]" in md


# ---------- respond (整轮) ----------

def test_respond_streams_progress_then_answer():
    """先 yield 进行中的检索过程, 最后一次是 [检索过程 (done), 最终回答 (普通消息), 出处]."""
    outs, script = run("曹操去过浚仪吗", [],
                       resp(calls=[tc("c1", "search", query="浚儀")]), resp(calls=[tc("c2", "search", query="官渡")]),
                       resp("七年春正月, 公軍譙, 遂至浚儀「公軍譙」[1]"))
    assert outs[0][0]["metadata"]["status"] == "pending", "先显示检索过程 (进行中)"
    final = outs[-1]
    assert final[0]["metadata"]["status"] == "done" and "搜索: 浚儀" in final[0]["content"] and "搜索: 官渡" in final[0]["content"]
    assert final[1]["content"].endswith("[1]") and "metadata" not in final[1], "最终回答是普通消息"
    assert final[2]["metadata"]["title"].startswith("出处") and "01-0002" in final[2]["content"]


def test_respond_passes_history_to_agent():
    """上一轮的问题和最终回答 (不含检索过程) 接在 system prompt 后面发给模型."""
    page = [{"role": "user", "content": "曹操去过浚仪吗"},
            {"role": "assistant", "content": "· 搜索", "metadata": {"title": "检索过程 (1 步)"}},
            {"role": "assistant", "content": "去过"}]
    _, script = run("他后来去了哪", page, resp("我只讲三国历史."))
    msgs = script.seen[0]["messages"]
    assert [m["content"] for m in msgs[1:]] == ["曹操去过浚仪吗", "去过", "他后来去了哪"]


def test_respond_shows_error():
    """模型请求抛异常时, 最后一条消息显示错误, 不让页面卡住."""
    def boom(messages, model=None, **kw):
        raise RuntimeError("服务商挂了")
    respond = chat.make_respond(LIB, complete_factory=lambda: boom)
    outs = list(respond("曹操是谁", []))
    assert "出错" in outs[-1][-1]["content"] or "服务商挂了" in outs[-1][-1]["content"]


def test_respond_ignores_blank():
    """空白问题什么都不 yield."""
    assert list(chat.make_respond(LIB)("  ", [])) == []


def test_describe_tolerates_non_object_args():
    """模型给的工具参数不是 JSON 对象 (如 [1]) 时, 网页显示一行而不是抛异常中断回答."""
    line = chat.describe("tool", {"tool": "search", "args": [1], "chunks": [], "error": "TypeError"})
    assert isinstance(line, str) and line.startswith("· 搜索") and "出错" in line



# ---------- 限制和停止 ----------

def test_limits_stop_before_calling_model():
    """问题太长, 今天的预算用完: 只回一条说明, 不调模型."""
    out, script = run("问" * (chat.MAX_QUESTION_CHARS + 1), [])
    assert len(out) == 1 and "太长" in out[0][0]["content"] and script.seen == []
    budget = chat.Budget(0.01)
    budget.add(0.02)
    respond = chat.make_respond(LIB, complete_factory=lambda: Script(), budget=budget)
    out = list(respond("曹操去过浚仪吗", []))
    assert "额度" in out[0][0]["content"]


def test_stop_cancels_remaining_requests():
    """用户点停止 (关闭生成器) 后, 后台线程不再发模型请求."""
    import threading
    gate = threading.Event()
    script = Script(resp(calls=[tc("c1", "search", query="浚儀")]), resp("不该用到的回答"))
    first = script.__call__

    def slow(*a, **kw):  # 第一次请求等页面关掉再返回
        r = first(*a, **kw)
        gate.wait(5)
        return r
    respond = chat.make_respond(LIB, complete_factory=lambda: slow)
    gen = respond("曹操去过浚仪吗", [])
    next(gen)
    gen.close()
    gate.set()
    import time
    time.sleep(0.5)
    assert len(script.seen) == 1, "停止后不应再请求模型"

def test_respond_routes_game_with_progress():
    """触发词交给游戏 (不调问答 agent); 游戏这一问的进度显示在 "思考过程" 折叠框里, 最后是游戏的回复."""
    class FakeRouter:
        def claims(self, sid, text):
            return text == "猜人物"

        def handle(self, sid, text, on_event=None, cancel=None):
            on_event("tool", {"tool": "search", "args": {"query": "刘备"}, "chunks": ["32-0000"], "error": None})
            return "第 1 问: 他是男性吗？"
    respond = chat.make_respond(LIB, complete_factory=lambda: Script(), router=FakeRouter())
    out = list(respond("猜人物", [], "s1"))
    assert out[-1][-1]["content"] == "第 1 问: 他是男性吗？"
    assert out[-1][0]["metadata"]["title"] == "思考过程 (1 步)" and "搜索: 刘备" in out[-1][0]["content"]



class EchoGame:
    """假游戏: 开局说 "开局了"; "怎么玩" 不算游戏里的话 (返回 None), 别的都答 "是"."""
    over = False

    def intro(self, on_event=None):
        return "开局了"

    def ask(self, text, on_event=None):
        return None if "怎么玩" in text else "是"

    def reveal(self):
        return "谜底是 X"


def test_agent_starts_game_and_takes_non_game_messages():
    """问答 agent 调 start_game 就开局 (显示游戏的开局说明); 游戏中游戏不管的话交给 agent, system prompt 里带着正在玩的游戏."""
    router = skills.Router({"guess_person": EchoGame}, skills.load_skills())
    script = Script(resp(calls=[tc("c1", "start_game", game="guess_person")]))
    respond = chat.make_respond(LIB, complete_factory=lambda: script, router=router)
    out = list(respond("我们来玩猜人物吧", [], "u"))
    assert out[-1][-1]["content"] == "【猜人物】开局了" and router.active("u") is not None
    assert any(t["function"]["name"] == "start_game" for t in script.seen[0]["kw"]["tools"])
    assert list(respond("他是男的吗", [], "u"))[-1][-1]["content"] == "是"
    script2 = Script(resp("先问是非题, 最后说名字."))
    respond2 = chat.make_respond(LIB, complete_factory=lambda: script2, router=router)
    assert list(respond2("这游戏怎么玩", [], "u"))[-1][-1]["content"].startswith("先问是非题")
    assert "正在玩 猜人物" in script2.seen[0]["messages"][0]["content"]


def test_new_message_interrupts_old_one():
    """同一个 sid 发新消息时, 还在进行的那一问被取消: 不再请求模型, 也不再刷新页面."""
    import threading
    gate = threading.Event()
    old_script = Script(resp(calls=[tc("c1", "search", query="浚儀")]), resp("旧回答"))
    new_script = Script(resp("新回答"))

    def slow(*a, **kw):  # 旧的一问: 第一次请求等新消息来了才返回
        r = old_script(*a, **kw)
        gate.wait(5)
        return r
    factories = iter([lambda: slow, lambda: new_script])
    respond = chat.make_respond(LIB, complete_factory=lambda: next(factories)())
    old = respond("曹操去过浚仪吗", [], "s")
    next(old)
    new = list(respond("刘备是谁", [], "s"))
    gate.set()
    assert list(old) == [] and new[-1][-1]["content"] == "新回答"
    assert len(old_script.seen) == 1, "被打断后不再请求模型"

if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
