"""网页聊天: Gradio 薄包装. 检索和问答全在 rag/agent.py, 这里只管把过程和结果显示出来.

    uv sync --extra index --extra ui
    uv run python -m app.chat --model deepseek-v4.1-flash --effort low   # 后端和 key 在 .env (LLM_BACKEND); .env 设 LLM_BACKEND=anthropic 后用 --model claude-haiku-5-5 --effort medium
    uv run python -m app.chat                    # 不加参数: 模型取 .env 的 LLM_MODEL, 后端取 LLM_BACKEND
    uv run python -m tests.test_chat             # 假模型测试, 不连网, 不开网页

一轮对话在页面上是这样 (Gradio 的 messages 格式, 每条是一个 dict):

    history = [
        {"role": "user", "content": "曹操因为什么逃离洛阳？"},
        {"role": "assistant", "content": "· 搜索: 曹操 逃离洛阳 → 10 段\n· 读原文: 01-0004 → 5 段 ...",   <- 检索过程 (折叠)
         "metadata": {"title": "检索过程 (5 步)", "status": "done"}},
        {"role": "assistant", "content": "董卓表太祖为骁骑校尉 ... [1]"},                             <- 最终回答
        {"role": "assistant", "content": "**[1]** 三国志·魏書•武帝紀 正文 (01-0004)\n\n> 金城邊章 ...",
         "metadata": {"title": "出处 (2 段)", "status": "done"}},                                     <- 引用的资料 (折叠)
        {"role": "user", "content": [{"type": "text", "text": "他后来去了哪里？"}]},                  <- Gradio 6 回传时 content 可能是列表
        ...
    ]

带 metadata.title 的是显示用的折叠块, 不是模型说的话; 下一轮交给 run_agent 的 history 只取用户问题和最终回答
(to_agent_history). content 可能是字符串, 也可能是 [{"type": "text", "text": ...}] 列表, text_of 两种都认.

一题约 1 分钟 (请求模型约 5 次), 所以 run_agent 放在后台线程跑, on_event 回调把进度放进队列,
前台每收到一条就刷新一次 "检索过程", 用户不用干等.

聊天里还能玩游戏 (猜人物, 海龟汤, 见 rag/skills.py): 问答 agent 多了 start_game / end_game 两个工具, 对方说想玩时由它开局;
"猜人物" "海龟汤" 按钮发快捷命令直接开局. 游戏进行中, 消息先交给游戏, 游戏判断不是游戏里的话时再交给问答 agent;
提示 / 放弃 / 退出 三个按钮只在游戏中可点.

停止: 点停止, 或在回答过程中直接发新消息, 都会取消正在进行的那一问 (同一个浏览器 id 同时只跑一问): 后台线程在下一次模型请求前
抛 Cancelled, 已经发出的那次请求无法撤回, 结果丢掉. 每一问的模型请求 (耗时, token, 工具调用) 记进 rag/skills.trace 的日志. 对话历史和浏览器 id 存在
gr.BrowserState (浏览器 localStorage) 里, 刷新页面后对话和正在玩的游戏都还在; "清除对话" 清掉聊天记录和游戏 (见 build_ui).
"""
import argparse
import contextvars
from datetime import date
import queue
import re
import threading
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from rag import agent, config, llm
from rag.quotes import source_label
from rag.skills import Cancelled, SkillReply, logged_request, set_context, trace

if TYPE_CHECKING:
    from rag.skills import Router

CITE = re.compile(r"\[(\d+)\]")  # 回答里的资料编号 [n]
EXAMPLES = ["曹操因为什么逃离洛阳？", "昭烈皇帝有哪些儿子？", "三国志里有几个马忠？", "Which emperor was stabbed to death in the street?"]
CJK = re.compile(r"[\u4e00-\u9fff]")

# 运行时的提示 (进度, 折叠框标题, 出处, 限制和出错说明): 跟着这一问用的语言 (lang_of) 显示; 界面上固定的文字见 UI_TEXT
MSG: dict[str, dict[str, str]] = {
    "zh": {
        "search": "搜索: {q}", "read": "读原文: {id}", "who": "查人: {name}", "era": "年号换算: {era}{year}年",
        "start_game": "开始游戏: {game}", "end_game": "结束游戏", "tool_error": " (出错: {e})", "chunks": " → {n} 段",
        "nudge": "资料还不够, 再查一次", "nudge_notfound": "回答说没找到, 换说法再搜", "retry": "回答不完整, 重新作答",
        "review": "核对回答 (逐句对照资料)", "quote_fix": "有引文对不上原文, 修改引文", "lang_fix": "改写成提问用的语言",
        "thinking": "· 思考中……", "steps": "{title} ({n} 步)", "search_title": "检索过程", "think_title": "思考过程",
        "sources": "出处 ({n} 段)", "trans": "白话", "check": "引文核对", "ok": " ✓", "missing": " ✗ 原文里找不到",
        "bad_number": " ✗ 没有这个编号", "misplaced": " ✗ 出处标错, 文字在 ", "bad_nums": "回答里有不存在的编号: ",
        "draft": "*（初稿，正在逐句核对资料，核对完会更新）*", "no_answer": "(没有得到回答)", "error": "出错了: {e}",
        "too_long": "问题太长了 (超过 {n} 字), 请精简后再问.", "budget": "今日 API token 额度已用完, 请明天再来.",
        "working_title": "查阅《三国志》原文", "working": "正在检索和核对原文……",
    },
    "en": {
        "search": "Search: {q}", "read": "Read: {id}", "who": "Who: {name}", "era": "Era year: {era} {year}",
        "start_game": "Start game: {game}", "end_game": "End game", "tool_error": " (error: {e})", "chunks": " → {n} passages",
        "nudge": "Not enough material, searching again", "nudge_notfound": "The answer found nothing, searching with other words",
        "retry": "The answer was incomplete, answering again", "review": "Checking the answer against the sources",
        "quote_fix": "A quotation does not match the source, fixing it", "lang_fix": "Rewriting in the language of the question",
        "thinking": "· Thinking…", "steps": "{title} ({n} steps)", "search_title": "Search steps", "think_title": "Thinking",
        "sources": "Sources ({n})", "trans": "Modern Chinese", "check": "Quotation check", "ok": " ✓",
        "missing": " ✗ not found in the source", "bad_number": " ✗ no such number",
        "misplaced": " ✗ wrong source, the text is in ", "bad_nums": "Numbers in the answer that do not exist: ",
        "draft": "*(Draft: being checked against the sources sentence by sentence; it will update when done)*",
        "no_answer": "(no answer)", "error": "Something went wrong: {e}",
        "too_long": "The question is too long (over {n} characters); please shorten it.",
        "budget": "Today's API token quota has been used up; please come back tomorrow.",
        "working_title": "Consulting the Sanguozhi", "working": "Searching and checking the sources…",
    },
}

# 界面上固定的文字: build_ui 里用 gr.I18n, 按浏览器语言显示 (没有对应语言时用英文). 聊天框的 placeholder 不支持 I18n, 写成双语
UI_TEXT: dict[str, dict[str, str]] = {
    "zh": {"title": "三国志问答", "ask": "问一个三国历史问题, 或者回答游戏里的问题", "clear": "清除对话", "examples": "示例问题",
           "guess": "猜人物", "soup": "海龟汤", "hint": "提示", "give_up": "放弃 / 揭晓", "quit": "退出游戏"},
    "en": {"title": "Sanguozhi Q&A", "ask": "Ask about Three Kingdoms history, or answer the game's question", "clear": "Clear chat",
           "examples": "Example questions", "guess": "Guess the Person", "soup": "Turtle Soup", "hint": "Hint",
           "give_up": "Give up / reveal", "quit": "Quit game"},
}
CHAT_PLACEHOLDER = ("依据《三国志》(陈寿正文和裴松之注) 回答三国历史问题, 中文或英文提问都可以; 点下面的按钮可以玩猜人物或海龟汤 (游戏只有中文)."
                    "\n\nAnswers questions on Three Kingdoms history from the *Sanguozhi* (Chen Shou's text and Pei Songzhi's"
                    " annotations), in English or Chinese. The buttons below start the games (Chinese only).")


def ui_locales() -> dict[str, dict[str, str]]:
    """gr.I18n 要的 {语言代码: {键: 文字}}: 浏览器报的中文代码可能带地区 (zh-CN, zh-TW, zh-HK), 都用中文那一份."""
    return {"en": UI_TEXT["en"], **{code: UI_TEXT["zh"] for code in ("zh", "zh-CN", "zh-TW", "zh-HK")}}


def lang_of(text: str) -> str:
    """这一问用的语言: 有汉字算中文 "zh", 否则英文 "en". 运行时的提示跟着它走 (同 agent 的 LANG_NOTE: 用用户的语言作答)."""
    return "zh" if CJK.search(text or "") else "en"


def t(lang: str, key: str, **kw: Any) -> str:
    """MSG 里 lang 语言的 key, 用 kw 填空; 不认识的语言用中文."""
    return MSG.get(lang, MSG["zh"])[key].format(**kw)


def text_of(content: Any) -> str:
    """Gradio 消息的 content -> 纯文字. 字符串原样返回; 列表取其中 type == "text" 的部分拼起来; 其他 (文件等) 返回 ""."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def to_agent_history(history: list[dict]) -> list[dict]:
    """页面上的 history -> run_agent(history=) 要的消息列表: 只留用户问题和最终回答, 去掉显示用的折叠块.

    输入 (Gradio 回传的 history, 按页面顺序; tests/test_chat 的例子):
        [{"role": "user", "content": "曹操去过浚仪吗"},
         {"role": "assistant", "content": "· 搜索: 浚儀", "metadata": {"title": "检索过程 (1 步)", "status": "done"}},  <- 折叠块, 去掉
         {"role": "assistant", "content": "去过 [1]"},                                                                <- 最终回答, 留 (去掉编号)
         {"role": "assistant", "content": "[1] ...", "metadata": {"title": "出处 (1 段)"}},                          <- 折叠块, 去掉
         {"role": "user", "content": [{"type": "text", "text": "后来呢"}]},                                          <- content 是列表
         {"role": "assistant", "content": [{"type": "text", "text": "进军官渡"}], "metadata": None},                 <- metadata 为 None, 留
         {"role": "assistant", "content": ""}]                                                                        <- 空, 去掉
    输出:
        [{"role": "user", "content": "曹操去过浚仪吗"}, {"role": "assistant", "content": "去过"},
         {"role": "user", "content": "后来呢"}, {"role": "assistant", "content": "进军官渡"}]

    按页面顺序逐条处理:
        - role 不是 "user" / "assistant" 的丢掉
        - metadata 里 "title" 非空的 (折叠块) 丢掉; 没有 metadata, metadata 为 None 或 {} 都算普通消息
        - content 用 text_of 转成字符串并去掉首尾空白, 为空的丢掉; 回答里的资料编号 [n] 去掉 (编号只对那一次检索有效,
          留着会让模型以为要对上这次的编号, 甚至解释 "上一条的编号来自另一次检索")
        - 输出每条只含 "role" 和 "content" (不带 metadata)
    不截取轮数: 只带最近几轮由 run_agent 做 (agent.HISTORY_TURNS). 空列表返回 []."""
    out = []
    for m in history:
        meta = m.get("metadata") or {}
        if m.get("role") not in ("user", "assistant") or meta.get("title"):
            continue
        text = text_of(m.get("content")).strip()
        if m["role"] == "assistant":
            text = CITE.sub("", text).strip()
        if text:
            out.append({"role": m["role"], "content": text})
    return out


def describe(kind: str, info: Any, lang: str = "zh") -> str | None:
    """on_event 的一条事件 -> 检索过程里的一行文字; 不需要显示的 (如 "call") 返回 None.

    kind / info 的含义见 agent.run_agent 的 docstring. 例:
        ("tool", {"tool": "search", "args": {"query": "浚儀"}, "chunks": ["01-0002"], "error": None}) -> "· 搜索: 浚儀 → 1 段"
        ("stage", "review") -> "· 核对回答 (逐句对照资料)"
    工具参数不是 JSON 对象时 (模型偶尔给 [1] 之类), 参数部分留空, 只显示工具名和出错信息, 不抛异常.
    lang 是这一问的语言 (lang_of), 文字取 MSG[lang]."""
    if kind == "tool":
        a = info.get("args")
        a = a if isinstance(a, dict) else {}  # 模型给的参数不是 JSON 对象 (如 [1]) 时 run_tool 已记错, 这里参数部分留空
        what = {"search": lambda: t(lang, "search", q=a.get("query", "")),
                "read": lambda: t(lang, "read", id=a.get("chunk_id", "")),
                "who": lambda: t(lang, "who", name=a.get("name", "")),
                "era": lambda: t(lang, "era", era=a.get("era", ""), year=a.get("year", "")),
                "start_game": lambda: t(lang, "start_game", game=a.get("game", "")),
                "end_game": lambda: t(lang, "end_game")}.get(info.get("tool"), lambda: f"{info.get('tool')}")()
        if info.get("error"):
            return f"· {what}" + t(lang, "tool_error", e=info["error"])
        n = len(info.get("chunks") or [])
        return f"· {what}" + (t(lang, "chunks", n=n) if n else "")
    if kind == "think":  # 模型的推理文字 (游戏 skill 里 Agent 来猜时); 压成一段, 太长截断
        text = " ".join(str(info).split())
        return "💭 " + (text if len(text) <= 600 else text[:600] + "……")
    if kind == "stage":
        return "· " + (t(lang, info) if info in ("nudge", "nudge_notfound", "retry", "review", "quote_fix", "lang_fix") else str(info))
    return None


def render_sources(res: dict, max_len: int = 300, lang: str = "zh") -> str:
    """回答里引用到的 [n] -> 出处列表 (Markdown), 各段之间用 "---" 分隔; 没有可显示的内容时返回 "".

    res 是 agent.run_agent 的返回值 (只用到这几个键):
        {"reply": "至浚仪 [1], 又 [9]",
         "chunks": [{"id": "01-0002", "text": "...", "trans": "...", ...}, ...],   # 第 n 块对应编号 [n]
         "cites": [{"quote": "公軍譙", "cites": [1], "status": "ok", "found": [1]}, ...]}
    每个编号一段: 出处行, 原文和白话译文 (各超过 max_len 字截断). 只列回答里出现过的编号, 按编号排序.
    之后附引文核对结果: status 为 "ok" 标 ✓, "missing" 标原文里找不到, "bad_number" 标没有这个编号,
    其他 (出处标错) 标出文字实际所在的块 (found). 编号超出 chunks 范围 (模型编错) 的不列出处, 最后单独标出.
    说明文字按 lang (这一问的语言) 取 MSG; 出处行 (source_label) 和原文、白话照原样."""
    chunks = res.get("chunks") or []
    nums = sorted({int(n) for n in CITE.findall(res.get("reply") or "") if 0 < int(n) <= len(chunks)})
    parts = []
    for n in nums:
        c = chunks[n - 1]
        text = c["text"] if len(c["text"]) <= max_len else c["text"][:max_len] + "……"
        part = f"**[{n}]** {source_label(c)}\n\n> {text}"
        if c.get("trans"):
            trans = c["trans"] if len(c["trans"]) <= max_len else c["trans"][:max_len] + "……"
            part += f"\n\n{t(lang, 'trans')}: {trans}"
        parts.append(part)
    cites = res.get("cites") or []
    mark = {k: t(lang, k) for k in ("ok", "missing", "bad_number")}
    if cites:
        parts.append(t(lang, "check") + ": " + "; ".join(
            f"「{c['quote']}」{''.join(f'[{n}]' for n in c['cites'])}"
            + mark.get(c["status"], t(lang, "misplaced") + "".join(f"[{n}]" for n in c["found"])) for c in cites))
    bad = sorted({int(n) for n in CITE.findall(res.get("reply") or "")} - set(range(1, len(chunks) + 1)))
    if bad:
        parts.append(t(lang, "bad_nums") + " ".join(f"[{n}]" for n in bad))
    return "\n\n---\n\n".join(parts)


MAX_QUESTION_CHARS = 500  # 一个问题最长多少字


class Budget:
    """每天的花费上限 (美元), 按接口返回的实际花费累计 (rag/calls.make_complete 的 stats["cost"]: OpenRouter 和
    Claude 原生接口都有); 日期变了清零, 进程重启也清零. limit 为 None 时不限.
    多个会话共用一个 Budget, 加锁累加; 问答和游戏的请求都记进来. 每次答完才记账, 几个问题同时在答时可能略超上限."""

    def __init__(self, limit: float | None) -> None:
        self.limit, self.day, self.spent = limit, date.today(), 0.0
        self.lock = threading.Lock()

    def _roll(self) -> None:
        if date.today() != self.day:
            self.day, self.spent = date.today(), 0.0

    def exhausted(self) -> bool:
        with self.lock:
            self._roll()
            return self.limit is not None and self.spent >= self.limit

    def add(self, cost: float) -> None:
        with self.lock:
            self._roll()
            self.spent += cost


def check_limits(question: str, history: list[dict], budget: Budget | None) -> str | None:
    """超出限制时返回给用户看的说明, 没超出返回 None: 问题太长, 今天的预算用完. 对话轮数不设上限 (游戏的回合也在历史里,
    按轮数拦会误伤); 历史太长时由用户点 "清除对话". 说明用这一问的语言 (lang_of)."""
    lang = lang_of(question)
    if len(question) > MAX_QUESTION_CHARS:
        return t(lang, "too_long", n=MAX_QUESTION_CHARS)
    if budget is not None and budget.exhausted():
        return t(lang, "budget")
    return None


def make_respond(lib: agent.Library, model: str | None = None, effort: str | None = None,
                 complete_factory: Callable[[], Callable[..., Any]] | None = None, budget: Budget | None = None,
                 router: "Router | None" = None, show_steps: bool = True) -> Callable[..., Iterator[list[dict]]]:
    """返回 build_ui 用的 respond(message, history, sid).

    message 是字符串或 {"text": ...} (多模态输入框); history 是页面上的消息列表 (格式见模块 docstring).
    respond 是生成器, 每次 yield 本轮要显示的全部 assistant 消息 (list), Gradio 用它整体替换本轮显示:
        先 [检索过程 (pending)], 每多一行可显示的进度 (describe 不返回 None 的事件) 刷新一次;
        最后 [检索过程 (done), 最终回答, 出处 (有引用时)]; run_agent 抛异常时最后是 [检索过程, "出错了: ..."].
    回答为空时显示 "(没有得到回答)". 这些文字都跟着这一问的语言 (lang_of, 文字在 MSG), 上面写的是中文的样子.
    问题为空白时什么都不 yield.
    lib 是 agent.load_library() 的结果; model / effort 原样传给 run_agent, None 时用 .env 的 LLM_MODEL / LLM_REASONING_EFFORT.
    complete_factory() 每题调用一次, 返回交给 run_agent 的 complete (如 rag.calls.make_complete 带限流重试和计时);
    不给就用 rag.llm.complete.
    超出 check_limits 的限制时只 yield 一条说明, 不调模型. 用户点停止 (Gradio 关闭生成器), 或同一个 sid 发来新消息时,
    后台线程在下一次模型请求前抛 Cancelled 结束, 不再继续花钱; 被新消息打断的那一问不再刷新页面.
    budget 不为 None 时, 每题结束后把这题的花费 (complete.stats["cost"]) 记进去.
    router (rag/skills.Router) 不为 None 时: 正在玩游戏 (或是快捷命令) 的消息先交给它 (先 yield "思考过程" 折叠块, 最后是游戏的回复);
    游戏不管的消息走问答 agent, agent 的 system prompt 附上 router.agent_note, tools 加上 start_game / end_game,
    agent 调用它们时显示游戏的回复 (SkillReply). 预算用完时任何消息 (包括游戏的) 都只回一条说明; 游戏的消息不经过
    check_limits 的长度检查, 花费由 main() 传给 make_router 的 complete 记进 budget.
    sid 是 BrowserState 里存的浏览器 id, 区分各人的游戏.
    show_steps 为 False 时不列出检索的每一步 (公开的演示页用): 回答出来之前只显示一行 "正在检索和核对原文……" (带转圈),
    回答出来后这一行去掉, 只留回答和出处."""

    current: dict[str, threading.Event] = {}  # 每个浏览器 id 正在进行的那一问的取消标记
    current_lock = threading.Lock()

    def begin(sid: str) -> threading.Event:
        """登记新的一问; 同一个 id 上一问还没结束就取消它 (在回答过程中发新消息 = 打断)."""
        cancel = threading.Event()
        with current_lock:
            old, current[sid] = current.get(sid), cancel
        if old is not None:
            old.set()
        return cancel

    def say(text: str) -> dict:
        return {"role": "assistant", "content": text}

    def background(fn: Callable[[Callable[[str, Any], None]], Any], cancel: threading.Event, title: str,
                   box: dict, lang: str = "zh") -> Iterator[list[dict]]:
        """在后台线程跑 fn(emit), 进度 (describe 能显示的事件) 实时显示在名为 title 的折叠框里.
        结束后 box 里有 "value" (fn 的返回值) 或 "error" (异常), "head" 是要留在页面上的折叠框 (没有进度时为空列表).
        这一问被新消息取消时静静结束 (不再刷新页面, 页面交给新的一问); 页面关掉生成器 (点停止) 时设上 cancel."""
        events: queue.Queue = queue.Queue()

        def work() -> None:
            try:
                box["value"] = fn(lambda k, i: events.put((k, i)))
            except BaseException as e:  # noqa: BLE001  网络/服务商出错, 停止, SkillReply: 交给调用方
                box["error"] = e
            finally:
                events.put(("done", None))

        # 带上当前的 contextvars (Gradio 的请求上下文): 在 ZeroGPU 上, 后台线程里的 @spaces.GPU 函数要靠它知道是哪个访客的额度
        threading.Thread(target=contextvars.copy_context().run, args=(work,), daemon=True).start()
        lines: list[str] = []
        draft: list[dict] = []  # 自查前的初稿 (run_agent 的 "draft" 事件): 先显示, 最终回答出来后被整体换掉

        def progress(status: str) -> dict:
            if not show_steps:
                return {"role": "assistant", "content": t(lang, "working"),
                        "metadata": {"title": t(lang, "working_title"), "status": status}}
            return {"role": "assistant", "content": "\n".join(lines) or t(lang, "thinking"),
                    "metadata": {"title": t(lang, "steps", title=title, n=len(lines)), "status": status}}
        first = True
        try:
            while True:
                try:
                    kind, info = events.get(timeout=1.0)
                except queue.Empty:  # 一秒内没有事件: 第一次时先显示 "思考中", 免得页面空着
                    if cancel.is_set():
                        return
                    if first:
                        first = False
                        yield [progress("pending")]
                    continue
                if cancel.is_set():  # 被新消息打断: 不再刷新页面
                    return
                if kind == "done":
                    break
                if kind == "draft" and info:
                    draft[:] = [say(f"{info}\n\n{t(lang, 'draft')}")]
                    yield [progress("pending"), *draft]
                    continue
                line = describe(kind, info, lang)
                if line:
                    lines.append(line)
                    first = False
                    yield [progress("pending"), *draft]
        except GeneratorExit:  # 用户点了停止
            cancel.set()
            raise
        box["head"] = [progress("done")] if lines and show_steps else []

    def respond(message: Any, history: list[dict] | None, sid: str = "default") -> Iterator[list[dict]]:
        question = text_of(message.get("text") if isinstance(message, dict) else message).strip()
        if not question:
            return
        cancel = begin(sid)
        head: list[dict] = []
        lang = lang_of(question)
        if budget and budget.exhausted():
            yield head + [say(t(lang, "budget"))]
            return
        if router is not None and router.claims(sid, question):
            box: dict = {}
            yield from background(lambda emit: router.handle(sid, question, on_event=emit, cancel=cancel), cancel,
                                       t(lang, "think_title"), box, lang)
            if cancel.is_set():
                return
            head = box.get("head", [])
            if "error" in box:
                e = box["error"]
                yield head + [say(t(lang, "error", e=f"{type(e).__name__}: {e}"))]
                return
            if box["value"] is not None:
                yield head + [say(box["value"])]
                return
            # 游戏判断这句话不是游戏里的话: 交给问答 agent (system prompt 里带上正在玩的游戏)
        limit = check_limits(question, history or [], budget)
        if limit:
            yield head + [say(limit)]
            return
        base = complete_factory() if complete_factory else llm.complete
        extra: dict[str, Any] = {}
        if router is not None:
            extra = {"system": agent.system_for(model) + router.agent_note(sid), "tools": agent.TOOLS + router.tools(),
                     "handlers": router.handlers(sid)}

        def fn(emit: Callable[[str, Any], None]) -> dict:
            set_context(sid, "qa", None, cancel)
            trace({"type": "user", "text": question})
            t0 = time.time()
            try:
                res = agent.run_agent(question, lib, model=model, effort=effort,
                                      complete=lambda m, **kw: logged_request(base, m, **kw),
                                      history=to_agent_history(history or []), on_event=emit, **extra)
                trace({"type": "reply", "text": res["reply"], "seconds": round(time.time() - t0, 2)})
                return res
            finally:
                if budget is not None:
                    try:
                        budget.add(float(getattr(base, "stats", {}).get("cost", 0.0)))
                    except Exception:  # 记账出错不影响回答
                        pass
        box = {}
        yield from background(fn, cancel, t(lang, "search_title"), box, lang)
        if cancel.is_set():
            return
        head += box.get("head", [])
        if "error" in box:
            e = box["error"]
            if isinstance(e, SkillReply):  # agent 调了 start_game / end_game: 游戏的回复直接显示
                yield head + [say(e.text)]
            elif not isinstance(e, Cancelled):
                yield head + [say(t(lang, "error", e=f"{type(e).__name__}: {e}"))]
            return
        res = dict(box["value"])  # 改显示用的副本, 不动 run_agent 的返回值
        out = head + [say(res["reply"] or t(lang, "no_answer"))]
        sources = render_sources(res, lang=lang)
        if sources:
            n = len({m for m in CITE.findall(res["reply"]) if 0 < int(m) <= len(res.get("chunks") or [])})
            out.append({"role": "assistant", "content": sources, "metadata": {"title": t(lang, "sources", n=n), "status": "done"}})
        yield out

    return respond


def main() -> None:
    ap = argparse.ArgumentParser(description="三国志问答 网页聊天")
    ap.add_argument("--model", help="不给就用 .env 的 LLM_MODEL")
    ap.add_argument("--effort", help="reasoning_effort (none / low / medium / high), 覆盖 .env 的 LLM_REASONING_EFFORT")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Gradio 公网临时链接 (默认只在本机)")
    ap.add_argument("--lan", action="store_true", help="局域网里的其他设备也能访问 (监听 0.0.0.0, 不设密码)")
    ap.add_argument("--hide-steps", action="store_true", help="不列出检索的每一步, 回答出来前只显示一行 \"正在检索和核对原文\" (公开演示页用)")
    ap.add_argument("--daily-budget", type=float, default=None,
                    help="每天最多花多少美元 (按接口返回的花费累计, OpenRouter 和 Claude 原生接口都有; 问答和游戏都算), 用完后拒绝新消息; 默认不限")
    args = ap.parse_args()

    import gradio as gr  # ui extra; 在这里才导入, 所以测试不需要装 gradio

    from rag import calls
    model = calls.setup(args.model)  # 后端 (DeepSeek 经 OpenRouter / Claude 原生接口) 由 .env 的 LLM_BACKEND 决定; key 或模型不对时退出
    sem = threading.Semaphore(calls.PER_MODEL)
    factory = lambda: calls.make_complete(model, sem)  # noqa: E731  每题新建一个: 计时和记账按题
    print("加载检索索引……")
    lib = agent.load_library()
    from rag.skills import make_router
    budget = Budget(args.daily_budget)
    def _cal_cost(*a, **kw):
        c = factory()
        try:
            return c(*a, **kw)
        finally:
            budget.add(c.stats['cost'])
        
    router = make_router(lib, _cal_cost, model, args.effort)
    respond = make_respond(lib, model, args.effort, factory, budget, router, show_steps=not args.hide_steps)

    desc = f"`{model or config.LLM_MODEL}`" + (f" · effort {args.effort}" if args.effort else "")  # 不分语言
    demo = build_ui(respond, router, desc)
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    if args.lan:
        print(f"局域网访问: http://{lan_ip()}:{args.port}", flush=True)
    demo.launch(server_name=host, server_port=args.port, share=args.share, i18n=gr.I18n(**ui_locales()))


def lan_ip() -> str:
    """本机在局域网里的 IP (连一下外部地址看用的是哪块网卡, 不真的发数据); 拿不到时返回 "本机IP"."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.168.0.1", 80))
            return s.getsockname()[0]
    except OSError:
        return "本机IP"


GAME_BUTTONS = [("guess", "/猜人物"), ("soup", "/海龟汤"), ("hint", "/提示"), ("give_up", "/放弃"), ("quit", "/退出")]  # (UI_TEXT 的键, 发给游戏的命令)
IN_GAME = {"/提示", "/放弃", "/退出"}  # 只在游戏中可用的按钮


def build_ui(respond: Callable[..., Iterator[list[dict]]], router: "Router", description: str) -> Any:
    """搭网页 (gr.Blocks): 聊天框, 输入框 (带停止), 游戏按钮, 清除对话, 示例问题.

    对话历史和一个浏览器 id 存在 gr.BrowserState (浏览器 localStorage) 里: 刷新页面后对话还在,
    游戏状态按浏览器 id (而不是每次连接都变的 session_hash) 存在 router 里, 刷新后也接得上 (服务重启后游戏会丢).
    "清除对话" 清掉聊天记录和正在玩的游戏, 之后的问题不再带前文. 提示 / 放弃 / 退出 三个按钮只在游戏中可点."""
    import uuid

    import gradio as gr

    def ensure(uid: str) -> str:
        return uid or uuid.uuid4().hex

    def buttons(uid: str) -> list[Any]:
        active = router.active(uid) is not None
        return [gr.update(interactive=active or cmd not in IN_GAME) for _, cmd in GAME_BUTTONS]

    live: dict[str, list[dict]] = {}  # 每个浏览器 id 页面上正在显示的对话 (每次刷新显示时更新), 含被打断或停止的那一问

    def send(text: str, saved_history: list[dict] | None, uid: str) -> Iterator[tuple]:
        """前文取服务端记下的当前显示内容 (live), 没有时 (服务刚重启) 用浏览器里存的记录: 被打断或停止的那一问也留在页面和
        下一问的前文里. 不把聊天框本身当输入: Gradio 6 校验聊天框回传的消息格式, 字符串 content 会报错."""
        uid = ensure(uid)
        history = [dict(m, metadata={**m["metadata"], "status": "done"}) if (m.get("metadata") or {}).get("status") == "pending"
                   else m for m in (live.get(uid, saved_history) or [])]
        text = (text or "").strip()
        if not text:
            yield (history, "", history, uid, *buttons(uid))
            return
        shown = history + [{"role": "user", "content": text}]
        out: list[dict] = []
        live[uid] = shown
        yield (shown, "", history, uid, *buttons(uid))
        for out in respond(text, history, uid):
            live[uid] = shown + out
            yield (shown + out, "", history, uid, *buttons(uid))
        final = shown + out
        live[uid] = final
        yield (final, "", final, uid, *buttons(uid))

    def clear(uid: str) -> tuple:
        uid = ensure(uid)
        router.reset(uid)
        live[uid] = []
        return ([], [], uid, *buttons(uid))

    def load(saved: list[dict] | None, uid: str) -> tuple:
        uid = ensure(uid)
        return (saved or [], uid, *buttons(uid))

    i18n = gr.I18n(**ui_locales())  # 界面文字按浏览器语言显示; main() 把同一份传给 demo.launch(i18n=...)
    with gr.Blocks(title=i18n("title")) as demo:
        gr.Markdown(f"# {i18n('title')}\n{description}")
        saved = gr.BrowserState([], storage_key="sanguozhi_chat_history")
        uid = gr.BrowserState("", storage_key="sanguozhi_browser_id")
        chatbot = gr.Chatbot(height=560, placeholder=CHAT_PLACEHOLDER)  # placeholder 不支持 I18n, 写成双语
        box = gr.Textbox(placeholder=i18n("ask"), show_label=False,
                         submit_btn=True, stop_btn=True)
        with gr.Row():
            btns = [gr.Button(i18n(key), size="sm") for key, _ in GAME_BUTTONS]
            clear_btn = gr.Button(i18n("clear"), size="sm", variant="stop")
        gr.Examples(EXAMPLES, box, label=i18n("examples"))
        outs = [chatbot, box, saved, uid, *btns]
        # trigger_mode="multiple": 回答过程中也能发新消息, 新的一问会打断旧的 (make_respond 的 begin)
        events: list[Any] = [box.submit(send, [box, saved, uid], outs, concurrency_limit=8, trigger_mode="multiple")]
        for b, (_, cmd) in zip(btns, GAME_BUTTONS):
            events.append(b.click(lambda h, u, c=cmd: (yield from send(c, h, u)), [saved, uid], outs,
                                  concurrency_limit=8, trigger_mode="multiple"))
        box.stop(None, cancels=events)
        clear_btn.click(clear, [uid], [chatbot, saved, uid, *btns])
        chatbot.clear(clear, [uid], [chatbot, saved, uid, *btns])  # 聊天框自带的垃圾桶图标也同样清掉存下的记录和游戏
        demo.load(load, [saved, uid], [chatbot, uid, *btns])
    return demo


if __name__ == "__main__":
    main()
