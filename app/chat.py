"""网页聊天: Gradio 薄包装. 检索和问答全在 rag/agent.py, 这里只管把过程和结果显示出来.

    uv sync --extra index --extra ui
    uv run python -m app.chat --openrouter --model deepseek-v4.1-flash --effort low   # DeepSeek V4.1 Flash 经 OpenRouter
    uv run python -m app.chat                    # 不加参数: 模型和接口地址取 .env 的 LLM_MODEL / LLM_BASE_URL
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
"""
import argparse
from datetime import date
import queue
import re
import threading
from collections.abc import Callable, Iterator
from typing import Any

from rag import agent, config, llm
from rag.quotes import source_label

CITE = re.compile(r"\[(\d+)\]")  # 回答里的资料编号 [n]
EXAMPLES = ["曹操因为什么逃离洛阳？", "昭烈皇帝有哪些儿子？", "三国志里有几个马忠？", "Which emperor was stabbed to death in the street?"]


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
         {"role": "assistant", "content": "去过 [1]"},                                                                <- 最终回答, 留
         {"role": "assistant", "content": "[1] ...", "metadata": {"title": "出处 (1 段)"}},                          <- 折叠块, 去掉
         {"role": "user", "content": [{"type": "text", "text": "后来呢"}]},                                          <- content 是列表
         {"role": "assistant", "content": [{"type": "text", "text": "进军官渡"}], "metadata": None},                 <- metadata 为 None, 留
         {"role": "assistant", "content": ""}]                                                                        <- 空, 去掉
    输出:
        [{"role": "user", "content": "曹操去过浚仪吗"}, {"role": "assistant", "content": "去过 [1]"},
         {"role": "user", "content": "后来呢"}, {"role": "assistant", "content": "进军官渡"}]

    按页面顺序逐条处理:
        - role 不是 "user" / "assistant" 的丢掉
        - metadata 里 "title" 非空的 (折叠块) 丢掉; 没有 metadata, metadata 为 None 或 {} 都算普通消息
        - content 用 text_of 转成字符串并去掉首尾空白, 为空的丢掉
        - 输出每条只含 "role" 和 "content" (不带 metadata)
    不截取轮数: 只带最近几轮由 run_agent 做 (agent.HISTORY_TURNS). 空列表返回 []."""
    out = []
    for m in history:
        meta = m.get("metadata") or {}
        if m.get("role") not in ("user", "assistant") or meta.get("title"):
            continue
        text = text_of(m.get("content")).strip()
        if text:
            out.append({"role": m["role"], "content": text})
    return out


def describe(kind: str, info: Any) -> str | None:
    """on_event 的一条事件 -> 检索过程里的一行文字; 不需要显示的 (如 "call") 返回 None.

    kind / info 的含义见 agent.run_agent 的 docstring. 例:
        ("tool", {"tool": "search", "args": {"query": "浚儀"}, "chunks": ["01-0002"], "error": None}) -> "· 搜索: 浚儀 → 1 段"
        ("stage", "review") -> "· 核对回答 (逐句对照资料)"
    工具参数不是 JSON 对象时 (模型偶尔给 [1] 之类), 参数部分留空, 只显示工具的中文名和出错信息, 不抛异常."""
    if kind == "tool":
        a = info.get("args")
        a = a if isinstance(a, dict) else {}  # 模型给的参数不是 JSON 对象 (如 [1]) 时 run_tool 已记错, 这里参数部分留空
        what = {"search": lambda: f"搜索: {a.get('query', '')}",
                "read": lambda: f"读原文: {a.get('chunk_id', '')}",
                "who": lambda: f"查人: {a.get('name', '')}",
                "era": lambda: f"年号换算: {a.get('era', '')}{a.get('year', '')}年"}.get(info.get("tool"),
                                                                                lambda: f"{info.get('tool')}")()
        if info.get("error"):
            return f"· {what} (出错: {info['error']})"
        n = len(info.get("chunks") or [])
        return f"· {what}" + (f" → {n} 段" if n else "")
    if kind == "stage":
        return {"nudge": "· 资料还不够, 再查一次", "nudge_notfound": "· 回答说没找到, 换说法再搜",
                "retry": "· 回答不完整, 重新作答", "review": "· 核对回答 (逐句对照资料)",
                "quote_fix": "· 有引文对不上原文, 修改引文"}.get(info, f"· {info}")
    return None


def render_sources(res: dict, max_len: int = 300) -> str:
    """回答里引用到的 [n] -> 出处列表 (Markdown), 各段之间用 "---" 分隔; 没有可显示的内容时返回 "".

    res 是 agent.run_agent 的返回值 (只用到这几个键):
        {"reply": "至浚仪 [1], 又 [9]",
         "chunks": [{"id": "01-0002", "text": "...", "trans": "...", ...}, ...],   # 第 n 块对应编号 [n]
         "cites": [{"quote": "公軍譙", "cites": [1], "status": "ok", "found": [1]}, ...]}
    每个编号一段: 出处行, 原文和白话译文 (各超过 max_len 字截断). 只列回答里出现过的编号, 按编号排序.
    之后附引文核对结果: status 为 "ok" 标 ✓, "missing" 标原文里找不到, "bad_number" 标没有这个编号,
    其他 (出处标错) 标出文字实际所在的块 (found). 编号超出 chunks 范围 (模型编错) 的不列出处, 最后单独标出."""
    chunks = res.get("chunks") or []
    nums = sorted({int(n) for n in CITE.findall(res.get("reply") or "") if 0 < int(n) <= len(chunks)})
    parts = []
    for n in nums:
        c = chunks[n - 1]
        text = c["text"] if len(c["text"]) <= max_len else c["text"][:max_len] + "……"
        part = f"**[{n}]** {source_label(c)}\n\n> {text}"
        if c.get("trans"):
            trans = c["trans"] if len(c["trans"]) <= max_len else c["trans"][:max_len] + "……"
            part += f"\n\n白话: {trans}"
        parts.append(part)
    cites = res.get("cites") or []
    mark = {"ok": " ✓", "missing": " ✗ 原文里找不到", "bad_number": " ✗ 没有这个编号"}
    if cites:
        parts.append("引文核对: " + "; ".join(
            f"「{c['quote']}」{''.join(f'[{n}]' for n in c['cites'])}"
            + mark.get(c["status"], " ✗ 出处标错, 文字在 " + "".join(f"[{n}]" for n in c["found"])) for c in cites))
    bad = sorted({int(n) for n in CITE.findall(res.get("reply") or "")} - set(range(1, len(chunks) + 1)))
    if bad:
        parts.append("回答里有不存在的编号: " + " ".join(f"[{n}]" for n in bad))
    return "\n\n---\n\n".join(parts)


MAX_QUESTION_CHARS = 500  # 一个问题最长多少字
MAX_TURNS = 20            # 一次对话最多问几轮 (页面上的用户消息数)


class Cancelled(Exception):
    """用户点了停止: 下一次模型请求前抛出, 结束这一题."""


class Budget:
    """每天的花费上限 (美元), 按 OpenRouter 返回的实际花费累计; 日期变了清零. limit 为 None 时不限.
    多个会话共用一个 Budget, 加锁累加. 不走 OpenRouter 时 complete 没有 stats (rag.llm.complete) 或 stats 里没有花费,
    每题的花费记为 0, 预算不起作用."""

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
    """超出限制时返回给用户看的说明, 没超出返回 None: 问题太长, 这次对话的轮数到上限, 今天的预算用完."""
    if len(question) > MAX_QUESTION_CHARS:
        return f"问题太长了 (超过 {MAX_QUESTION_CHARS} 字), 请精简后再问."
    if sum(1 for m in history if m.get("role") == "user") >= MAX_TURNS:
        return f"这次对话已经问了 {MAX_TURNS} 轮, 请点「清除」开始新的对话."
    if budget is not None and budget.exhausted():
        return "今天的额度已经用完, 请明天再来."
    return None


def make_respond(lib: agent.Library, model: str | None = None, effort: str | None = None,
                 complete_factory: Callable[[], Callable[..., Any]] | None = None, budget: Budget | None = None
                 ) -> Callable[[Any, list[dict] | None], Iterator[list[dict]]]:
    """返回给 gr.ChatInterface 的 respond(message, history).

    message 是字符串或 {"text": ...} (多模态输入框); history 是页面上的消息列表 (格式见模块 docstring).
    respond 是生成器, 每次 yield 本轮要显示的全部 assistant 消息 (list), Gradio 用它整体替换本轮显示:
        先 [检索过程 (pending)], 每多一行可显示的进度 (describe 不返回 None 的事件) 刷新一次;
        最后 [检索过程 (done), 最终回答, 出处 (有引用时)]; run_agent 抛异常时最后是 [检索过程, "出错了: ..."].
    回答为空时显示 "(没有得到回答)".
    问题为空白时什么都不 yield.
    lib 是 agent.load_library() 的结果; model / effort 原样传给 run_agent, None 时用 .env 的 LLM_MODEL / LLM_REASONING_EFFORT.
    complete_factory() 每题调用一次, 返回交给 run_agent 的 complete (如 rag.openrouter.make_complete 带限流重试和计时);
    不给就用 rag.llm.complete.
    超出 check_limits 的限制时只 yield 一条说明, 不调模型. 用户点停止 (Gradio 关闭生成器) 时, 后台线程在下一次模型请求前
    抛 Cancelled 结束, 不再继续花钱; budget 不为 None 时, 每题结束后把这题的花费 (complete.stats["cost"]) 记进去."""

    def respond(message: Any, history: list[dict] | None) -> Iterator[list[dict]]:
        question = text_of(message.get("text") if isinstance(message, dict) else message).strip()
        if not question:
            return
        limit = check_limits(question, history or [], budget)
        if limit:
            yield [{"role": "assistant", "content": limit}]
            return
        events: queue.Queue = queue.Queue()
        box: dict = {}
        cancel = threading.Event()
        base = complete_factory() if complete_factory else llm.complete

        def complete(*a: Any, **kw: Any) -> Any:
            if cancel.is_set():
                raise Cancelled()
            return base(*a, **kw)

        def work() -> None:
            try:
                box["res"] = agent.run_agent(question, lib, model=model, effort=effort, complete=complete,
                                             history=to_agent_history(history or []),
                                             on_event=lambda k, i: events.put((k, i)))
            except Cancelled:
                box["error"] = "已停止"
            except Exception as e:  # 网络/服务商出错: 显示出来, 不让页面卡住
                box["error"] = f"{type(e).__name__}: {e}"
            finally:
                try:
                    if budget is not None:
                        budget.add(float(getattr(base, "stats", {}).get("cost", 0.0)))
                finally:  # 记账出错也要发 done, 否则前台一直等
                    events.put(("done", None))

        threading.Thread(target=work, daemon=True).start()
        lines: list[str] = []

        def progress(status: str) -> dict:
            body = "\n".join(lines) or "· 思考中……"
            return {"role": "assistant", "content": body,
                    "metadata": {"title": f"检索过程 ({len(lines)} 步)", "status": status}}

        try:
            yield [progress("pending")]
            while True:
                kind, info = events.get()
                if kind == "done":
                    break
                line = describe(kind, info)
                if line:
                    lines.append(line)
                    yield [progress("pending")]
        except GeneratorExit:  # 用户点了停止
            cancel.set()
            raise
        if "error" in box:
            yield [progress("done"), {"role": "assistant", "content": f"出错了: {box['error']}"}]
            return
        res = dict(box["res"])  # 改显示用的副本, 不动 run_agent 的返回值
        out = [progress("done"), {"role": "assistant", "content": res["reply"] or "(没有得到回答)"}]
        sources = render_sources(res)
        if sources:
            n = len({m for m in CITE.findall(res["reply"]) if 0 < int(m) <= len(res.get("chunks") or [])})
            out.append({"role": "assistant", "content": sources, "metadata": {"title": f"出处 ({n} 段)", "status": "done"}})
        yield out

    return respond


def main() -> None:
    ap = argparse.ArgumentParser(description="三国志问答 网页聊天")
    ap.add_argument("--model", help="不给就用 .env 的 LLM_MODEL")
    ap.add_argument("--effort", help="reasoning_effort (none / low / medium / high), 覆盖 .env 的 LLM_REASONING_EFFORT")
    ap.add_argument("--openrouter", action="store_true", help="走 OpenRouter (key 见 rag/openrouter.use_openrouter), 只用 rag/openrouter.PROVIDERS 里给这个模型指定的服务商")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Gradio 公网临时链接 (默认只在本机)")
    ap.add_argument("--daily-budget", type=float, default=None,
                    help="每天最多花多少美元 (按 OpenRouter 返回的花费累计, 只在 --openrouter 时有数), 用完后拒绝新问题; 默认不限")
    args = ap.parse_args()

    import gradio as gr  # ui extra; 在这里才导入, 所以测试不需要装 gradio

    factory, model = None, args.model
    if args.openrouter:
        from rag.openrouter import PER_MODEL, full_name, make_complete, use_openrouter
        use_openrouter()
        model = full_name(model or config.LLM_MODEL)
        sem = threading.Semaphore(PER_MODEL)
        factory = lambda: make_complete(model, True, sem)  # noqa: E731
    if not (model or config.LLM_MODEL):
        raise SystemExit("没有模型: --model 或 .env 的 LLM_MODEL")
    print("加载检索索引……")
    lib = agent.load_library()
    demo = gr.ChatInterface(
        make_respond(lib, model, args.effort, factory, Budget(args.daily_budget)),
        chatbot=gr.Chatbot(height=640, placeholder="依据《三国志》(陈寿正文和裴松之注) 回答三国历史问题. 中文或英文提问都可以."),
        title="三国志问答", description=f"模型: {model or config.LLM_MODEL}" + (f" (思考 {args.effort})" if args.effort else ""),
        examples=EXAMPLES, run_examples_on_click=False, concurrency_limit=4,
    )
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
