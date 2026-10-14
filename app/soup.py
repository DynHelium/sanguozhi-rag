"""三国海龟汤网页 (Gradio): 选一道题, 看汤面, 问是 / 否问题, 卡住了要提示, 最后提交汤底或揭晓.

    uv run python -m app.soup --openrouter --model deepseek-v4.1-flash --effort low   # 打开 http://localhost:7861

游戏规则和裁判见 rag/soup.py. 每个会话的状态是服务端的一个 Game 对象 (含汤底和原文证据), 存在 gr.State 里,
gr.State 不下发到浏览器; 汤底只在揭晓或猜中时显示. 游戏逻辑在 Game 里, 不依赖 gradio, 测试直接调它 (tests/test_soup.py).
"""
import argparse
import threading
from typing import Any, Callable

from rag import config, llm, soup

TIER_NAMES = {"common": "常见人物", "rare": "冷门人物", "obscure": "最冷门"}


class Game:
    """一局游戏: 题目, 已问的问题数, 用过的提示数, 是否结束. complete 签名同 rag.llm.complete."""

    def __init__(self, puzzle: soup.Puzzle, complete: Callable[..., Any], model: str | None = None,
                 effort: str | None = None) -> None:
        self.p, self.complete, self.model, self.effort = puzzle, complete, model, effort
        self.asked = self.hints = 0
        self.over = False

    def ask(self, question: str) -> str:
        """回答一个是非问题, 返回页面上显示的一行 (只有判词, 不含模型的其他文字)."""
        question = question.strip()
        if self.over:
            return "这一局已经结束, 换一道题吧."
        if not question:
            return ""
        if self.asked >= self.p["max_questions"]:
            return f"已经问了 {self.asked} 个问题, 用完了. 可以提交你的解释, 或者点揭晓."
        r = soup.judge_question(self.p, question, self.complete, self.model, self.effort)
        if r["verdict"] != "请问是非题":
            self.asked += 1
        return f"{r['verdict']}  ({self.asked}/{self.p['max_questions']})"

    def hint(self) -> str:
        """给下一条提示; 提示用完了就说没有了."""
        if self.hints >= len(self.p["hints"]):
            return "没有更多提示了."
        self.hints += 1
        return f"提示 {self.hints}: {self.p['hints'][self.hints - 1]}"

    def guess(self, text: str) -> str:
        """判玩家提交的汤底. 猜中就揭晓; 没猜中只说说到了几个要点, 不说缺哪个 (免得泄底)."""
        if self.over:
            return "这一局已经结束."
        r = soup.judge_solution(self.p, text, self.complete, self.model, self.effort)
        if r["solved"]:
            self.over = True
            return "猜中了!\n\n" + self.reveal_text()
        n, total = len(r["covered"]), len(self.p["facts"])
        return f"还差一点: 说到了 {n}/{total} 个要点" + ("; 有和真相矛盾的地方" if r["contradiction"] else "") + ". 继续问吧."

    def reveal(self) -> str:
        """放弃, 直接揭晓."""
        self.over = True
        return self.reveal_text()

    def reveal_text(self) -> str:
        """汤底 + 原文出处 (源 id 和逐字引文)."""
        ev = "\n".join(f"- {e['source_id']}: 「{e['quote']}」" for e in self.p["evidence"])
        return f"汤底: {self.p['truth']}\n\n《三国志》原文:\n{ev}"


def main() -> None:
    ap = argparse.ArgumentParser(description="三国海龟汤")
    ap.add_argument("--model", help="不给就用 .env 的 LLM_MODEL")
    ap.add_argument("--effort", help="reasoning_effort, 覆盖 .env 的 LLM_REASONING_EFFORT")
    ap.add_argument("--openrouter", action="store_true", help="走 OpenRouter (key 见 rag/openrouter.use_openrouter)")
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()

    import gradio as gr  # ui extra

    model, complete = args.model, llm.complete
    if args.openrouter:
        from rag.openrouter import PER_MODEL, full_name, make_complete, use_openrouter
        use_openrouter()
        model = full_name(model or config.LLM_MODEL)
        sem = threading.Semaphore(PER_MODEL)  # 所有玩家共用并发上限

        def complete(*a: Any, **kw: Any) -> Any:  # 每次请求新包一层: make_complete 按创建时刻计时, 不能全站共用一个
            return make_complete(model, True, sem)(*a, **kw)
    puzzles = {p["id"]: p for p in soup.load_puzzles()}
    choices = [(f"[{TIER_NAMES.get(p['tier'], p['tier'])}] {p['title']}", pid) for pid, p in puzzles.items()]

    def start(pid: str) -> tuple[Game, str, list]:
        g = Game(puzzles[pid], complete, model, args.effort)
        return g, f"**汤面**: {g.p['surface']}\n\n最多问 {g.p['max_questions']} 个是非问题.", []

    def on_ask(q: str, g: Game, chat: list) -> tuple[str, list]:
        return "", chat + [{"role": "user", "content": q}, {"role": "assistant", "content": g.ask(q)}]

    def on_hint(g: Game, chat: list) -> list:
        return chat + [{"role": "assistant", "content": g.hint()}]

    def on_guess(text: str, g: Game, chat: list) -> tuple[str, list]:
        return "", chat + [{"role": "user", "content": "我的解释: " + text}, {"role": "assistant", "content": g.guess(text)}]

    def on_reveal(g: Game, chat: list) -> list:
        return chat + [{"role": "assistant", "content": g.reveal()}]

    with gr.Blocks(title="三国海龟汤") as demo:
        gr.Markdown("# 三国海龟汤\n题目都来自《三国志》(正文和裴松之注). 只能问能用 是 / 不是 回答的问题.")
        pick = gr.Dropdown(choices, value=choices[0][1], label="选一道题")
        surface = gr.Markdown()
        chat = gr.Chatbot(height=420)
        state = gr.State()
        with gr.Row():
            q = gr.Textbox(placeholder="例: 这件事和老鼠有关吗?", label="问一个是非问题", scale=4)
            gr.Button("问").click(on_ask, [q, state, chat], [q, chat])
        with gr.Row():
            gr.Button("提示").click(on_hint, [state, chat], chat)
            gr.Button("揭晓").click(on_reveal, [state, chat], chat)
        guess = gr.Textbox(placeholder="把你认为的完整经过写出来", label="提交汤底")
        gr.Button("提交").click(on_guess, [guess, state, chat], [guess, chat])
        q.submit(on_ask, [q, state, chat], [q, chat])
        pick.change(start, pick, [state, surface, chat])
        demo.load(start, pick, [state, surface, chat])
    demo.launch(server_port=args.port)


if __name__ == "__main__":
    main()
