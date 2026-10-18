"""人工判读: 在浏览器里逐条判 eval/run_agent.py 的回答 "对 / 部分 / 错".

    uv run python -m eval.judge --name <名>                              # 判 answers/<名>.jsonl, 全部模型
    uv run python -m eval.judge --name <名> --models deepseek-v4.1-flash  # 只判某些模型 (按结果里的 model 字段)
    uv run python -m eval.judge --name <名> --port 8800                  # 端口被占时换一个

启动后打开终端里打印的地址. 服务只监听 127.0.0.1, 只用标准库 (http.server); 页面是同目录的 judge.html.
页面: 左边题目, 答案依据 (eval/retrieval.jsonl 里这题的 note, 别的题目文件的题没有), gold 段落的原文;
右边模型回答, 「」引文按结果行里 quotes 的核对结果标绿 (原文里找到) / 标红 (找不到); 模型看过的块可以展开看.
键盘: 1 对, 2 部分, 3 错, 0 清除; ← → 上一条/下一条; u 跳到下一条没判的. 判完自动跳下一条.
顺序: 按题号, 同一题的各模型挨着 (读一次 gold, 连判几个模型).

结果写到 data/processed/eval_runs/answers/<name>.judged.jsonl, 每判一条整文件重写一次 (同一 (模型, 题) 只留最后一次):

    {"model": "deepseek-v4.1-flash", "qid": "r-49", "verdict": "对"}       verdict: "对" / "部分" / "错"
    {"model": "deepseek-v4.1-flash", "qid": "r-50", "verdict": "错", "note": "把曹奂当成了曹髦"}   note 可选

和 eval/ai_judge.py 写的是同一个文件, 格式兼容; 判完跑 eval/run_agent.py --summary 看 "gold 在不在上下文 x 回答对不对" 的 2x2.
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from eval import metrics
from eval.common import OUT_DIR
from eval.run_retrieval import CHUNKS_PATH, QUESTIONS_PATH, load_jsonl

HTML = Path(__file__).with_name("judge.html")
VERDICTS = ("对", "部分", "错")


def build_items(name: str, models: list[str] | None) -> tuple[list[dict], dict[str, str]]:
    """回答记录 + 题目 note + gold 原文 -> 页面要的条目列表 (按题号, 同题各模型挨着),
    以及条目里模型看过的块的原文 {chunk id: 文字}. gold 原文取自原文切块 (CHUNKS_PATH), 按 sources 拼起来."""
    rows = load_jsonl(OUT_DIR / f"{name}.jsonl")
    if models:
        rows = [r for r in rows if r["model"] in models]
    notes = {q["id"]: q.get("note", "") for q in load_jsonl(QUESTIONS_PATH)}
    chunks = load_jsonl(CHUNKS_PATH)
    by_source: dict[str, list[dict]] = {}
    for c in chunks:
        for s in c["sources"]:
            by_source.setdefault(s, []).append(c)
    by_id = {c["id"]: c for c in chunks}
    order = {m: i for i, m in enumerate(dict.fromkeys(r["model"] for r in rows))}
    rows.sort(key=lambda r: (r["qid"], order[r["model"]]))
    texts: dict[str, str] = {}
    items = []
    for r in rows:
        gold = []
        for g in r["gold"]:
            parts = by_source.get(g, [])
            gold.append({"id": g, "label": parts[0]["juan_title"] if parts else "",
                         "text": "".join(p["text"] for p in parts)})
        for cid in r["retrieved"]:
            texts[cid] = by_id[cid]["text"] if cid in by_id else ""
        items.append({"model": r["model"], "qid": r["qid"], "type": r["type"], "q": r["q"], "note": notes.get(r["qid"], ""),
                      "gold": gold, "gold_in_ctx": r["gold_in_ctx"], "retrieved": r["retrieved"],
                      "gold_chunks": [cid for cid in r["retrieved"] if cid in by_id and metrics.is_hit(by_id[cid], r["gold"])],
                      "reply": r["reply"], "quotes": r["quotes"], "refused": r["refused"], "sec": r["sec"],
                      "error": r.get("error")})
    return items, texts


class Store:
    """judged.jsonl 的读写. 键是 (model, qid); 写入时整文件重写, 加锁防并发请求交错."""

    def __init__(self, path: Path):
        self.path, self.lock = path, threading.Lock()
        self.data: dict[tuple[str, str], dict] = {}
        if path.exists():
            for j in load_jsonl(path):
                self.data[(j["model"], j["qid"])] = j

    def set(self, model: str, qid: str, verdict: str | None, note: str = "") -> None:
        with self.lock:
            if verdict is None:
                self.data.pop((model, qid), None)
            else:
                row = {"model": model, "qid": qid, "verdict": verdict}
                if note.strip():
                    row["note"] = note.strip()
                self.data[(model, qid)] = row
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in self.data.values()), encoding="utf-8")
            tmp.replace(self.path)


def make_handler(items: list[dict], texts: dict[str, str], store: Store, name: str) -> type[BaseHTTPRequestHandler]:
    """HTTP 接口: GET / 返回页面, GET /api/data 返回条目和已有判读, POST /api/judge 写一条判读
    ({"model", "qid", "verdict": "对" / "部分" / "错" / null (清除), "note"})."""
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")  # 每次重读, 改页面不用重启
            elif self.path == "/api/data":
                judged = {f"{m}|{q}": v for (m, q), v in store.data.items()}
                self._json({"name": name, "items": items, "texts": texts, "judged": judged, "verdicts": VERDICTS})
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if self.path != "/api/judge":
                return self._send(404, b"not found", "text/plain")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                verdict = body.get("verdict")
                if verdict is not None and verdict not in VERDICTS:
                    return self._json({"error": f"verdict 只能是 {VERDICTS} 或 null"}, 400)
                store.set(body["model"], body["qid"], verdict, body.get("note", ""))
                self._json({"ok": True, "count": len(store.data)})
            except (KeyError, ValueError) as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 400)

        def log_message(self, format: str, *args: Any) -> None:  # 不刷屏
            pass
    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="在浏览器里判读回答评测结果")
    ap.add_argument("--name", default="agent", help=f"{OUT_DIR}/<name>.jsonl")
    ap.add_argument("--models", nargs="+", help="只判这些模型 (默认全部)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    src = OUT_DIR / f"{args.name}.jsonl"
    if not src.exists():
        raise SystemExit(f"找不到 {src}; 先跑 uv run python -m eval.run_agent --models deepseek-v4.1-flash --effort low --name {args.name}")
    items, texts = build_items(args.name, args.models)
    store = Store(OUT_DIR / f"{args.name}.judged.jsonl")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(items, texts, store, args.name))
    print(f"{len(items)} 条回答, 已判 {len(store.data)} 条. 打开 http://localhost:{args.port}  (Ctrl+C 退出)")
    print(f"判读结果: {store.path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n已判 {len(store.data)} 条. 看汇总: uv run python -m eval.run_agent --summary --name {args.name}")


if __name__ == "__main__":
    main()
