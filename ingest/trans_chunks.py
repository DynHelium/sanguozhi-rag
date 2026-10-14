"""白话检索用的切块: 每个原文 chunk 配上它对应的白话译文, 检索时搜白话.

    uv run python -m ingest.trans_chunks            # -> data/processed/chunks-trans.jsonl
    uv run python -m tests.test_trans_chunks

为什么: 用户用白话提问, 原文是文言, 而且常用简称 ("太祖" "公" "王") 和不带年号的年份 ("七年");
译文是现代汉语, 用词和提问接近, 并把年份补成完整年号 ("七年" -> "建安七年"), 拿白话去检索命中率更高
(48 道检索题上, BM25 + 向量 + 别名扩展, 宽松口径, hit@10 从搜原文的 0.69 升到搜白话的 0.81; 见 docs/DESIGN.md "离线把全书译成白话" 一行).
给模型看的仍是原文 + 译文: 回答里的引文只认原文, 因为译文是模型翻译的, 难免有错, 只能帮助检索和理解.

输入 (都在 data/processed/):

    chunks.jsonl                                  rag/chunk.py 的切块, 每行一个 chunk
    └─ {"id": "01-0004", "sources": ["01-0004"],  一个 chunk 可能合并了几个源 id (短段合并) 或是长段的一部分
        "text": "卓表太祖爲驍騎校尉…", "search": "…", ...}
    sanguozhi.jsonl                               ingest/sanguozhi.py 的解析结果, 用来取原文段全文 (切分长段时用)
    translations/卷NN.jsonl                       白话译文, 每行一个源 id (正文段或裴注)
    └─ {"id": "01-0004", "kind": "正文", "text": "董卓上表推荐曹操（太祖）…", "stage": "raw|checked|fixed"}

输出 chunks-trans.jsonl: 与 chunks.jsonl 同样的行, 只改两处:

    {"id": "01-0004", ...原有字段...,
     "trans": "董卓上表推荐曹操（太祖）…",   这个 chunk 的源 id 的译文按顺序拼接 (去掉〔待核〕等批注)
     "search": "董卓上表推荐曹操（太祖）…"}   检索用的文本换成白话 (BM25 和 dense 都读 search 字段)

长段被切成几个 chunk 时 (如 01-0076 切成 01-0076.0 / .1), 每个 chunk 只配它那一部分的译文, 分两步:

1. 找这个 chunk 覆盖原文段的哪一段 (_portion). chunk 的 offset 字段不是这个, 它是裴注挂在正文里的位置,
   同一段切出的几块 offset 相同. 一个 chunk 可能是 "长段 A 的末尾 + 换行 + 下一段 B" (短段合并), 所以先把
   chunk 按换行分开, 取能整体在 A 里找到的连续几行, 用 str.find 定位; 不能用 chunk 开头的固定长度前缀去找,
   因为 A 的末尾不足这个长度时前缀会跨过换行进到 B, 在 A 里找不到.
2. 把原文区间映射到译文 (slice_translation). 译文按源 id 存, 没有逐句对齐, 但翻译基本逐句进行: 原文和
   译文的句末数 (。！？；, 连着的几个算一个) 相等的段落约占九成. 相等时按序号对齐: 切点前原文有 k 个句末,
   译文就切在第 k 个句末之后. 不相等时退回按比例近似切: 原文区间 [起点, 起点 + len) 占原文段的比例映射到
   译文的同一比例区间, 两端挪到最近的句末之后. 按比例切在原文和译文长短比例不均匀时会错开一两句, 所以只作后备.
   两种切法的切点都跳过紧跟句末的闭引号 (”’"」』), 否则引语结尾的 ！” 会把闭引号留给下一块开头.

不切的话每块都带整段译文 (最长 1.3 万字), 检索区分不开; bge-m3 编码时按最长样本补齐, 内存也会暴涨到几十 GB.
"""
import json
import re
import sys

from rag import config

PROCESSED = config.DATA_DIR / "processed"
CHUNKS = PROCESSED / "chunks.jsonl"
TRANS_DIR = PROCESSED / "translations"
OUT = PROCESSED / "chunks-trans.jsonl"

# 译者批注: 〔待核〕, 〔原文作"林"〕, 〔年号提示为建安……待核〕 等. 给模型和检索都去掉 (对人有用, 对模型是噪声:
# 小模型实测会把 "璋〔待核〕" 读成 "此人身份待定"). 不含这些关键词的 〔〕 (如原文带过来的校勘字 〔颺〕) 保留.
_NOTE = re.compile(r"〔[^〕]*(?:待核|原文|疑|提示|似|如此)[^〕]*〕")


def clean(text: str) -> str:
    """去掉译者批注, 其余不动."""
    return _NOTE.sub("", text)


def load_translations() -> dict[str, str]:
    """读 translations/卷*.jsonl 的全部译文, 返回 {源 id: 译文}, 如 {"01-0004": "董卓上表推荐曹操（太祖）…", ...}.
    只认 卷*.jsonl, 目录里的其他文件不读."""
    out = {}
    for path in sorted(TRANS_DIR.glob("卷*.jsonl")):
        with open(path, encoding="utf-8") as f:
            out.update({r["id"]: r["text"] for r in map(json.loads, f)})
    return out


_SENT_END = "。！？；"
_CLOSE_QUOTE = "”’\"」』"


def _skip_end(text: str, pos: int) -> int:
    """从 pos 起跳过连续的句末标点和闭引号, 返回第一个不是它们的位置. 用来让切点落在 "！”" 这类组合之后.
    半角 " 不分开闭, 译文里两种引号都有: 前面的半角 " 是奇数个时才算闭引号, 否则它是下一句的开引号, 留给下一块."""
    while pos < len(text) and text[pos] in _SENT_END + _CLOSE_QUOTE:
        if text[pos] == '"' and text.count('"', 0, pos) % 2 == 0:
            break
        pos += 1
    return pos


def _sentence_ends(text: str) -> list[int]:
    """text 里每个句末的切点位置 (句末标点及紧跟的闭引号之后), 连着的几个句末标点 (如 "？！") 算一个.
    例: _sentence_ends("甲曰：「善！」乙。") -> [7, 9]."""
    ends, i = [], 0
    while i < len(text):
        if text[i] in _SENT_END:
            i = _skip_end(text, i)
            ends.append(i)
        else:
            i += 1
    return ends


def _snap(text: str, pos: int, window: int = 60) -> int:
    """把切点 pos 挪到附近 (前后 window 字内) 最近的句末标点之后, 再跳过紧跟的句末标点和闭引号; 附近没有就不动.
    两端 (<= 0, >= len) 截到 [0, len]."""
    if pos <= 0 or pos >= len(text):
        return max(0, min(pos, len(text)))
    best = None
    for i in range(max(0, pos - window), min(len(text), pos + window)):
        if text[i] in _SENT_END and (best is None or abs(i + 1 - pos) < abs(best - pos)):
            best = i + 1
    return _skip_end(text, best) if best is not None else pos


def slice_translation(trans: str, offset: int, length: int, src_len: int, src: str | None = None) -> str:
    """原文段 (长 src_len) 的 [offset, offset + length) 部分 -> 译文里对应的部分.

    给了原文段 src 且原文和译文的句末数相等时按句末序号对齐: 切点前原文有 k 个句末, 译文就切在第 k 个句末之后
    (句末的算法见 _sentence_ends). 否则按比例切, 两端挪到最近的句末之后 (_snap). 理由见模块 docstring.

    例: slice_translation("第一句话。第二句话。第三句话。第四句话。", 50, 50, 100) -> "第三句话。第四句话。" (按比例)
        slice_translation("一一一一一一一一。二。三。", 0, 4, 6, "甲。乙。丙。") -> "一一一一一一一一。二。"
            (按序号; 按比例会切在第一句后, 漏掉 "二。")
    覆盖整段或 src_len <= 0 时返回整段译文; 切出来是空串时 (如切点都挪到了同一处) 也退回整段译文."""
    if src_len <= 0 or (offset <= 0 and offset + length >= src_len):
        return trans
    t_ends = _sentence_ends(trans)
    s_ends = _sentence_ends(src) if src is not None else []
    if s_ends and len(s_ends) == len(t_ends):
        def cut(pos: int) -> int:
            k = sum(e <= pos for e in s_ends)        # 原文切点前完整的句子数
            return t_ends[k - 1] if k else 0
        a, b = cut(offset), cut(offset + length)
        if offset + length >= src_len:
            b = len(trans)
    else:
        a = _snap(trans, round(offset / src_len * len(trans)))
        b = _snap(trans, round((offset + length) / src_len * len(trans)))
    return trans[a:b].strip() or trans


def _portion(ctext: str, src: str) -> tuple[int, int] | None:
    r"""chunk 文字 ctext 覆盖原文段 src 的哪一段, 返回 (起点, 长度); 整段都在 chunk 里或找不到时返回 None (= 用整段译文).

    先把 ctext 按换行分开 (短段合并时各源之间用换行连接), 在能整体在 src 里找到的连续几行中取最长的一组定位,
    如 "未至，權退。\n十四年，……" 对上一段只取 "未至，權退。". 连续几行按原样 (带换行) 或去掉换行拼起来找都算,
    因为切块有时在同一段里也插了换行 (如 08-0000.0 的最后一句前), 原文段里却没有. 都找不到时退回前缀查找: 依次用 30/15/8 字的
    前缀找 chunk 在 src 里的起点, 或 src 在 chunk 里的起点."""
    if src in ctext:
        return None
    lines = ctext.split("\n")
    best = ""
    for i in range(len(lines)):
        for j in range(i, len(lines)):
            found = [p for p in ("\n".join(lines[i:j + 1]), "".join(lines[i:j + 1])) if p and p in src]
            if not found:
                break
            if len(found[0]) > len(best):
                best = found[0]
    if best:
        return src.find(best), len(best)
    for n in (30, 15, 8):
        i = src.find(ctext[:n])
        if i >= 0:                       # chunk 从这段的中间开始
            return i, min(len(src) - i, len(ctext))
        j = ctext.find(src[:n])
        if j >= 0:                       # chunk 以别的源开头, 这段从 chunk 的 j 处开始, 一直到 chunk 末尾
            return 0, min(len(src), len(ctext) - j)
    return None


def attach(chunks: list[dict], trans: dict[str, str], src_text: dict[str, str] | None = None) -> tuple[list[dict], list[str]]:
    """给每个 chunk 加上 trans 并把 search 换成译文, 返回 (新 chunk 列表, 缺译文的源 id). 不改传入的 chunk.

    trans       {源 id: 译文} (load_translations 的结果); 译文先经 clean 去掉批注
    src_text    {源 id: 原文段全文}; 给了就对 "只有一部分在这个 chunk 里" 的源只取对应部分的译文 (见模块 docstring);
                没给就每个源都用整段译文
    一个 chunk 有几个源 id 时, 各源的译文按 sources 的顺序用换行拼接; 缺译文的源跳过并记进第二个返回值.
    所有源都缺译文时 trans 为空串, search 保留原文."""
    out, missing = [], []
    for c in chunks:
        parts = []
        for sid in c["sources"]:
            if sid not in trans:
                missing.append(sid)
                continue
            t = clean(trans[sid])
            src = src_text.get(sid) if src_text else None
            span = _portion(c["text"], src) if src is not None else None
            if src is not None and span:
                t = slice_translation(t, span[0], span[1], len(src), src)
            parts.append(t)
        t = "\n".join(parts)
        out.append({**c, "trans": t, "search": t or c["search"]})
    return out, missing


def main() -> None:
    """读 chunks.jsonl, sanguozhi.jsonl 和全部译文, 写 chunks-trans.jsonl; 缺译文的源 id 打印到 stderr."""
    with open(CHUNKS, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    src_text = {}
    with open(PROCESSED / "sanguozhi.jsonl", encoding="utf-8") as f:
        for p in map(json.loads, f):
            src_text[p["id"]] = p["text"]
            src_text.update({f"{p['id']}n{i}": n["text"] for i, n in enumerate(p["notes"])})
    new, missing = attach(chunks, load_translations(), src_text)
    with open(OUT, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in new)
    print(f"{len(new)} 个 chunk -> {OUT}; 缺译文的源 id {len(set(missing))} 个", file=sys.stderr)
    if missing:
        print("  例:", missing[:10], file=sys.stderr)


if __name__ == "__main__":
    main()
