"""出处与引文: 给模型看的出处行 (source_label), 从回答里抽「」引文 (extract_quotes), 逐字核对引文 (verify_quotes),
把写成 chunk id 的出处换成编号 (renumber_cites), 核对引文后面标的编号是否指向含这段文字的块 (check_cites),
改正能确定的错编号 (fix_cites).
source_label 给 agent 的工具结果 (rag/library.Seen), rag/context.format_context 和网页聊天的出处列表 (app/chat.py) 共用.
引文核对与大模型无关, 是纯代码检查:
看回答里「」引用的原文能不能在模型看过的 chunk 里逐字找到.
"""
import re
from collections.abc import Mapping
from typing import Any, TypedDict

from rag.normalize import to_simplified


def source_label(chunk: Mapping[str, Any]) -> str:
    """出处一行, 如 "三国志·魏書•武帝紀 裴注引《魏書》《世語》 (01-0004n1)".

    chunk 用到的字段: id, juan_title, tier (1 = "正文", 其他 = "裴注"), 可选的 headings 和 cites (字符串列表, 缺失或为空时省略).
    卷名, 小标题和书名原样拼接 (原文是繁体就是繁体, 不做转换). 合传里的小标题接在卷名后, 帮模型分清是合传里哪个人."""
    where = "三国志·" + chunk["juan_title"] + "".join("·" + h for h in chunk.get("headings") or [])
    if chunk["tier"] == 1:
        layer = "正文"
    else:
        layer = "裴注" + ("引" + "".join(f"《{b}》" for b in chunk["cites"]) if chunk.get("cites") else "")
    return f"{where} {layer} ({chunk['id']})"


def extract_quotes(reply: str) -> list[str]:
    """从回答里抽出引用的原文片段, 按出现顺序, 去掉引号本身.

    只抽取「」包围的文本; 用这个函数的问答流程要在 system prompt 里约定用「」引原文.
    例: "原文：「太祖乃變易姓名，間行東歸」[1]，又「遂不就拜」" -> ["太祖乃變易姓名，間行東歸", "遂不就拜"]
    没有引文时返回 []. 嵌套引号 (原文里的对话 「…『…』…」) 只取最外层.
    只认「」: 原文里的对话用「」『』, 模型照抄时会带上内层引号; ASCII 双引号不分开闭, 没法处理嵌套.
    没闭合的「 (模型输出被截断) 丢掉.
    """
    quotes, depth, buf = [], 0, []
    for ch in reply:
        if ch == "「":
            depth += 1
            if depth == 1:
                buf = []
                continue
        elif ch == "」" and depth:
            depth -= 1
            if depth == 0:
                quotes.append("".join(buf))
                continue
        if depth:
            buf.append(ch)
    return quotes


def verify_quotes(quotes: list[str], chunks: list[dict]) -> list[tuple[str, str | None]]:
    """逐条检查引文能不能在 chunk 原文 (chunk["text"]) 里逐字找到.

    chunks 每项至少有 id 和 text. 返回 [(引文, 找到它的 chunk id 或 None), ...], 与 quotes 同序; 多个 chunk 都有时取第一个.
    比较方法: 两边先转简体 (rag.normalize.to_simplified), 再删掉空白和非文字字符 (正则 [\\s\\W_…*]+, 标点, 下划线,
    markdown 加粗的 *, 省略号都在内, 数字保留), 然后看引文是不是某块的连续子串. 所以繁简和标点不同不影响, 其他字不同就找不到.
      - 空引文, 或删完非文字字符后为空的, 返回 None (免得模型输出 「」 蒙混过关).
      - 模型常在引文两头加 ** 或把 "，" 写成 ",": 这些被删掉, 不影响.
      - 用 "……" 省掉中间一段的: 省略号被删掉, 两头拼起来通常不是原文的连续子串, 所以找不到 (那不是逐字引用).
    找不到只说明在传入的 chunk 里没有这段文字, 不一定是编造 (也可能是异体字, 见下).
    已知的误判 (待定, 记在这里):
      - 假 ✗: OpenCC t2s 不转部分异体字 (呉 衞 等), 模型写 "吴" 而原文是 "呉" 时找不到.
      - 假 ✓: 很短的引文 (如 「白衣」 两个字) 几乎总能找到, 等于没校验; 目前没有设最短长度.
    """
    keys = [(c["id"], _normalize_quote(c["text"])) for c in chunks]
    result = []
    for q in quotes:
        nq = _normalize_quote(q)
        result.append((q, next((cid for cid, text in keys if nq and nq in text), None)))
    return result


_CITED = re.compile(r"「((?:[^「」]|「[^「」]*」)*)」\s*((?:[\[［]\d+[\]］]\s*[,，、]?\s*)+)")  # 「引文」[3][5]
_NUM = re.compile(r"\d+")


class CiteCheck(TypedDict):
    """check_cites 的一项结果."""
    quote: str               # 引文 (不含「」)
    cites: list[int]         # 紧跟在引文后面的编号, 如 「……」[3][5] -> [3, 5]
    status: str              # "ok" | "wrong" | "missing" | "bad_number", 见 check_cites
    found: list[int]         # 实际含这段文字的块的编号 (可能为空)


def check_cites(reply: str, chunks: list[dict]) -> list[CiteCheck]:
    """核对 "「引文」[n]" 里的编号: 引文是不是出自编号 n 指的那一块.

    chunks 是模型看过的块, 按给模型看时的编号排列: chunks[n-1] 就是编号 [n] 的块. 只看紧跟着编号的引文;
    后面没有编号的引文不在结果里 (它的文字对不对由 verify_quotes 判断). 文字比较方法同 verify_quotes.
    每条引文的 status:
        "ok"          至少一个所标编号的块含这段文字
        "wrong"       看过的块里有这段文字 (found 列出是哪几块), 但所标的块里没有: 出处标错
        "missing"     看过的块里都没有这段文字, 所标编号都在范围内
        "bad_number"  看过的块里都没有这段文字, 且所标编号里有超出范围的 (没有这一块)
    verify_quotes 只回答 "原文里有没有这段文字"; 这里另外回答 "标的出处对不对"."""
    keys = [_normalize_quote(c["text"]) for c in chunks]
    out: list[CiteCheck] = []
    for m in _CITED.finditer(reply):
        quote, cites = m.group(1), [int(x) for x in _NUM.findall(m.group(2))]
        nq = _normalize_quote(quote)
        found = [i + 1 for i, text in enumerate(keys) if nq and nq in text]
        if any(n in found for n in cites):
            status = "ok"
        elif found:
            status = "wrong"
        elif any(not 0 < n <= len(chunks) for n in cites):
            status = "bad_number"
        else:
            status = "missing"
        out.append({"quote": quote, "cites": cites, "status": status, "found": found})
    return out


def fix_cites(reply: str, chunks: list[dict]) -> tuple[str, int]:
    """改正能确定的错编号: 引文后只标了一个编号, 文字不在那一块, 而在看过的块里恰好只有一块含它 -> 换成那一块的编号.
    其他情况 (标了几个编号, 文字在多块里, 或找不到) 不动. 返回 (改后的回答, 改了几处). 判断方法同 check_cites."""
    keys = [_normalize_quote(c["text"]) for c in chunks]
    fixed = 0

    def sub(m: re.Match) -> str:
        nonlocal fixed
        cites = [int(x) for x in _NUM.findall(m.group(2))]
        nq = _normalize_quote(m.group(1))
        found = [i + 1 for i, text in enumerate(keys) if nq and nq in text]
        if len(cites) != 1 or cites[0] in found or len(found) != 1:
            return m.group(0)
        fixed += 1
        return m.group(0)[: m.start(2) - m.start(0)] + _NUM.sub(str(found[0]), m.group(2), count=1)
    return _CITED.sub(sub, reply), fixed


# chunk id: 01-0004 (正文), 01-0004n1 (裴注); 长段切开时再加 .0 .1 (rag/chunk.py). CITE_ID 认括号里的一个或几个 id
_ID = r"\d{2}-\d{4}(?:n\d+)?(?:\.\d+)?"
CITE_ID = re.compile(rf"[\[(（]\s*({_ID}(?:\s*[,，、]\s*{_ID})*)\s*[\])）]")


def renumber_cites(reply: str, chunks: list[dict]) -> str:
    """把模型写成 chunk id 的出处标注换成资料编号: "「……」[41-0000]" -> "「……」[1]".

    给模型看的每块资料出处行是 "[1] 三国志·…… (41-0000)", 模型常把括号里的 id 当编号抄进回答.
    认的写法: [41-0000], (41-0000), （41-0000）, [41-0000, 41-0001] (换成 [1][2]). 编号 n 是这个块在 chunks
    (模型看过的块, 按编号顺序) 里的位置 + 1; 括号里只要有一个 id 不在 chunks 里 (模型编的), 整个括号原样保留.
    已经是 [n] 的不动.
    """
    num = {c["id"]: i for i, c in enumerate(chunks, start=1)}

    def sub(m: re.Match) -> str:
        ids = re.split(r"\s*[,，、]\s*", m.group(1))
        if not all(i in num for i in ids):
            return m.group(0)
        return "".join(f"[{num[i]}]" for i in ids)
    return CITE_ID.sub(sub, reply)


_PUNCT = re.compile(r"[\s\W_…*]+")  # 空白, 中英文标点 (\W 不含汉字), markdown 符号


def _normalize_quote(text: str) -> str:
    """繁转简后去掉所有非文字字符. 只用于比较, 不改变展示."""
    return _PUNCT.sub("", to_simplified(text))
