"""把检索到的 chunk 拼成交给模型读的上下文 (format_context), 作为 tool 消息的 content.

eval/tool_calling.py (早期单工具实验) 用它. agent 用的是 rag/library.Seen.add: 同样的 "[编号] 出处行 + 原文" 格式,
另外跨多次工具调用沿用编号, 附上白话译文, 并提示这一段还有哪些裴注没显示.
"""
from rag.quotes import source_label
from rag.schema import ChunkDict


def format_context(chunks: list[ChunkDict]) -> str:
    """把检索到的 chunk 拼成交给模型读的一段文字 (作为 tool 消息的 content).

    输入: search 返回的 chunk 列表 (data/processed/chunks.jsonl 的行, rag/chunk.py 生成, 结构见 rag/schema.ChunkDict), 用到的字段:
        {"id": "01-0004n1", "tier": 2, "juan_title": "魏書•武帝紀", "headings": [], "cites": ["魏書", "世語"],
         "text": "《魏書》曰：太祖以卓終必覆敗，遂不就拜，逃歸鄉里。……"}

    输出 (真实数据, 传入 01-0004 和 01-0004n1 两块, 原文有删节):

        [1] 三国志·魏書•武帝紀 正文 (01-0004)
        金城邊章、韓遂殺刺史郡守以叛，……冬十二月，始起兵於己吾，是歲中平六年也。

        [2] 三国志·魏書•武帝紀 裴注引《魏書》《世語》 (01-0004n1)
        《魏書》曰：太祖以卓終必覆敗，遂不就拜，逃歸鄉里。……

    每块: 一行 "[编号] " + source_label(块) (rag/quotes.py, 出处和 chunk id 都在里面), 换行后接块的完整 text
    (原文自带的换行保留, 所以一块可能不止两行).
    编号从 1 起, 按传入顺序 (模型回答时用编号指代出处); 块与块之间空一行 ("\n\n" 连接).
    text 原样放, 不转简体: 模型引原文时要照抄, 引文核对 (verify_quotes) 比较时两边都会转简.
    chunks 为空时返回 ""; 不修改 chunks.
    """
    return "\n\n".join(f"[{i}] {source_label(c)}\n{c['text']}" for i, c in enumerate(chunks, start=1))
