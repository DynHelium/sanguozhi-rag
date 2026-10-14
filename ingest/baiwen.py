"""生成白文 (不带标点的原文): data/processed/sanguozhi.jsonl -> data/sources/sanguozhi/baiwen.jsonl.

    uv run python -m ingest.baiwen

为什么: 公开仓库只放白文. 维基文库的录入按中华书局点校本断句标点, 点校标点在部分法域可能受保护;
陈寿正文和裴注本身是公有领域. 要跑检索和问答, 用 `uv run python -m ingest.use_wikisource` 按固定 revid 下载
维基文库的标点版, 解析, 切块 (切块按句号断句, 白文不能直接用). rag/ 和 eval/ 不读 baiwen.jsonl,
它只是仓库里可以直接查看的原文副本.

做法: 每段正文和每条裴注只留汉字 (含扩展区汉字) 和表示缺字的 □, 去掉标点, 书名号, 分节符号 (◎ ○ ◇) 和换行.
用字是解析时采用的读法 (点校本改定的字); 不带 variants (校勘记录), offset (依赖标点的位置) 和 search 字段.
段落划分和源 id 与 sanguozhi.jsonl 一致, 所以检索题, 别名表的出处 id 在白文里也能对上.

输出一行一段 (和 sanguozhi.jsonl 同序):

    {"id": "01-0000", "juan": 1, "book": "魏書", "juan_title": "魏書•武帝紀", "headings": [],
     "text": "太祖武皇帝沛國譙人也姓曹諱操字孟德漢相國參之後……",
     "notes": [{"id": "01-0000n0", "cites": ["曹瞞傳", "魏書"], "text": "曹瞞傳曰太祖一名吉利小字阿瞞……"}]}
"""
import json
import re
from typing import TypedDict

from rag import config
from rag.schema import Paragraph

SRC = config.DATA_DIR / "processed" / "sanguozhi.jsonl"
OUT = config.DATA_DIR / "sources" / "sanguozhi" / "baiwen.jsonl"
_NOT_TEXT = re.compile(r"[^㐀-䶿一-鿿豈-﫿\U00020000-\U0003134f□]")


class BaiwenNote(TypedDict):
    """白文里的一条裴注."""
    id: str           # 裴注的源 id, 如 "01-0000n0"
    cites: list[str]  # 同 rag/schema.Note.cites
    text: str         # 只留汉字和 □


class BaiwenParagraph(TypedDict):
    """baiwen.jsonl 的一行: 去掉标点的段落. 字段同 rag/schema.Paragraph, 没有 search, variants 和注的 offset."""
    id: str
    juan: int
    book: str
    juan_title: str
    headings: list[str]
    text: str
    notes: list[BaiwenNote]


def strip(text: str) -> str:
    """只留汉字和 □."""
    return _NOT_TEXT.sub("", text)


def to_baiwen(row: Paragraph) -> BaiwenParagraph:
    """一个段落记录 (rag/schema.Paragraph) -> 白文记录 (格式见模块 docstring); 裴注带上源 id."""
    return {"id": row["id"], "juan": row["juan"], "book": row["book"], "juan_title": row["juan_title"],
            "headings": row["headings"], "text": strip(row["text"]),
            "notes": [{"id": f"{row['id']}n{i}", "cites": n.get("cites", []), "text": strip(n["text"])}
                      for i, n in enumerate(row.get("notes", []))]}


def main() -> None:
    rows: list[Paragraph] = [json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines() if line.strip()]
    OUT.write_text("".join(json.dumps(to_baiwen(r), ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    n_notes = sum(len(r.get("notes", [])) for r in rows)
    print(f"{len(rows)} 段, {n_notes} 条裴注 -> {OUT}")


if __name__ == "__main__":
    main()
