"""文本归一化: 繁->简. 入库和查询走同一套 (改了要重建索引). 查询侧的别名扩展在 rag/alias.py."""
from opencc import OpenCC

_t2s = OpenCC("t2s")


def to_simplified(text: str) -> str:
    """繁体转简体 (OpenCC t2s). 逐字转换, 长度不变. 不处理异体字 (如 "呉" "衞")."""
    return _t2s.convert(text)
