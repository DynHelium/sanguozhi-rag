"""rag/library.py 的测试: Library 的 read / search / era 查询, Seen 的展示文字. 不依赖 pytest, 不连网, 不调模型:

    uv run python -m tests.test_library

用几块手写的小数据 (CHUNKS, ERAS) 和按词查表的假检索器, 不读 data/ 下的真实文件.
"""
from rag import library

CHUNKS = [
    {"id": "01-0001", "tier": 1, "juan": 1, "juan_title": "魏書•武帝紀", "headings": [], "parent": None, "cites": [],
     "sources": ["01-0001"], "text": "六年夏四月，揚兵河上。"},
    {"id": "01-0002", "tier": 1, "juan": 1, "juan_title": "魏書•武帝紀", "headings": [], "parent": None, "cites": [],
     "sources": ["01-0002"], "text": "七年春正月，公軍譙。遂至浚儀，治睢陽渠。"},
    {"id": "01-0002n0", "tier": 2, "juan": 1, "juan_title": "魏書•武帝紀", "headings": [], "parent": "01-0002",
     "cites": ["魏書"], "sources": ["01-0002n0"], "text": "《魏書》曰：太祖至浚儀。"},
    {"id": "01-0003", "tier": 1, "juan": 1, "juan_title": "魏書•武帝紀", "headings": [], "parent": None, "cites": [],
     "sources": ["01-0003"], "text": "進軍官渡。"},
    {"id": "02-0000", "tier": 1, "juan": 2, "juan_title": "魏書•文帝紀", "headings": [], "parent": None, "cites": [],
     "sources": ["02-0000"], "text": "文皇帝諱丕。"},
]
ERAS = {"建安": {"status": "ref", "start": 196, "by_regime": {"东汉": 196}},
        "建興": {"status": "多政权", "start": None, "by_regime": {"蜀汉": 223, "孙吴": 252}},
        "黃初": {"status": "ref", "start": 220, "by_regime": {"曹魏": 220}},
        "某某": {"status": "待核", "start": None, "by_regime": {}}}
TABLE = {"浚儀": ["01-0002"], "官渡": ["01-0003"]}


def fake_search(query: str, k: int) -> list[dict]:
    """假检索器: 按 TABLE 查词, 返回对应的块."""
    return [c for c in CHUNKS if c["id"] in TABLE.get(query, [])][:k]


LIB = library.Library(CHUNKS, fake_search, ERAS)


def test_read_neighbors_same_juan():
    """前后窗口只取同一卷的块, after 超出本卷时不跨到下一卷."""
    got = [c["id"] for c in LIB.read("01-0002", before=1, after=5)]
    assert got == ["01-0001", "01-0002", "01-0002n0", "01-0003"], "只读同一卷, 不跨到卷 2"


def test_read_note_includes_parent():
    """读裴注块时, 它所注的正文块排在最前面."""
    got = [c["id"] for c in LIB.read("01-0002n0")]
    assert got[0] == "01-0002" and "01-0002n0" in got, "裴注要附上它注的正文"


def test_read_attaches_notes_of_body_chunks():
    """窗口里的正文块自动带上它的裴注, 不需要加大 after; 同一块不重复输出."""
    got = [c["id"] for c in LIB.read("01-0001", after=1)]
    assert got == ["01-0001", "01-0002", "01-0002n0"], "读到正文块时附上它的裴注 (不用把 after 加大)"
    got = [c["id"] for c in LIB.read("01-0002", before=1, after=5)]
    assert got.count("01-0002n0") == 1, "不重复"


def test_search_groups_split_paragraph_as_one_slot():
    """同一源段切出的两块只占一个名额: 返回 SEARCH_K 组, 共 SEARCH_K + 1 块."""
    many = [{"id": f"09-{i:04d}", "tier": 1, "juan": 9, "juan_title": "", "headings": [], "parent": None, "cites": [],
             "sources": [f"09-{i:04d}"], "text": str(i)} for i in range(30)]
    split = [{"id": "08-0001.0", "tier": 1, "juan": 8, "juan_title": "", "headings": [], "parent": None, "cites": [],
              "sources": ["08-0001"], "text": "a"}, {"id": "08-0001.1", "tier": 1, "juan": 8, "juan_title": "",
              "headings": [], "parent": None, "cites": [], "sources": ["08-0001"], "text": "b"}]
    lib = library.Library(split + many, lambda q, k, where=None: (split + many)[:k], ERAS)
    got = lib.search("x")
    assert len({c["sources"][0] for c in got}) == library.SEARCH_K, "按源段算名额"
    assert [c["id"] for c in got[:2]] == ["08-0001.0", "08-0001.1"] and len(got) == library.SEARCH_K + 1


def test_read_by_source_id():
    """用段落 id 读: 被切开的段落读所有片 (连同它的注), 并进别的块的段落读那一块; 块 id 只读这一片 (不带同段的其他片), 照常附上注; 都不是就抛 KeyError."""
    def mk(cid, sources, tier=1, parent=None):
        return {"id": cid, "tier": tier, "juan": 3, "juan_title": "魏書•明帝紀", "headings": [], "parent": parent,
                "cites": [], "sources": sources, "text": cid}
    chunks = [mk("03-0001.0", ["03-0001"]), mk("03-0001.1", ["03-0001"]), mk("03-0001n0", ["03-0001n0"], 2, "03-0001"),
              mk("03-0002", ["03-0002", "03-0003"]), mk("03-0004", ["03-0004"])]
    lib = library.Library(chunks, lambda q, k, where=None: [], ERAS)
    assert [c["id"] for c in lib.read("03-0001")] == ["03-0001.0", "03-0001.1", "03-0001n0"]
    assert [c["id"] for c in lib.read("03-0003")] == ["03-0002"]
    assert [c["id"] for c in lib.read("03-0003", after=1)] == ["03-0002", "03-0004"]
    assert [c["id"] for c in lib.read("03-0001.1")] == ["03-0001.1", "03-0001n0"], "块 id 只读这一片, 照常附上注"
    try:
        lib.read("99-0001")
    except KeyError:
        pass
    else:
        raise AssertionError("不存在的 id 要抛 KeyError")


def test_read_unknown_raises():
    """不存在的 chunk_id 抛 KeyError."""
    try:
        LIB.read("99-9999")
    except KeyError:
        return
    raise AssertionError("不存在的 chunk_id 要抛 KeyError")


def test_era_ok_and_numerals():
    """单一政权的年号: year 可以是 int, 数字字符串 ("7年") 或汉字 ("七", "元"); 年号写简体也能查到."""
    assert "202" in LIB.era("建安", 7)
    assert "202" in LIB.era("建安", "7年")
    assert "202" in LIB.era("建安", "七")
    assert "196" in LIB.era("建安", "元")
    assert "220" in LIB.era("黄初", 1), "简体写法也要认"


def test_era_multi_regime():
    """多政权年号: 不给政权时列出各政权的公元年, 给了政权只答那一个."""
    both = LIB.era("建興", 1)
    assert "223" in both and "252" in both, "不给政权时列出各政权"
    shu = LIB.era("建興", 1, "蜀汉")
    assert "223" in shu and "252" not in shu


def test_era_unknown():
    """表里没有的年号和待核的年号都返回说明文字, 不抛异常."""
    assert "202" not in LIB.era("不存在", 7) and LIB.era("某某", 1)


def test_seen_empty_batch():
    """search 查无结果时, 模型收到 "(没有结果)" 而不是空的 tool 消息."""
    assert library.Seen().add([]) == "(没有结果)"


def test_unseen_notes_are_listed():
    """正文段有没看过的注时, 展示文字列出注 id 并提示 read; 注已在同一批里时不提示."""
    seen = library.Seen(LIB.notes)
    text = seen.add([CHUNKS[1]])
    assert "01-0002n0" in text and "read" in text
    assert "此段还有裴注" not in seen.add([CHUNKS[1], CHUNKS[2]]), "注已在同一批里, 不再提示"


def test_search_scope():
    """book / part 交给检索器筛选 (简繁都认, 只看首字); 都不给时照旧只传 (query, k); 认不出的值抛 ValueError."""
    def mk(cid, book, tier):
        return {"id": cid, "tier": tier, "juan": 1, "book": book, "juan_title": "", "headings": [], "parent": None,
                "cites": [], "sources": [cid], "text": cid}
    chunks = [mk("01-0001", "魏書", 1), mk("47-0001", "吳書", 1), mk("47-0001n0", "吳書", 2)]

    def search(q, k, where=None):
        return [c for c in chunks if where is None or where(c)][:k]
    lib = library.Library(chunks, search, ERAS)
    assert [c["id"] for c in lib.search("x", book="吴书", part="裴注")] == ["47-0001n0"]
    assert [c["id"] for c in lib.search("x", book="吳")] == ["47-0001", "47-0001n0"]
    assert [c["id"] for c in lib.search("x", part="正文")] == ["01-0001", "47-0001"]
    assert len(LIB.search("浚儀")) == 1, "不给范围时只传 (query, k), 只收两个参数的检索器照样能用"
    for bad in ({"book": "晋书"}, {"part": "注"}, {"book": 1}):
        try:
            lib.search("x", **bad)
            raise AssertionError(f"{bad} 应该抛 ValueError")
        except ValueError:
            pass


if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
