"""rag/alias.py 的测试: alias_index 的 keys 顺序, expand_query 的最长匹配, 同名不扩, 异体只认不追加.

    uv run python -m tests.test_alias

不依赖 pytest. 用测试里写的小别名表 (放进临时目录), 不读 data/aliases.json.
"""
import json
import tempfile
from pathlib import Path

from rag import alias, config

FAKE = {"people": [
    {"id": "01-0000", "name": "曹操", "surname": "曹", "zi": "孟德",
     "forms": [{"form": "魏太祖", "orig": "太祖", "src": "01-0001", "kind": "帝号"},
               {"form": "曹公", "orig": "公", "src": "01-0001", "kind": "称呼"},
               {"form": "魏王", "orig": "魏王", "src": "01-0090", "kind": "爵号"},
               {"form": "丞相", "orig": "丞相", "src": "01-0060", "kind": "官称"}]},
    {"id": "02-0000", "name": "曹丕", "surname": "曹", "zi": "子桓",
     "forms": [{"form": "魏文帝", "orig": "文帝", "src": "02-0001", "kind": "帝号"},
               {"form": "魏王", "orig": "魏王", "src": "02-0000", "kind": "爵号"}]},
    {"id": "03-0000", "name": "曹叡", "surname": "曹", "zi": "元仲", "forms": []},
    {"id": "43-0011", "name": "马忠", "surname": "马", "zi": "德信", "forms": []},
    {"id": "55-0049", "name": "马忠", "surname": "马", "zi": "", "forms": []},
    {"id": "28-0000", "name": "毌丘俭", "surname": "毌丘", "zi": "仲恭", "forms": []}],
    "variants": {"曹叡": ["曹睿"]}, "variant_surnames": {"毌丘": ["毋丘"]}}


def _with_table(table):
    """装饰器: 测试期间把 config.DATA_DIR 指到临时目录, 里面放 table 当 aliases.json."""
    return lambda fn: _with_fake(fn, table)


def _with_fake(fn, table=FAKE):
    def run():
        path = config.DATA_DIR / "aliases.json"
        old = path.read_text(encoding="utf-8") if path.exists() else None
        alias.alias_index.cache_clear(); alias.load_aliases.cache_clear()
        orig_dir = config.DATA_DIR
        with tempfile.TemporaryDirectory() as d:
            config.DATA_DIR = Path(d)
            (Path(d) / "aliases.json").write_text(json.dumps(table, ensure_ascii=False), encoding="utf-8")
            try:
                fn()
            finally:
                config.DATA_DIR = orig_dir
                alias.alias_index.cache_clear(); alias.load_aliases.cache_clear()
        assert old is None or path.read_text(encoding="utf-8") == old
    run.__name__ = fn.__name__
    return run


@_with_fake
def test_unique_name_expands():
    q = alias.expand_query("曹操因为什么逃离洛阳？")
    assert q.startswith("曹操因为什么逃离洛阳？ ")
    extra = q.split(" ", 1)[1].split()
    assert extra[:3] == ["曹孟德", "魏太祖", "曹公"], extra


@_with_fake
def test_generic_not_added_and_shared_title_not_key():
    q = alias.expand_query("曹操")
    assert "丞相" not in q, "泛称不追加"
    assert alias.expand_query("魏王是谁") == "魏王是谁", "魏王 对应两个人, 不展开"


@_with_fake
def test_same_name_not_expanded():
    assert alias.expand_query("三国志里有几个马忠？") == "三国志里有几个马忠？"


@_with_fake
def test_title_expands_to_name():
    q = alias.expand_query("魏太祖是谁")
    assert "曹操" in q and q.count("魏太祖") == 1, "已在查询里的不重复追加"


@_with_fake
def test_variants_are_keys_only():
    assert "曹叡" in alias.expand_query("曹睿字什么？")
    assert "毌丘俭" in alias.expand_query("毋丘俭是谁")
    assert "曹睿" not in alias.expand_query("曹叡是谁").split(" ", 1)[-1], "异体写法不追加"


@_with_fake
def test_longest_match_first():
    """"曹孟德" 含 "曹" 开头的别的名字也不会误配; 长的称呼先匹配, 匹配过的字不再参与."""
    q = alias.expand_query("曹孟德")
    assert "曹操" in q and "曹丕" not in q


@_with_fake
def test_keys_longest_first():
    """keys 按长度从长到短, expand_query 依赖这个顺序做最长匹配."""
    keys, _ = alias.alias_index()
    lens = [len(k) for k in keys]
    assert lens == sorted(lens, reverse=True), list(keys)


# 张鲁的 "原侯" 是曹植 "平原侯" 中间的一段 (data/aliases.json: 原侯 出处 08-0028, 平原侯 出处 19-0004).
OVERLAP = {"people": [
    {"id": "08-0024", "name": "张鲁", "surname": "张", "zi": "公祺",
     "forms": [{"form": "原侯", "orig": "原侯", "src": "08-0028", "kind": "帝号"}]},
    {"id": "19-0003", "name": "曹植", "surname": "曹", "zi": "子建",
     "forms": [{"form": "平原侯", "orig": "平原侯", "src": "19-0004", "kind": "爵号"}]}]}


@_with_table(OVERLAP)
def test_overlap_not_at_same_start():
    """短称呼落在长称呼中间 (起点不同) 也不能再单独匹配; 但它在别处单独出现时照常匹配."""
    q = alias.expand_query("平原侯是谁")
    assert "曹植" in q and "张鲁" not in q, q
    q = alias.expand_query("平原侯和原侯")
    assert "曹植" in q and "张鲁" in q, q


@_with_fake
def test_no_placeholder_join():
    """匹配过的字换成占位符而不是删掉: 删掉 "魏王" 的话 "曹" "丕" 会拼成 "曹丕"."""
    assert alias.expand_query("曹魏王丕") == "曹魏王丕"


def test_missing_table_is_noop():
    alias.alias_index.cache_clear(); alias.load_aliases.cache_clear()
    orig_dir = config.DATA_DIR
    with tempfile.TemporaryDirectory() as d:
        config.DATA_DIR = Path(d)
        try:
            assert alias.expand_query("曹操") == "曹操"
        finally:
            config.DATA_DIR = orig_dir
            alias.alias_index.cache_clear(); alias.load_aliases.cache_clear()


HOMONYMS = {"people": [
    {"id": "43-0011", "name": "马忠", "surname": "马", "zi": "德信", "forms": [{"form": "彭乡亭侯", "orig": "彭鄉亭侯", "src": "43-0011", "kind": "爵号"}]},
    {"id": "32-0000", "name": "刘备", "surname": "刘", "zi": "玄德", "forms": [{"form": "先主", "orig": "先主", "src": "32-0000", "kind": "称呼"}]}],
    "homonyms": {"马忠": [{"src": "47-0016", "desc": "孙权部将, 擒关羽"}]}, "variants": {}, "variant_surnames": {}}


@_with_table(HOMONYMS)
def test_homonym_name_not_expanded():
    """有表外同名的名字 (马忠, 见 homonyms) 不展开, 称号 (先主) 展开成本名."""
    assert alias.expand_query("有几个马忠") == "有几个马忠", "表外有同名, 不展开"
    assert "刘备" in alias.expand_query("先主是谁")


if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
