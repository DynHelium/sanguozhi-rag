"""ingest/eras.py 的测试: 中文数字, YL 模板解析, 年号表的自洽性和换算.

    uv run python -m tests.test_eras

不依赖 pytest. 后两个用例读 data/eras.json, 需先跑 `uv run python -m ingest.eras`.
"""
from ingest import eras


def test_cn_number():
    cases = {"元": 1, "九": 9, "十": 10, "十二": 12, "二十": 20, "二十四": 24, "三十": 30}
    for s, n in cases.items():
        assert eras.cn_number(s) == n, s
    for bad in ["十十", "百", ""]:
        try:
            eras.cn_number(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} 应该抛 ValueError")


def test_parse_yl():
    assert eras.parse_yl("興平元年", "194年") == ("興平", 1, 194)
    assert eras.parse_yl("太和六年", "232年") == ("太和", 6, 232)
    assert eras.parse_yl("建安三年", None) == ("建安", 3, None)
    assert eras.parse_yl("皓以四年", None) is None, "NOT_ERAS 里的误标"
    assert eras.parse_yl("某某", None) is None


def test_era_to_ce_with_table():
    table = {"興平": {"start": 194, "status": "ok"}, "建興": {"start": None, "status": "conflict"}}
    assert eras.era_to_ce("興平", 2, table) == 195
    assert eras.era_to_ce("建興", 1, table) is None, "冲突的不给年份"
    assert eras.era_to_ce("不存在", 1, table) is None


def test_built_table_consistent():
    """生成的表 (含参考表合并): 各 status 的 start / 候选 / 参考条目要自洽."""
    t = eras.load()
    assert t["興平"]["status"] == "ok" and t["興平"]["start"] == 194
    for era, e in t.items():
        st = e["status"]
        if st == "ok":
            assert len(e["candidates"]) == 1 and e["start"] == e["candidates"][0]["start"], era
        elif st == "ref":
            assert not e["candidates"] and len(e["by_regime"]) == 1 and e["start"] in e["by_regime"].values(), era
        elif st == "多政权":
            assert e["start"] is None and len(e["by_regime"]) > 1, era
            assert {c["start"] for c in e["candidates"]} <= set(e["by_regime"].values()), era
        else:
            assert e["start"] is None, era
        for c in e["candidates"]:
            assert c["count"] == len(c["evidence"]) > 0, era
        for r in e.get("ref", []):
            assert r["source"].startswith("https://"), era


def test_era_to_ce_multi_regime():
    t = eras.load()
    if t["建興"]["status"] == "多政权":
        assert eras.era_to_ce("建興", 1, t) is None, "多政权不传 regime 不给"
        assert eras.era_to_ce("建興", 1, t, regime="蜀汉") == t["建興"]["by_regime"]["蜀汉"]


if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
