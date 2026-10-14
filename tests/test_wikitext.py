"""ingest/wikitext.py 的测试: wikitext 切分 (scan, split_args, unwrap_links), 补丁 (apply_patches), 模板渲染 (render_template).

    uv run python -m tests.test_wikitext

不依赖 pytest. 短字符串用例覆盖边界情况 (嵌套模板, 单花括号, 相邻模板, 链接里的 |, 括号不配对).
真实文件用例读 data/sources/sanguozhi/raw/卷NN.wiki (先跑 ingest.use_wikisource 下载, 没有则跳过):
65 卷切开再拼回一字不差, 全书模板参数切分往返一致, 卷01 裴注条数 (122).
"""
import yaml

from tests._run import Skip
from rag import config
from ingest.wikitext import Rendered, Template, apply_patches, render_template, scan, split_args, unwrap_links

SGZ_DIR = config.DATA_DIR / "sources" / "sanguozhi"
RAW_DIR = SGZ_DIR / "raw"


def _load_patches():
    with open(SGZ_DIR / "patches.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def test_plain_text():
    assert scan("太祖武皇帝，沛國譙人也") == ["太祖武皇帝，沛國譙人也"]


def test_one_template():
    assert scan("甲{{YL|太和三年|229年}}乙") == ["甲", Template("YL|太和三年|229年"), "乙"]


def test_nested_template_kept_raw():
    # 卷46 的真实写法: 裴注里嵌着书名线模板
    text = "之後也。{{*|{{WavyBookMark|呉書}}曰：堅世仕呉}}少爲縣吏"
    assert scan(text) == ["之後也。", Template("*|{{WavyBookMark|呉書}}曰：堅世仕呉"), "少爲縣吏"]


def test_single_brace_is_text():
    # -{後}- 是繁简转换保护符，不是模板
    assert scan("孫武之-{後}-也") == ["孫武之-{後}-也"]


def test_adjacent_templates_no_empty_strings():
    assert scan("{{A}}{{B}}") == [Template("A"), Template("B")]


def test_unbalanced_raises():
    for bad in ["甲{{YL|太和", "甲}}乙", "{{A}}}}"]:
        try:
            scan(bad)
        except ValueError:
            continue
        raise AssertionError(f"应当报错却没有: {bad!r}")


PATCH = [{"file": "卷01.wiki", "find": "甲乙", "replace": "甲{{*|注}}乙"}]


def test_patch_applies():
    assert apply_patches("卷01.wiki", "前甲乙後", PATCH) == "前甲{{*|注}}乙後"


def test_patch_other_file_skipped():
    assert apply_patches("卷02.wiki", "前甲乙後", PATCH) == "前甲乙後"


def test_patch_missing_or_duplicate_raises():
    for text in ["前丙丁後", "甲乙甲乙"]:  # 找不到 / 出现两次
        try:
            apply_patches("卷01.wiki", text, PATCH)
        except ValueError:
            continue
        raise AssertionError(f"应当报错却没有: {text!r}")



# ---- 第 2a 步: split_args ----

def test_split_simple():
    assert split_args("YL|太和三年|229年") == ["YL", "太和三年", "229年"]
    assert split_args("gap") == ["gap"]
    assert split_args("quote|") == ["quote", ""]


def test_split_skips_link_pipe():
    assert split_args("WavyBookMark|[[王沈魏書|魏書]]") == ["WavyBookMark", "[[王沈魏書|魏書]]"]


def test_split_skips_nested_template_pipe():
    assert split_args("blue|{{-|臣}}海濵孤微") == ["blue", "{{-|臣}}海濵孤微"]


def test_split_keeps_whitespace():
    raw = "header2\n| title = [[../]]\n| section = 魏書·文帝紀\n"
    assert split_args(raw) == ["header2\n", " title = [[../]]\n", " section = 魏書·文帝紀\n"]


def test_split_unbalanced_link_raises():
    for bad in ["WavyBookMark|[[傅子", "ul|倭人]]"]:
        try:
            split_args(bad)
        except ValueError:
            continue
        raise AssertionError(f"应当报错却没有: {bad!r}")


def _all_templates():
    """全书所有模板 (含嵌套在内的), 套用补丁后."""
    patches = _load_patches()
    out = []

    def walk(parts):
        for p in parts:
            if isinstance(p, Template):
                out.append(p.raw)
                walk(scan(p.raw))

    for f in sorted(RAW_DIR.glob("卷*.wiki")):
        walk(scan(apply_patches(f.name, f.read_text(encoding="utf-8"), patches)))
    return out


def test_split_real_templates():
    raws = _all_templates()
    if not raws:
        raise Skip("没有下载 raw 文件")
    for raw in raws:
        args = split_args(raw)
        assert "|".join(args) == raw, raw[:40]
        name = args[0].strip()
        if name == "ProperNoun":  # 全书 2550 处都是一个参数
            assert len(args) == 2, raw
        if name == "YL":  # 只有年号，或年号 + 公元年
            assert len(args) in (2, 3), raw



# ---- 第 2b 步: unwrap_links ----

def test_unwrap_basic():
    assert unwrap_links("[[w:孫堅|孫堅]]字文臺") == "孫堅字文臺"
    assert unwrap_links("吳人作《[[曹瞞傳]]》") == "吳人作《曹瞞傳》"
    assert unwrap_links("王沈《[[王沈魏書|魏書]]》曰") == "王沈《魏書》曰"


def test_unwrap_category_removed():
    assert unwrap_links("評曰[[Category:三國志]]") == "評曰"
    assert unwrap_links("[[category:西晉作品]]") == ""
    assert unwrap_links("評曰[[category:三國志|62]]") == "評曰"  # 带排序键, 卷62 等卷末


def test_unwrap_keeps_templates():
    assert unwrap_links("[[w:孫堅|{{ProperNoun|孫堅}}]]字") == "{{ProperNoun|孫堅}}字"
    assert unwrap_links("{{WavyBookMark|[[傅子]]}}曰") == "{{WavyBookMark|傅子}}曰"


def test_unwrap_real_files():
    files = sorted(RAW_DIR.glob("卷*.wiki"))
    if not files:
        raise Skip("没有下载 raw 文件")
    patches = _load_patches()
    for f in files:
        out = unwrap_links(apply_patches(f.name, f.read_text(encoding="utf-8"), patches))
        assert "[[" not in out and "]]" not in out, f.name
        scan(out)  # 拆完链接后模板仍然配对



# ---- 第 2c 步: render_template ----

def test_render_take_first():
    assert render_template(["ProperNoun", "張遼"]) == Rendered("張遼", "張遼")
    assert render_template(["blue", "{{-|臣}}海濵"]) == Rendered("{{-|臣}}海濵", "{{-|臣}}海濵")
    assert render_template(["!", "𢢼", "〈忄剽〉"]) == Rendered("𢢼", "𢢼")
    assert render_template(["ProperNoun", ""]) == Rendered("", "")  # 卷46 的空专名线


def test_render_year():
    assert render_template(["YL", "太和三年", "229年"]) == Rendered("太和三年", "太和三年 229年")
    assert render_template(["YL", "黃初元年"]) == Rendered("黃初元年", "黃初元年")


def test_render_variants():
    assert render_template(["校", "潘陽", "鄱陽"]) == Rendered(
        "鄱陽", "鄱陽", {"kind": "校", "adopted": "鄱陽", "other": "潘陽"})
    assert render_template(["另", "且", "旦"]) == Rendered(
        "且", "且", {"kind": "另", "adopted": "且", "other": "旦"})
    assert render_template(["別", "禦", "御"]).variant == {"kind": "另", "adopted": "禦", "other": "御"}


def test_render_dropped():
    for args in (["gap"], ["footer"], ["header2\n", " title = x\n"], ["wikipedia", "曹芳"], ["PD-old"]):
        assert render_template(args) == Rendered("", ""), args


def test_render_raises():
    for args in (["*", "注文"], ["*s"], ["*e"], ["不存在的模板", "x"]):
        try:
            render_template(args)
        except ValueError:
            continue
        raise AssertionError(f"应当报错却没有: {args!r}")


def test_render_real_templates():
    raws = _all_templates()
    if not raws:
        raise Skip("没有下载 raw 文件")
    for raw in raws:
        args = split_args(raw)
        if args[0].strip() in ("*", "*s", "*e"):
            continue
        render_template(args)  # 全书除裴注外每种模板都要认识


def _rebuild(parts):
    return "".join(p if isinstance(p, str) else "{{" + p.raw + "}}" for p in parts)


def test_real_files_roundtrip():
    # 不变量: 把切出来的东西拼回去, 必须和原文一字不差
    files = sorted(RAW_DIR.glob("卷*.wiki"))
    if not files:
        raise Skip("没有下载 raw 文件")
    patches = _load_patches()
    for f in files:
        text = apply_patches(f.name, f.read_text(encoding="utf-8"), patches)
        assert _rebuild(scan(text)) == text, f.name


def test_juan01_note_count():
    f = RAW_DIR / "卷01.wiki"
    if not f.exists():
        raise Skip("没有下载 raw 文件")
    parts = scan(f.read_text(encoding="utf-8"))
    notes = [p for p in parts if isinstance(p, Template) and p.raw.startswith("*|")]
    assert len(notes) == 122, len(notes)


if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
