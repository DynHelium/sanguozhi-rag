"""人名的拼音索引: data/aliases.json 和 data/aliases_extra.json 里的名字 -> 拼音键, 写 data/aliases_pinyin.json.
给 who 工具认拼音写的人名 (英文提问常写 Zhou Tai, Chou T'ai), 并把拼音相同的人一起列出来.

    uv run --with pypinyin python -m ingest.pinyin_index     # 改了别名表后重建; pypinyin 只在这里用, 不进依赖

每个人收这几种写法: 姓名, 姓 + 字, 字 (两个字以上; 键指向 "姓 + 字", 因为 who 认 "诸葛孔明" 不认单独的 "孔明");
aliases_extra.json 的 name 和 sgz. 每种写法生成两个键:
    汉语拼音 (不带声调)   周泰 -> zhoutai          诸葛亮 -> zhugeliang
    威妥玛拼音            周泰 -> choutai          诸葛亮 -> chukoliang
键都经过 key(): 小写, 去掉空格, 撇号, 连字符和声调, ü / v 写成 u. 所以 "Zhou Tai" "zhou-tai" "Chou T'ai" 查到的是同一个键.

输出 (键 -> 这个拼音对应的中文写法, 去重, 按表里的顺序):
    {"zhoutai": ["州泰", "周泰"], "liuyan": ["刘焉", "刘琰", "刘延"], "kongming": ["诸葛孔明"], ...}

多音字按 pypinyin 的默认读音, 人名里的特殊读音会错: 乐进 默认生成 lejin, 英文通行的是 Yue Jin. 姓氏的特殊读音在 SURNAME_READINGS
里补上 (以这个字开头的名字另外生成一组键, 首字换成这个读音); 名字中间的多音字不处理.
只收表里的人 (有传的人和少数常被只称称号的人).
"""
import json

from pypinyin import Style, lazy_pinyin

from rag import config
from rag.alias import key
from rag.normalize import to_simplified


SURNAME_READINGS = {"乐": ("yue", "yueh")}  # 姓氏的特殊读音: 字 -> (汉语拼音, 威妥玛拼音)


def keys_of(name: str) -> set[str]:
    """一个中文写法的拼音键 (汉语拼音和威妥玛拼音, 都经 key() 归一); 首字是 SURNAME_READINGS 里的姓时另加一组."""
    py, wg = lazy_pinyin(name), lazy_pinyin(name, style=Style.WADEGILES)
    keys = {key("".join(py)), key("".join(wg))}
    if name[:1] in SURNAME_READINGS:
        a, b = SURNAME_READINGS[name[0]]
        keys |= {key(a + "".join(py[1:])), key(b + "".join(wg[1:]))}
    return keys


def build() -> dict[str, list[str]]:
    """读两张别名表, 返回 {拼音键: [中文写法, ...]}."""
    data = json.loads((config.DATA_DIR / "aliases.json").read_text(encoding="utf-8"))
    pairs: list[tuple[str, str]] = []  # (拿来转拼音的写法, 查到后交给 who 的写法)
    for p in data["people"]:
        pairs.append((p["name"], p["name"]))
        if p.get("zi"):
            pairs.append((p["surname"] + p["zi"], p["surname"] + p["zi"]))
            if len(p["zi"]) >= 2:
                pairs.append((p["zi"], p["surname"] + p["zi"]))
    pairs += [(n, n) for n in data.get("homonyms", {})]
    extra = config.DATA_DIR / "aliases_extra.json"
    if extra.exists():
        for e in json.loads(extra.read_text(encoding="utf-8"))["names"]:
            pairs += [(e["name"], e["name"]), (e["sgz"], e["sgz"])]
    index: dict[str, list[str]] = {}
    for src, target in dict.fromkeys((to_simplified(a), to_simplified(b)) for a, b in pairs):
        for k in keys_of(src):
            index.setdefault(k, [])
            if target not in index[k]:
                index[k].append(target)
    return dict(sorted(index.items()))


def main() -> None:
    index = build()
    out = config.DATA_DIR / "aliases_pinyin.json"
    out.write_text(json.dumps(index, ensure_ascii=False, indent=0), encoding="utf-8")
    shared = {k: v for k, v in index.items() if len(v) > 1}
    print(f"{len(index)} 个拼音键 -> {out}; 其中 {len(shared)} 个键对应不止一个写法")


if __name__ == "__main__":
    main()
