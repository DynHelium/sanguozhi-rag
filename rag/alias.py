"""别名表 data/aliases.json 的读取, 以及检索前的查询扩展 expand_query.

原文和译文里同一个人常用不同称呼: 武帝纪写 "太祖" "曹公", 不写 "曹操". 用户问 "曹操" 时, 先把 "魏太祖 曹公 ..." 追加到
查询里再检索. 只改查询, 不改索引, 所以别名表更新后不用重建索引.

检索之外, 按称呼查人 (这个称呼指谁, 有没有同名的人) 也用这里的 load_aliases / alias_index / person_forms.

data/aliases.json 的来历: 大模型对照原文 (和白话译文) 逐卷整理有传人物的姓名, 字, 称号, 每个称呼带出处 (src);
再用代码核对每个 orig 确实出现在 src 的原文里, 通过的卷合并成这一个文件. 只收原文里有的称呼, 不用演义或现代资料.
"""
import json
from functools import lru_cache
from typing import NotRequired, TypeAlias, TypedDict

from rag import config
from rag.normalize import to_simplified


class Form(TypedDict):
    """一个称呼, 如 {"form": "魏太祖", "orig": "太祖", "src": "01-0000", "kind": "帝号"}."""
    form: str  # 译文 (简体) 里的写法
    orig: str  # 原文 (繁体) 里的原词
    src: str   # 出处的段落 id 或裴注 id
    kind: str  # 帝号 / 爵号 / 官称 / 字 / 称呼 / 异名


class Person(TypedDict):
    """别名表里的一个人 (people 的一项). 全表四百多人, 例:

        {"id": "01-0000", "name": "曹操", "surname": "曹", "zi": "孟德", "juan": 1,
         "forms": [{"form": "魏太祖", "orig": "太祖", "src": "01-0000", "kind": "帝号"},
                   {"form": "魏武帝", "orig": "武皇帝", "src": "01-0000", "kind": "帝号"}, ...],
         "note": "武帝纪的单字'公''王'未收(泛称). ..."}

        三国志里没有传的人 (如司马懿) 多一个 "notrans": true, 这时 id 是原文提到他的一处段落 id, juan 是 0:
        {"id": "02-0036", "name": "司马懿", ..., "juan": 0, "notrans": true}
    """
    id: str        # 唯一标识 (同名的人 id 不同): 有传的人是传的首段 id; notrans 的人见上
    name: str      # 姓名, 如 "曹操"
    surname: str   # 姓, 复姓写全, 如 "诸葛"
    zi: str        # 字, 原文没写就是 ""
    forms: list[Form]
    note: str      # 整理时的备注 (哪些称呼没收, 为什么), 常为 ""; 只给人看, 代码不用
    juan: int      # 传所在的卷; notrans 的人为 0
    notrans: NotRequired[bool]  # true = 三国志里没有传, 只是常被用称号称呼的人


class Homonym(TypedDict):
    """表外的一个同名者, 如 {"src": "04-0030", "desc": "高贵乡公(曹髦)时三月所立皇后卞氏, ..."}."""
    src: str   # 出处的段落 id 或裴注 id
    desc: str  # 这个人是谁 (简短描述)


class AliasTable(TypedDict):
    """data/aliases.json 的结构."""
    people: list[Person]
    homonyms: dict[str, list[Homonym]]        # "马忠" -> 表外同名者的出处: 这个名字不能唯一指表里那个人
    variants: dict[str, list[str]]            # "曹叡" -> ["曹睿"]: 异体写法, 只用来认人
    variant_surnames: dict[str, list[str]]    # "毌丘" -> ["毋丘"]: 复姓的异体


Forms: TypeAlias = tuple[str, ...]                # 一个人的全部称呼, 如 ("曹操", "曹孟德", "魏太祖", ...)
AliasKeys: TypeAlias = dict[str, frozenset[int]]  # 称呼 -> people 里的下标; -1 表示表外还有同名的人

# 泛称: 单独出现不能确定是谁, 既不当作查询里的人名, 也不追加进查询.
GENERIC = {"丞相", "将军", "大将军", "主公", "明公", "先帝", "至尊", "陛下", "主上", "皇帝", "天子", "太子", "世子",
           "公", "王", "帝", "上", "主", "君", "太祖", "高祖", "世祖", "太后", "皇后", "夫人"}
MAX_EXTRA = 8  # 每个人最多追加几个称呼


@lru_cache
def load_aliases() -> AliasTable:
    """读 data/aliases.json, 不做任何处理. 文件不存在时返回空表 (这时 expand_query 原样返回查询)."""
    path = config.DATA_DIR / "aliases.json"
    if not path.exists():
        return {"people": [], "homonyms": {}, "variants": {}, "variant_surnames": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def person_forms(p: Person) -> Forms:
    """一个人的全部称呼, 转成简体, 去重, 保持表里的顺序: 姓名, 姓+字, 然后是 forms 里的各称呼.
    不收单字和泛称 (GENERIC): 它们单独出现时认不出是谁.

        曹操 -> ("曹操", "曹孟德", "魏太祖", "魏武帝", "魏武王", "魏公", "魏王", "曹公", "阿瞒")
    """
    forms = [p["name"]] + ([p["surname"] + p["zi"]] if p.get("zi") else []) + [f["form"] for f in p["forms"]]
    return tuple(dict.fromkeys(to_simplified(f) for f in forms if len(f) >= 2 and f not in GENERIC))


@lru_cache
def alias_index() -> tuple[AliasKeys, tuple[Forms, ...]]:
    """把别名表建成两张方向相反的表, 返回 (keys, people):

        people[i]  第 i 个人的全部称呼, 用来往查询里追加 (见 person_forms)
                   people[0] = ("曹操", "曹孟德", "魏太祖", "魏武帝", "曹公", ...)
        keys[称呼] 这个称呼指谁 (people 的下标集合), 用来认出查询里提到的人
                   keys["魏太祖"] = {0}        唯一, 可以展开
                   keys["魏王"]   = {0, 1}     曹操和曹丕都称魏王, 不展开
                   keys["马忠"]   = {295, -1}  表外还有东吴马忠 (homonyms), 加 -1 使它不唯一, 不展开

    异体字 (variants: "曹睿" -> 曹叡) 和复姓异体 (variant_surnames: "毋丘俭" -> 毌丘俭) 只进 keys, 不进 people:
    查询里写 "曹睿" 能认出是曹叡, 但不会把 "曹睿" 追加进查询.

    people 用下标而不用姓名当 key, 是因为姓名不保证唯一.

    keys 按称呼长度从长到短排列 (长度相同保持表里的顺序): expand_query 按这个顺序做最长匹配, 排序放在这里只做一次 (有 cache).
    """
    data = load_aliases()
    people: list[Forms] = []
    keys: dict[str, set[int]] = {}
    for i, p in enumerate(data["people"]):
        people.append(person_forms(p))
        extra = list(data.get("variants", {}).get(p["name"], []))
        for sur, alts in data.get("variant_surnames", {}).items():
            if p["name"].startswith(sur):
                extra += [a + p["name"][len(sur):] for a in alts]
        for f in people[-1] + tuple(to_simplified(x) for x in extra):
            keys.setdefault(f, set()).add(i)
    for name in data.get("homonyms", {}):
        keys.setdefault(to_simplified(name), set()).add(-1)
    return {k: frozenset(v) for k, v in sorted(keys.items(), key=lambda x: -len(x[0]))}, tuple(people)


def expand_query(query: str) -> str:
    """把查询里提到的人的其他称呼追加到查询末尾, 让检索能搜到原文里用别称写的段落. 返回繁转简后的查询.

        "曹操因为什么逃离洛阳？" -> "曹操因为什么逃离洛阳？ 曹孟德 魏太祖 魏武帝 曹公 ..."
        "魏太祖是谁"           -> "魏太祖是谁 曹操 曹孟德 ..."    从任何一个称呼都能认出人
        "曹睿字什么？"         -> "曹睿字什么？ 曹叡 ..."         异体字能认人, 但只追加表里的写法
        "魏王是谁"             -> 不变                           魏王 = 曹操或曹丕, 不唯一
        "三国志里有几个马忠？"   -> 不变                           马忠不止一个人

    规则:
    - 称呼按长度从长到短在查询里找 (alias_index 已排好序). 找到后在工作副本里把它全部换成等长的占位符 "X",
      所以被长称呼占住的字不会再匹配短称呼 ("平原侯" 里的 "原侯"), 也不会因为删字把两边的字拼成新称呼.
    - 不唯一的称呼也占住这几个字, 只是不展开.
    - 追加顺序按人在别名表里的顺序; 每人最多 MAX_EXTRA 个; 查询里匹配到的称呼不追加; 几个人共有的称呼只追加一次.
    """
    query = to_simplified(query)
    original_query = query
    keys, people = alias_index()
    if not keys:
        return query
    valid_indices = set[int]()
    found_names = set[str]()
    for name in keys.keys():
        if query.find(name) > -1:
            if len(keys[name]) == 1:
                [index] = keys[name]
                valid_indices.add(index)
            found_names.add(name)
            query = query.replace(name, "X" * len(name))
    extra = list[str]()
    for index in sorted(list(valid_indices)):
        extra.extend([n_ for n_ in people[index] if n_ not in found_names][:MAX_EXTRA])
    extra_dedup = dict.fromkeys(extra)
    return original_query + (" " + " ".join(extra_dedup.keys()) if extra else "")
