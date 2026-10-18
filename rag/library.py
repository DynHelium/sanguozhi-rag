"""agent 工具背后的数据: Library (search / read / era / who 四个工具的查询) 和 Seen (模型看过的块, 编号, 展示文字).

    uv run python -m tests.test_library   # 不连网, 不调模型

这里只管查询和拼文字, 不涉及工具怎么声明给模型. 调用方 (把这些方法注册成工具的 agent 代码) 把模型给的参数传进来,
把结果交给 Seen.add 变成 role "tool" 消息的 content, 并接住方法抛出的 KeyError / ValueError / TypeError,
把错误说明回给模型.
"""
import threading
from typing import TYPE_CHECKING

from rag.alias import alias_index, key, load_aliases, load_extras, load_pinyin, person_forms
from rag.normalize import to_simplified
from rag.quotes import source_label
from rag.retrieve import load_chunks, make_search
from rag.schema import ChunkDict, Search, Where

if TYPE_CHECKING:
    from ingest.eras import Era

SEARCH_K = 10   # 模型调用 search 时取几组 (同一源段切出的几块算一组, 见 Library.search). 取得太少时, 只搜 "马忠"
                # 名额会被蜀汉马忠本传等占满, 吴书里顺带提到的另一个马忠排不进来
READ_MAX = 3    # read 前后各最多几块
BOOKS = {"魏": "魏書", "蜀": "蜀書", "吴": "吳書"}  # book 参数繁转简后的首字 -> 块的 book 字段 (繁体)
PARTS = {"正文": 1, "裴注": 2}                                   # search 的 part 参数 -> 块的 tier 字段


ALIAS = "route2"  # 检索时的别名扩展方式 (rag/retrieve.make_search): 原查询照常检索, 扩展后的查询另加两路融合.
                  # 用扩展后的查询替换原查询会丢掉原查询的结果; 另加一路让它们照样参与融合 (见 docs/DESIGN.md).


# ---------- 工具背后的数据查询 ----------

def scope(book: str | None, part: str | None) -> Where | None:
    """search 的 book / part 参数 -> 检索范围 (rag/schema.Where); 两个都没给时返回 None (搜全书).

    book 繁转简后只看首字 (魏 / 蜀 / 吴), 所以 "吴书" "吳書" "吴" 都认, "吴国" "魏志" 也当作吴书 / 魏书;
    part 要整词 "正文" (tier 1) 或 "裴注" (tier 2). None, 空字符串和只有空白都当作没给.
    认不出的值或不是字符串 (模型可能给数字, 列表) 抛 ValueError, 信息里带原值, 由调用方回给模型."""
    for name, v in (("book", book), ("part", part)):
        if v is not None and not isinstance(v, str):
            raise ValueError(f"{name} 要是字符串, 收到 {v!r}")
    b = to_simplified(book.strip())[:1] if book and book.strip() else None
    p = part.strip() if part and part.strip() else None
    if b and b not in BOOKS:
        raise ValueError(f"book 只能是 魏书 / 蜀书 / 吴书, 收到 {book!r}")
    if p and p not in PARTS:
        raise ValueError(f"part 只能是 正文 / 裴注, 收到 {part!r}")
    if not b and not p:
        return None
    want_book, want_tier = BOOKS.get(b or ""), PARTS.get(p or "")
    return lambda c: (not want_book or c["book"] == want_book) and (not want_tier or c["tier"] == want_tier)


class Library:
    """工具背后的数据查询: read / search / era / who 各一个方法, 对应模型可调用的四个工具.

    构造: Library(chunks, search, eras_table=None)
        chunks      load_chunks(index) 的结果 (rag/schema.ChunkDict 的列表), 顺序就是原文顺序:
                    每个正文块后面紧跟它那一段的裴注块, 如 [..., "01-0003n0", "01-0004", "01-0004n0", ..., "01-0004n3", "01-0005", ...]
        search      search(query, k, where=None) -> 前 k 个块, 从好到差 (rag/schema.Search; rag/retrieve.make_search 返回它).
                    Library.search 不限定范围时只传 (query, k), 所以只收两个参数的检索器也能用
        eras_table  年号表 (ingest/eras.load() 的结果); 不给就读 data/eras.json

    属性:
        chunks      同上
        search_fn   同上; 是普通属性, 调用方可以换成别的实现 (如多线程跑时加锁的版本)
        pos         chunk id -> 在 chunks 里的下标, 如 {"01-0000": 0, "01-0004": 15, ...}
        by_source   源 id (段落 id 或裴注 id) -> 含它的块的下标 (按原文顺序), 如
                    {"32-0011": [i, i + 1], ...}   段落太长被切成 32-0011.0 / 32-0011.1 两块
                    {"35-0012": [j], ...}          段落太短, 并进了 35-0011 那一块
        notes       段落 id -> 挂在它上面的裴注块 (按原顺序):
                    {"10-0023": [<10-0023n0>, <10-0023n1>, <10-0023n2.0>, <10-0023n2.1>], ...}
                    (n2.0 / n2.1 是同一条注太长被切成两块; 只有有注的段落才有键)
        eras        年号表, 键是繁体年号: {"建安": {"status": "ref", "start": 196, "by_regime": {"东汉": 196}, ...}, ...} (见 era)
        era_keys    简体写法 -> 表里的键, 如 {"黄初": "黃初", "建兴": "建興", ...}; 模型可能写简体
    """
    search_fn: Search

    def __init__(self, chunks: list[ChunkDict], search: Search, eras_table: "dict[str, Era] | None" = None) -> None:
        self.chunks, self.search_fn = chunks, search
        self.pos = {c["id"]: i for i, c in enumerate(chunks)}
        self.by_source: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            for s in c["sources"]:
                self.by_source.setdefault(s, []).append(i)
        self.notes: dict[str, list[dict]] = {}
        for c in chunks:
            if parent := c.get("parent"):
                self.notes.setdefault(parent, []).append(c)
        if eras_table is None:
            from ingest import eras
            eras_table = eras.load()
        self.eras: "dict[str, Era]" = eras_table
        self.era_keys = {to_simplified(k): k for k in eras_table}  # 模型可能写简体 (黄初, 青龙)

    def read(self, chunk_id: str, before: int = 0, after: int = 0) -> list[dict]:
        """read 工具: 返回 chunk_id 这一块和它前后的块 (块 dict 的列表, 按下面的顺序, 不重复).

        chunk_id 可以是块 id, 也可以是源 id (段落 id 或裴注 id): 模型常拿原文里的段落编号来读, 而块 id 不一定相同.
        源 id 不是块 id 时, 读含它的全部块: 段落被切开的读所有片 (read("32-0011") 读 32-0011.0 和 32-0011.1),
        并进别的块的读那一块 (read("35-0012") 读 35-0011). 下面的 "chunk_id 这一块" 指这一块或这几块.

        步骤:
            1. 窗口: self.chunks 里 chunk_id 前 before 块到后 after 块 (按列表位置数, 裴注块也算一块),
               只留和 chunk_id 同一卷 (juan) 的, 跨卷的直接丢掉, 不往回补. before / after 先转 int 再截到 0~READ_MAX.
            2. chunk_id 是裴注块 (有 parent) 时, 把它注的正文块 (tier 1 且 sources 含 parent; 正文段被切开时有几块)
               放到最前面, 已在窗口里的不重复.
            3. 依次输出上面的每一块; 段落的全部裴注 (self.notes) 紧跟在窗口里这个段落的最后一个正文块后面
               (段落被切成几片时, 注在最后一片后面, 和 chunks 文件的顺序一致), 已输出的跳过.
        为什么附全部注: 一段正文可能有好几条注, 窗口只取 after=1 会漏掉后面的注 (10-0023 有 n0 n1 n2.0 n2.1, 荀粲只在 n2.1 里).

        例 (真实数据, 只列 id):
            read("01-0004", before=1, after=1)  -> ["01-0003n0", "01-0004", "01-0004n0", "01-0004n1", "01-0004n2", "01-0004n3"]
                窗口是 01-0003n0, 01-0004, 01-0004n0; 01-0003n0 是裴注块, 不附任何东西; 01-0004 带出它的四条注.
            read("10-0023n2.1")                 -> ["10-0023", "10-0023n0", "10-0023n1", "10-0023n2.0", "10-0023n2.1"]
                先放所注正文 10-0023, 再跟它的全部注; 被读的 n2.1 已在其中, 不再重复.
        出错: chunk_id 既不是块 id 也不是源 id 时抛 KeyError; before / after 转不成 int 时抛 ValueError / TypeError
        (由调用方接住, 把错误说明回给模型)."""
        if chunk_id in self.pos:
            first = last = self.pos[chunk_id]
        elif chunk_id in self.by_source:
            first, last = self.by_source[chunk_id][0], self.by_source[chunk_id][-1]
        else:
            raise KeyError(chunk_id)
        i = first
        before, after = max(0, min(READ_MAX, int(before))), max(0, min(READ_MAX, int(after)))
        juan = self.chunks[i]["juan"]
        out = [c for c in self.chunks[max(0, first - before): last + after + 1] if c["juan"] == juan]
        parent = self.chunks[i].get("parent")
        if parent:
            head = [c for c in self.chunks if c["tier"] == 1 and parent in c["sources"] and c not in out]
            out = head + out
        last_of = {s: k for k, c in enumerate(out) if c["tier"] == 1 for s in c["sources"]}  # 段落 -> 它在窗口里的最后一块
        res, ids = [], set()
        for k, c in enumerate(out):
            notes = [n for s in c["sources"] if last_of.get(s) == k for n in self.notes.get(s, [])] if c["tier"] == 1 else []
            for x in [c] + notes:
                if x["id"] not in ids:
                    ids.add(x["id"]); res.append(x)
        return res

    def search(self, query: str, book: str | None = None, part: str | None = None) -> list[dict]:
        """search 工具: 返回最多 SEARCH_K 组检索结果 (块 dict 的列表, 保持检索器给的顺序).

        book / part 限定范围 (可选, 不给就搜全书): book 是 "魏书" "蜀书" "吴书" (简繁都认, 只看首字, "吴" 也行),
        part 是 "正文" 或 "裴注". 范围交给检索器先筛选再排名 (rag/schema.Search), 所以冷门的裴注不会被全书里
        更相关的段落挤出前几名. 认不出的值抛 ValueError (由调用方把说明回给模型).

        组 = 块的 sources[0]. 同一个源段太长被切成几块 (43-0011.0 / 43-0011.1, sources 都是 ["43-0011"]) 时算一组:
        都给模型, 但只占一个名额, 免得一段长传把名额占满.
        做法: 取 self.search_fn(query, 2 * SEARCH_K) 个候选, 按顺序走:
            - 新组, 且已有 SEARCH_K 组 -> 跳过这块
            - 新组, 还不满 -> 记下这一组, 收下这块
            - 已有的组 -> 收下这块 (名额满了以后也收)
        所以返回的块数可以多于 SEARCH_K (如 SEARCH_K 组共 SEARCH_K + 1 块: 头一组是切成两块的长段).
        查询里有《三国志》以外的叫法 (rag/alias.load_extras, 如 孙尚香) 时, 先在查询后面补上《三国志》的写法 (孙夫人):
        原文和译文里只有后者."""
        q = to_simplified(query)
        for name, e in load_extras().items():
            if name in q and e["sgz"] not in q:
                query += " " + e["sgz"]
                q += " " + e["sgz"]
        where = scope(book, part)
        hits = self.search_fn(query, SEARCH_K * 2, where) if where else self.search_fn(query, SEARCH_K * 2)
        out, groups = [], []
        for c in hits:
            g = c["sources"][0]
            if g not in groups:
                if len(groups) == SEARCH_K:
                    continue
                groups.append(g)
            out.append(c)
        return out

    def era(self, era: str, year: int | str, regime: str | None = None) -> str:
        """era 工具: 年号第 year 年 -> 交给模型的一句话.

        参数 (模型给的, 可能不规范):
            era     年号, 繁体或简体都认 ("黃初" / "黄初"): 去空白, 转简体后经 self.era_keys 找到表里的键
            year    第几年: int (7), 数字字符串 ("7", "7年"), 或汉字 ("七", "元", "七年"). 先去掉末尾的 "年",
                    剩下的是阿拉伯数字就直接转 int, 否则用 ingest.eras.cn_number(s: str) -> int, 不认识的抛 ValueError
            regime  政权 (可选), 只在多政权年号时用; 繁简都认 ("蜀汉")

        self.eras 里一项的样子 (只列用到的字段; status 决定怎么答):
            "建安": {"status": "ref", "start": 196, "by_regime": {"东汉": 196}, ...}
            "建興": {"status": "多政权", "start": None, "by_regime": {"蜀汉": 223, "孙吴": 252, "西晋": 313}, ...}
            status 不是以上三种 ("ok" / "ref" / "多政权") 时按待核处理: 公元年不确定

        返回 (真实输出; 公元年 = 元年的公元年 + year - 1):
            era("建安", 7) / era("建安", "七")  -> "建安7年 = 公元 202 年"          status "ok" / "ref", 用 ingest.eras.era_to_ce
                (签名 era_to_ce(era, n, table=None, regime=None) -> int | None; 传 regime 时多政权的也能算, None 表示算不出)
            era("黄初", "元")                   -> "黃初1年 = 公元 220 年"          输出用表里的键 (繁体)
            era("建興", 1, "蜀汉")              -> "蜀汉 建興1年 = 公元 223 年"     多政权, regime 对得上
            era("建興", 1)                      -> "建興 是多个政权的年号, 第 1 年分别是: 蜀汉 公元 223 年, 孙吴 公元 252 年, 西晋 公元 313 年"
                                                   多政权, 没给 regime 或对不上: 列出全部
            era("不存在", 1)                    -> "年号表里没有 不存在."
            status 是其他值                     -> "某某 的公元年待核, 无法换算."
        出错: 查不到年号或待核时返回说明文字, 不抛异常; year 解析不了 ("abc", None) 时 cn_number 抛 ValueError
        (由调用方接住, 把错误说明回给模型)."""
        from ingest.eras import cn_number, era_to_ce
        s = str(year).rstrip("年")
        n = year if isinstance(year, int) else (int(s) if s.isdigit() else cn_number(s))
        key = self.era_keys.get(to_simplified(era.strip()))
        if key is None:
            return f"年号表里没有 {era}."
        e = self.eras[key]
        if e["status"] in ("ok", "ref"):
            return f"{key}{n}年 = 公元 {era_to_ce(key, n, self.eras)} 年"
        if e["status"] == "多政权":
            regs = e["by_regime"]
            if regime and to_simplified(regime) in {to_simplified(r) for r in regs}:
                r = next(r for r in regs if to_simplified(r) == to_simplified(regime))
                return f"{r} {key}{n}年 = 公元 {regs[r] + n - 1} 年"
            return f"{key} 是多个政权的年号, 第 {n} 年分别是: " + ", ".join(f"{r} 公元 {s + n - 1} 年" for r, s in regs.items())
        return f"{key} 的公元年待核, 无法换算."


    def _who_lines(self, q: str) -> list[str]:
        """别名表里称呼 q (简体) 对得上的人, 每人一行 (格式见 who), 再加表外的同名者; 对不上时为空."""
        keys, _ = alias_index()
        data = load_aliases()
        lines = []
        for i in sorted(x for x in keys.get(q, ()) if x >= 0):
            p = data["people"][i]
            where = "无传, 出处" if p.get("notrans") else "传的首段"
            lines.append(f"- {p['name']}" + (f", 字{p['zi']}" if p.get("zi") else "") + f" ({where} {p['id']}); 称呼: "
                         + ", ".join(person_forms(p)))
        others = data.get("homonyms", {}).get(q, [])
        lines += [f"- 另一个 {q} (出处 {o['src']}): {o['desc']}" for o in others]
        return lines

    def _extra_line(self, q: str) -> str | None:
        """q 是《三国志》以外的叫法时 (rag/alias.load_extras), 返回说明它来历和《三国志》写法的一行; 否则 None."""
        e = load_extras().get(q)
        if e is None:
            return None
        what = {"他书": "其他史书的写法", "演义": "《三国演义》等小说里的叫法, 不是《三国志》的记载",
                "传说": "后世民间的说法, 不是《三国志》的记载"}.get(e["kind"], e["kind"])
        return f"- {q}: {what}; 《三国志》里查 {e['sgz']} (出处 {e['src']}): {e['desc']}. 检索请用 {e['sgz']}."

    def who(self, name: str) -> str:
        """who 工具, 查别名表 (data/aliases.json): name 可以是姓名, 姓 + 字, 称号 (魏太祖, 先主), 异体 (曹睿); 繁简都认.
        name 里没有汉字时当拼音查 (见 _who_pinyin); 有汉字时只按其中的汉字查 ("周泰 Zhou Tai" -> 周泰).
        name 是《三国志》以外的叫法 (data/aliases_extra.json, 如 孙尚香) 时, 先给一行来历 (_extra_line), 再按它的 sgz 查表.
        返回交给模型的文字: 每个对得上的人一行 (姓名, 字, 传的首段 id 或无传者的出处, 全部称呼), 再列表外的同名者 (如东吴马忠).
        表里只有有传的人和少数常被只称称号的人; 查不到时返回的文字提示模型直接 search. 例 (真实输出):
            who("马忠") -> "马忠: 共 2 个人.
                           - 马忠, 字德信 (传的首段 43-0011); 称呼: 马忠, 马德信, 狐笃, 彭乡亭侯
                           - 另一个 马忠 (出处 47-0016): 潘璋司马，擒获关羽及其子关平、都督赵累于章乡
                           同名的人要分别 search (加上字, 官职或所属势力区分)."
            who("先主") -> "先主:\n- 刘备, 字玄德 (传的首段 32-0000); 称呼: 刘备, 刘玄德, 先主, 蜀汉昭烈皇帝, 汉中王, 刘豫州"
        """
        q = to_simplified(name.strip())
        han = "".join(c for c in q if "\u4e00" <= c <= "\u9fff")
        if q and not han:  # 拼音 (Zhou Tai, Chou T'ai)
            return self._who_pinyin(name)
        q = han  # 有汉字就只按汉字查 ("周泰 Zhou Tai" -> 周泰)
        extra = self._extra_line(q)
        found = self._who_lines(load_extras()[q]["sgz"] if extra else q)
        lines = ([extra] if extra else []) + found
        if not lines:
            return (f"别名表里没有 {name} (表里只有有传的人和少数常被只称称号的人). 直接用 search 查; "
                    "如果是称号 (如 先主, 宣王), 可以换成全名再查.")
        head = f"{q}: 共 {len(found)} 个人." if len(found) > 1 and not extra else f"{q}:"
        return head + "\n" + "\n".join(lines) + ("\n同名的人要分别 search (加上字, 官职或所属势力区分)." if len(found) > 1 else "")


    def _who_pinyin(self, name: str) -> str:
        """who 收到拼音 (汉语拼音或威妥玛拼音, 声调, 空格, 撇号, 大小写都不论) 时: 查 rag/alias.load_pinyin 得到中文写法,
        逐个按中文查, 拼音相同的不止一个写法时在开头说明. 例 (真实输出, 有删节):
            who("Zhou Tai") -> "Zhou Tai (拼音) 对应 2 个写法: 州泰, 周泰. 拼音相同的不止一人: 按问题里的其他线索
                                (所属势力, 官职, 事件) 判断是哪一位, 拿不准时在回答里说明有几位.
                                - 州泰 (传的首段 28-0035); 称呼: ...
                                - 周泰, 字幼平 (传的首段 55-0014); 称呼: ..."
        别名表外的人 (没有传, 如 张世平) 查不到, 提示直接 search."""
        names = load_pinyin().get(key(name), [])
        if not names:
            return (f"别名表里没有拼音是 {name} 的人 (表里只有有传的人和少数常被只称称号的人). "
                    "把名字译成中文 (想一想对应哪几个汉字) 再用 search 查.")
        lines: list[str] = []
        for n in names:
            extra = self._extra_line(n)
            lines += ([extra] if extra else []) + self._who_lines(load_extras()[n]["sgz"] if extra else n)
        head = f"{name} (拼音) 对应 {len(names)} 个写法: {', '.join(names)}."
        if len(names) > 1:
            head += " 拼音相同的不止一人: 按问题里的其他线索 (所属势力, 官职, 事件) 判断是哪一位, 拿不准时在回答里说明有几位."
        return head + "\n" + "\n".join(dict.fromkeys(lines))


# ---------- 模型看过的块 ----------

class Seen:
    """模型看过的块和它们的全局编号 (每回答一个问题新建一个).

    编号在一次回答里不变, 模型在回答里用 [n] 标出处, 引文核对再按编号找回这一块.

    属性:
        chunks  模型看过的块, 按编号顺序: chunks[n-1] 是编号 [n] 的块; 引文核对用它
        num     chunk id -> 编号 (从 1 起), 如 {"10-0023": 1, "10-0023n0": 2}
        notes   段落 id -> 它的裴注块 (Library.notes), 用来提示 "此段还有哪些注没看"; 不给就是 {} (不提示)
    """

    def __init__(self, notes: dict[str, list[dict]] | None = None) -> None:
        self.chunks: list[dict] = []
        self.num: dict[str, int] = {}
        self.notes = notes or {}

    def add(self, chunks: list[dict]) -> str:
        """登记一批块 (search / read 的结果), 返回交给模型的文字 (作为 role "tool" 消息的 content).

        登记: 没见过的块追加到 self.chunks, 编号 = 追加后的长度; 见过的沿用原编号, 不重复登记 (但照样输出).
        输出: 每块一段, 段与段之间空一行 ("\n\n"); chunks 为空时返回 "(没有结果)". 一段 =
            "[编号] " + rag.quotes.source_label(块)                    出处行
            + "\n" + 块的 text                                       原文 (繁体)
            + "\n【白话译文】" + 块的 trans                           只有白话切块 (有 "trans" 字段) 才有
            + "\n(此段还有裴注 id, id 没有显示, 需要时 read 这一段)"   见下
        裴注提示: 对块 sources 里的每个段落 id, 查 self.notes; 既不在这一批里, 也没登记过的注列出来.
        裴注块的 sources 是注 id, notes 里没有, 所以只有正文块会带提示.

        例 (真实数据, 第一次登记 10-0023, 原文和译文有删节):
            [1] 三国志·魏書·荀彧荀攸賈詡傳·荀彧·彧子惲 正文 (10-0023)
            子惲，嗣侯，官至虎賁中郎將。……詵弟顗，咸熙中爲司空。
            【白话译文】荀彧的儿子荀恽，继承侯爵，官至虎贲中郎将。……荀诜的弟弟荀顗，咸熙年间任司空。
            (此段还有裴注 10-0023n0, 10-0023n1, 10-0023n2.0, 10-0023n2.1 没有显示, 需要时 read 这一段)

        译文只拼在展示文字里, 登记的块 text 仍是原文: 引文核对 (rag.quotes.verify_quotes) 只认原文, 引了白话的会判成找不到."""
        parts = []
        for c in chunks:
            if c["id"] not in self.num:
                self.chunks.append(c)
                self.num[c["id"]] = len(self.chunks)
            trans = f"\n【白话译文】{c['trans']}" if c.get("trans") else ""
            shown = {x["id"] for x in chunks} | set(self.num)
            unseen = [n["id"] for src in c["sources"] for n in self.notes.get(src, []) if n["id"] not in shown]
            # 列出没看过的注, 让模型知道可以 read (只看到 10-0023 和 n2.0 时, 荀粲在没显示的 n2.1 里)
            hint = f"\n(此段还有裴注 {', '.join(unseen)} 没有显示, 需要时 read 这一段)" if unseen else ""
            parts.append(f"[{self.num[c['id']]}] {source_label(c)}\n{c['text']}{trans}{hint}")
        return "\n\n".join(parts) if parts else "(没有结果)"


def load_library(index: str = "trans", alias: str = ALIAS) -> Library:
    """读切块, 建检索器, 返回 Library.

    index   "trans" (默认) = 白话索引: 按白话译文检索, 给模型看原文 + 白话;
            "" = 原文索引, 按原文检索, 用来对照. 见 rag/retrieve.load_chunks.
    alias   别名扩展方式, 见 rag/retrieve.make_search ("" = 不扩展).
    检索加了锁: 用 bge-m3 编码查询不是线程安全的, 多个问题同时回答 (评测并发跑题, 网页多人同时提问) 时检索排队,
    检索本身很快 (每次约 0.06 秒), 模型请求照样并发."""
    chunks = load_chunks(index)
    search = make_search(chunks, index, alias)
    lock = threading.Lock()

    def locked_search(query: str, k: int, where: Where | None = None) -> list[ChunkDict]:
        with lock:
            return search(query, k, where)
    return Library(chunks, locked_search)
