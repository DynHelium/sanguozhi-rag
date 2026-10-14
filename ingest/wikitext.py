"""清洗维基文库的 wikitext: 拆链接, 渲染模板, 去标记, 处理校勘括号, 得到展示文本, 检索文本和校勘记录.
裴注的分离和分段在 ingest/sanguozhi.py; 这里只处理不含裴注的片段.

    uv run python -m tests.test_wikitext

    clean_fragment("（爾）〔以〕{{YL|太和三年|229年}}")
    -> Cleaned(display="以太和三年", search="以太和三年 229年",
               variants=[{"kind": "改", "adopted": "以", "other": "爾"}])     (Variant 见 rag/schema.py)

反引号里的是史料原文或 wikitext 原样, 保留原有标点.

数据来源
    zh.wikisource.org 的 `三國志/卷01` ... `三國志/卷65`, 经 MediaWiki API 取原始 wikitext (不是渲染后的 HTML),
    由 ingest/fetch_wikisource.py 下载到 data/sources/sanguozhi/raw/卷NN.wiki, 每卷的 revid 记在 manifest.json.
    维基文库本注明按中华书局点校本校对, 断句标点沿用点校本, 所以仓库里只放白文 (ingest/baiwen.py),
    标点版由使用者用 ingest.use_wikisource 按 revid 下载 (见 data/sources/README.md).

原则
    raw/ 永远保持下载原样. 录入错误写进 data/sources/sanguozhi/patches.yaml, 解析前由
    apply_patches() 套用. 好处: 可复现; 每处改动有理由可查; 上游修正后补丁会报错, 而不是悄悄失效.

解析步骤
    1. scan(): 切成"正文片段"与"最外层模板"交替的序列.
       apply_patches(): scan 之前修正录入错误.
    2. 清洗一个不含裴注的片段 -> 展示文本, 检索文本, 校勘记录.
       2a split_args(): 按顶层 | 切模板参数 (跳过嵌套 {{ }} 与 [[ ]] 里的 |).
       2b unwrap_links(): 拆链接, 在 scan 之前做.
       2c render_template(): 按模板名分派 (见下面的"模板对照").
          只决定"输出哪个参数"; 参数里嵌套的模板由之后的组装函数递归清洗.
       2d clean_markup(): 其余非模板标记 (见下面的"非模板标记").
       2e apply_brackets(): 校勘括号 -> 改正后的读法 + 校勘记录.
       clean_fragment(): 把以上串起来, 递归清洗嵌套模板.
       裴注模板 * / *s / *e 不归第 2 步管: 第 3 步先把正文和注文分开, 再分别交给第 2 步清洗.
    3-4. 裴注分离, 按标题分段, 输出 JSONL: 见 ingest/sanguozhi.py.

模板对照 (按 manifest.json 记录的各卷 revid 版本, 65 卷全部统计; 含嵌套; 定义查自 zh.wikisource 的 Template: 页面)
    处理方式: "取1" = 输出第 1 个参数 (递归清洗); "丢" = 整个丢弃.

    裴注 (第 3 步处理; 第 2 步遇到会 raise ValueError)
    {{*|...}}         约 1760  短裴注, 插入位置即所注正文位置. 注文开头多为引书名, 如 `《曹瞞傳》曰`, `臣松之案`.
    {{*s}} ... {{*e}}  13 对 长裴注: 只是开始/结束标记, 注文在两个标记之间的正文里, 可跨段落, 内含 quote.
                             集中在卷30 (10 对), 卷19, 卷20. 第 3 步同时认这两种写法.

    有语义, 按模板决定取哪个参数
    {{YL|太和三年|229年}} 约 790  年号链接. 只有 262 处带第 2 参数 (公元年, 维基编辑所加, 用于消歧:
                             太和也是东晋年号); 527 处只有年号, 公元年由 Module:Year Link/data 自动查.
                             方案 A: 展示文本只取年号; 检索文本在年号后附公元年 (有第 2 参数才附).
    {{另|底本|另本}}      29   异文. 取1; 另本写法记进校勘 metadata. {{別|...}} 2 处, 是"另"的重定向, 同样处理.
    {{校|訛字|正字}}      3    维基编辑的校正: 页面显示第 1 参数 (原文讹字), 悬停提示 `應校正為` 第 2 参数.
                             取第 2 参数 (正字), 讹字记进校勘 metadata. 例: `{{校|潘陽|鄱陽}}`.
    {{!|𩅦}} {{!|𢢼|〈忄剽〉}} 18  僻字: 第 1 参数是字, 第 2 参数是部件描述. 取1. `{{僻字|𨪕|左金右芻}}` 1 处同.
    {{annotate|音飯。}}   1    小字音注 (卷23). 取1; 是否裴注待核.

    纯排版, 取1
    {{ProperNoun|張遼}}   2550 专名线 (只在卷46, 17, 40 大量出现).
    {{WavyBookMark|...}} 88   书名线, 内部常嵌 [[链接]].
    {{quote|...}}        164  引文块 (书信, 表, 诏), 内容是正文或注文的一部分, 保留; 可视作段落边界.
    {{red|...}} {{blue|...}} {{green|...}}  59/15/1  文字着色, 多用于诏令.
    {{ul|倭人}}          115  下划线, 只在卷30.
    {{-|臣}}             8    小字 (表文里的 `臣`), 重定向到 Template:Small.

    丢 (按前缀匹配, 见 render_template)
    {{gap}}              140  段首缩进空白.
    {{header}} {{header2|...}} {{footer}} {{Novel-f|...}} {{PD-old}} {{西晉作品}} {{Textquality|...}}
    {{wikipedia|曹芳}}   页眉页脚, 导航, 版权声明, 维基百科链接.

非模板标记
    [[w:孫堅|孫堅]] [[作者:...|...]] [[曹瞞傳]]   链接取显示文字 (有 | 取后半, 否则取全部).
    [[Category:...]] [[category:...]]         丢.
    -{後}-            1699 繁简转换保护符, 去掉 -{ }- 保留字.
    '''粗体'''        130  去掉引号.
    <br> <onlyinclude> <sub> <u> <small> <blockquote>   去标签留内容; <!-- 注释 --> 与 __FORCETOC__ 丢.
    行首 :           195  缩进 (多见于 `評曰`), 去掉冒号.
    全角空格 U+3000   2176 段首缩进, 去掉.
    标题 =...= 到 ====...====   1 级是分组 (`=武帝諸子=`, `=皇后=`), 2 级多为传主 (`==張遼==`), 3, 4 级为附传.
                     是段落 headings 字段的来源, 由 ingest/sanguozhi.py 处理. 卷首有 <sub> 包着的目录行, 丢.

校勘括号 (正文里的纯文字, 不是模板, 源自中华书局点校本体例)
    `（爾）〔以〕`   `（…）` 151 处 = 应删的衍字/误字; `〔…〕` 40 处 = 应补/改正的字; 二者成对 13 处.
                     分布在 47 卷. 清洗后文本用改正后的读法: 删去 `（）` 及其内容, `〔〕` 去括号留内容;
                     删了什么, 补了什么记进校勘 metadata.
                     例外: 卷01 裴注开头 `〔《曹瞞傳》曰：〕` 是整理者补的引书名, 按同样规则保留文字.
                     注意这两种括号是全角的 `（）〔〕`, 与说明文字里的半角 () 不是一回事.

已知数据问题 (录入错误, 异体字, 繁简混杂, 待核的几处) 见 data/sources/README.md.
"""
import re
from dataclasses import dataclass, field
from typing import TypedDict

from rag.schema import Variant


@dataclass
class Template:
    """一个最外层模板. raw 是 {{ 与 }} 之间的原始内容 (不含这两对括号),
    嵌套在内的子模板原样保留, 例如 `*|{{WavyBookMark|呉書}}曰：……`.
    模板名是 raw 里第一个 "|" 之前的部分 ("*", "YL", "另" ...), 由第 2 步解析."""

    raw: str


def scan(text: str) -> list[str | Template]:
    """逐字扫描, 按 {{ }} 的嵌套深度切分出正文片段和最外层模板.

    为什么只切最外层: 裴注 {{*|...}} 里嵌着 {{WavyBookMark|...}} 等子模板, 先整体切出一条
    注文, 内部结构留给第 2 步递归处理. 为什么不用正则: 正则处理嵌套括号很别扭,
    按深度扫描既简单, 又能在括号不配对时报出位置 (卷44 的录入错误就是这样发现的).

    不变量: 把结果拼回去 (Template 还原为 "{{" + raw + "}}") 与输入一字不差,
    65 卷全部成立.

    - 只认成对的 "{{" 和 "}}"; 单个 "{" "}" (如 `-{後}-`) 是普通正文.
    - 返回列表里不出现空字符串.
    - 括号不配对 (有 "{{" 没闭合, 或多出 "}}") 时 raise ValueError, 报出位置.

    例: scan("甲{{YL|太和三年|229年}}乙") == ["甲", Template("YL|太和三年|229年"), "乙"]
    """
    prev = 0
    curr = 0
    stack = []
    res = []
    while curr < len(text):
        if text[curr:curr+2] == "{{":
            stack.append(curr)
            if len(stack) == 1:
                res.append(text[prev:curr])
                prev = curr + 2
            curr += 2
        elif text[curr:curr+2] == "}}":
            if not stack:
                raise ValueError(f"Find Invalid Bracket at {curr}")
            stack.pop()
            if not stack:
                res.append(Template(raw=text[prev:curr]))
                prev = curr + 2
            curr += 2
        else:
            curr += 1

    if prev < len(text):
        res.append(text[prev:])

    if stack:
        raise ValueError(f"Find Invalid Bracket at {stack[0]}")

    return [s for s in res if s]


class Patch(TypedDict):
    """patches.yaml 的一条补丁."""
    file: str     # 卷文件名, 如 "卷44.wiki"
    find: str     # 原文片段, 在该卷里恰好出现一次 (否则 apply_patches 报错)
    replace: str  # 替换成什么
    reason: str   # 为什么改


def apply_patches(filename: str, text: str, patches: list[Patch]) -> str:
    """把 patches 里 file == filename 的补丁逐条套到 text 上, 返回修正后的文本.

    patches 来自 data/sources/sanguozhi/patches.yaml, 每条是
        {"file": "卷44.wiki", "find": 原文片段, "replace": 替换文本, "reason": 为什么改}
    raw/ 文件保持原样, 修正只在这里发生, 见模块 docstring 的"原则".

    - 每条补丁的 find 在当前 text 里恰好出现 1 次才替换, 否则 raise ValueError, 错误信息带上 filename 和 find.
      这样上游改动让补丁失效时会大声报错, 而不是悄悄漏改或改错位置.
    - 与 filename 无关的补丁跳过.
    - 不修改传入的 text 和 patches, 返回新字符串.
    """
    for patch in patches:
        if patch['file'] == filename:
            try:
                a, b = text.split(patch['find'])
            except ValueError:
                raise ValueError(f"Text {patch['find']} does not appear 1 and only 1 time in the text in {filename}")
            text = a + patch['replace'] + b
    return text


def split_args(raw: str) -> list[str]:
    """把模板内容 Template.raw 按"顶层"的 | 切成 [模板名, 参数1, 参数2, ...].

    为什么不直接 raw.split("|"): 参数里常嵌着带 | 的东西:
        "WavyBookMark|[[王沈魏書|魏書]]"   链接 [[目标|显示文字]] 里的 |
        "blue|{{-|臣}}海濵孤微"            嵌套模板里的 |
    这些 | 属于内层, 不是分隔符. 所以和 scan() 一样按括号深度扫描: {{ 与 [[ 进一层, }} 与 ]] 退一层,
    只有深度为 0 时遇到的 | 才是分隔符.

    - 不去空白, 不丢空参数: 原样切分, "|".join(结果) 等于 raw (与 scan 同样的往返不变量).
      去空白交给调用方, 例如 header2 的参数带换行和 "title = ..." 这种命名参数, 用不到也就不必处理.
    - 至少返回 1 个元素 (模板名). "gap" -> ["gap"]; "quote|" -> ["quote", ""].
    - 括号不配对 (raw 里 [[ 没闭合, 多出 ]] 等) 时 raise ValueError. scan 已保证 {{ }} 配对,
      [[ ]] 在这里检查.

    例: split_args("YL|太和三年|229年") == ["YL", "太和三年", "229年"]
    """
    stack = []
    prev, curr = 0, 0
    l = len(raw)
    res = []
    while curr < l:
        if (tmp := raw[curr:curr+2]) in {'[[', '{{'}:
            stack.append(tmp)
            curr += 2
        elif tmp in {']]', '}}'}:
            if not stack:
                raise ValueError("Bracket Dismatch Found.")
            left = stack.pop()
            if left == '[[' and tmp == ']]' or left == '{{' and tmp == '}}':
                curr += 2
            else:
                raise ValueError("Bracket Dismatch Found.")
        elif raw[curr] == "|" and not stack:
            res.append(raw[prev:curr])
            curr += 1
            prev = curr
        else:
            curr += 1

    if stack:
        raise ValueError("Bracket Dismatch Found.")

    res.append(raw[prev:])

    return res

def unwrap_links(text: str) -> str:
    """把 wikitext 链接换成显示文字. 在 scan() 之前对整段文本做.

    规则:
    - [[目标|显示文字]] -> 显示文字. 例: [[w:孫堅|孫堅]] -> 孫堅; [[王沈魏書|魏書]] -> 魏書.
    - [[目标]] -> 目标. 例: [[曹瞞傳]] -> 曹瞞傳.
    - [[Category:...]] 和 [[category:...]] -> 整个删掉 (分类标签, 不是正文), 包括带排序键的
      [[category:三國志|62]] (排序键不是显示文字; 全书 50 余卷卷末都有, 曾漏成正文里的 "62").
    - 显示文字里的模板原样保留, 交给后面的 scan(). 例:
      [[w:孫堅|{{ProperNoun|孫堅}}]] -> {{ProperNoun|孫堅}}.
    - 链接外面的文字 (包括模板里的链接以外部分) 一字不动.

    为什么先拆链接: 卷46 有 97 处链接包着模板, 先 scan 的话链接会被模板切成两半.
    全书链接目标段 (| 之前) 从不含模板或嵌套链接, 所以只需找 [[ 与配对的 ]].
    """
    def _replace(s: re.Match[str]) -> str:
        inner = s.group(1)
        if re.match(r"category:", inner, re.I):  # 先判分类: [[category:三國志|62]] 的 |62 是排序键
            return ""
        for i, ch in enumerate(inner):
            if ch == '|':
                return inner[i+1:]
        return inner

    return re.sub(r"\[\[(.*?)\]\]", _replace, text)


@dataclass
class Rendered:
    """render_template() 的结果.

    display: 展示文本, 用于回答里引用原文和逐字校验.
    search:  检索文本, 只在年号处比 display 多出公元年 (方案 A), 其余与 display 相同.
    variant: 校勘记录, 只有"校"和"另/別"才有, 其余为 None. 格式:
             {"kind": "校" 或 "另", "adopted": 采用的文字, "other": 没采用的文字}
    display / search 里可能还嵌着模板 (如 blue 的参数里有 {{-|臣}}), 由调用方继续清洗.
    """

    display: str
    search: str
    variant: Variant | None = None


def render_template(args: list[str]) -> Rendered:
    """按模板名 (args[0], 去掉首尾空白后) 决定输出什么. args 来自 split_args().

    规则 (即模块 docstring 的模板对照表):
    - 取1: ProperNoun, WavyBookMark, quote, red, blue, green, ul, -, annotate, !, 僻字
           display = search = 第 1 个参数; 没有参数 (如 {{ProperNoun}}) 时输出 "".
    - YL:  display = 年号 (第 1 参数); 有第 2 参数时 search = 年号 + " " + 公元年,
           例: ["YL", "太和三年", "229年"] -> display "太和三年", search "太和三年 229年";
           没有第 2 参数时 search = display.
    - 另, 別: 取第 1 参数 (底本); variant = {"kind": "另", "adopted": 第 1 参数, "other": 第 2 参数}.
    - 校:  取第 2 参数 (正字); variant = {"kind": "校", "adopted": 第 2 参数, "other": 第 1 参数}.
    - 丢 (按前缀匹配, 范围比表里宽): 名字以 header / gap / footer / Novel-f / PD-old / Textquality / wikipedia 开头,
           或以 "作品" 结尾 (如 西晉作品) -> "".
    - 裴注 *, *s, *e: raise ValueError. 它们归第 3 步, 走到这里说明调用顺序错了.
    - 未知模板名: raise ValueError, 报出模板名. 报错而不是悄悄丢掉: 上游新增模板时能及时发现.

    参数本身不去空白 (quote 的内容以全角空格开头, 留给后面的步骤清理).
    """
    template = args[0].strip()
    if template in {"ProperNoun", "WavyBookMark", "quote", "red", "blue", "green", "ul", "-", "annotate", "!", "僻字"}:
        display = args[1] if len(args) >= 2 else ""
        return Rendered(display=display, search=display)
    if template == "YL":
        display = args[1]
        search = display if len(args) == 2 else args[1] + " " + args[2]
        return Rendered(display=display, search=search)
    if template in {"另", "別"}:
        return Rendered(display=args[1], search=args[1], variant={'kind': "另", "adopted": args[1], "other": args[2]})
    if template == "校":
        return Rendered(display=args[2], search=args[2], variant={'kind': "校", "adopted": args[2], "other": args[1]})
    if template.startswith("header") or template.endswith("作品") or template.startswith(("gap", "footer", "Novel-f", "PD-old", "Textquality", "wikipedia")):
        return Rendered(display="", search="")
    raise ValueError("Unknown Template: " + template)


_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_MAGIC = re.compile(r"__[A-Z]+__")
_LANG_GUARD = re.compile(r"-\{(?:[A-Za-z;]+\|)?(.*?)\}-", re.S)  # -{後}- 与带标志位的 -{A|出}-
_LANG_VARIANTS = re.compile(r"-\{zh:([^;}]*);[^}]*\}-")  # -{zh:後;zh-hant:後;zh-hans:后}- 取 zh 默认值


def clean_markup(text: str) -> str:
    """第 2d 步: 清掉模板以外的标记. 输入是 scan() 切出的正文片段 (不含模板).

    - <!-- 注释 --> 与 __FORCETOC__ / __TOC__ 删掉.
    - <br> 换成换行; 其余 HTML 标签 (<onlyinclude> <u> <small> <blockquote> <sub>) 去标签留内容.
    - -{後}- -> 後 (繁简转换保护符); 带标志位的 -{A|出}- -> 出 (卷30 一处);
      分语言写法 -{zh:後;zh-hant:後;zh-hans:后}- -> 後 (取 zh 默认值, 卷20 一处).
    - '''粗体''' 去引号 (全书没有 ''斜体'').
    - 全角空格 U+3000 删掉 (古文正文本身不用空格, 这些都是缩进或排版).
    行首 : 缩进和 == 标题 == 是行级结构, 由 ingest/sanguozhi.py 按行处理, 这里不管.
    """
    text = _COMMENT.sub("", text)
    text = _MAGIC.sub("", text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = _TAG.sub("", text)
    text = _LANG_VARIANTS.sub(r"\1", text)
    text = _LANG_GUARD.sub(r"\1", text)
    text = text.replace("'''", "")
    return text.replace("\u3000", "")


# 括号里的内容不跨行, 不含另一对括号, 也不含私用区字符 (ingest/sanguozhi.py 用它标记裴注位置,
# 所以不会被当成衍字删掉).
_PAIR = re.compile(r"（([^（）〔〕\n\ue000-\uf8ff]{1,30})）〔([^（）〔〕\n\ue000-\uf8ff]{1,30})〕")
_DEL = re.compile(r"（([^（）〔〕\n\ue000-\uf8ff]{1,30})）")
_ADD = re.compile(r"〔([^（）〔〕\n\ue000-\uf8ff]{1,30})〕")
_ADD_ASCII = re.compile(r"\[([^\[\]（）〔〕\n\ue000-\uf8ff|]{1,30})\]")  # 卷33/35/38/41 有录入者用半角 [] 当补字, 待核


def apply_brackets(text: str) -> tuple[str, list[Variant]]:
    """第 2e 步: 中华书局点校本的校勘括号 -> 改正后的读法, 并返回校勘记录.

    `（爾）〔以〕` 成对 = 改字: 采用"以", 记 {"kind": "改", "adopted": "以", "other": "爾"}.
    `（祀）` 单独 = 衍字删去: 记 {"kind": "删", "adopted": "", "other": "祀"}.
    `〔歸降〕` 单独 = 补字: 记 {"kind": "补", "adopted": "歸降", "other": ""}.
    半角 `[六]` 也按补字处理: 卷33, 35, 38, 41 共 9 处, 看来是录入者用半角方括号代替 `〔〕`, 待与点校本核对.
    (调用前 unwrap_links 已拆掉所有 [[链接]], 所以这里的单个方括号不会是链接.)
    在模板渲染之后做: 有的括号里包着模板 (如 `（{{ProperNoun|辟}}、…）`).
    记录格式与 render_template 的 variant 一致, 顺序按出现先后.
    """
    records: list[Variant] = []

    def pair(m: re.Match[str]) -> str:
        records.append({"kind": "改", "adopted": m.group(2), "other": m.group(1)})
        return m.group(2)

    def delete(m: re.Match[str]) -> str:
        records.append({"kind": "删", "adopted": "", "other": m.group(1)})
        return ""

    def add(m: re.Match[str]) -> str:
        records.append({"kind": "补", "adopted": m.group(1), "other": ""})
        return m.group(1)

    text = _PAIR.sub(pair, text)
    text = _DEL.sub(delete, text)
    text = _ADD.sub(add, text)
    text = _ADD_ASCII.sub(add, text)
    return text, records


@dataclass
class Cleaned:
    """clean_fragment() 的结果. display / search 含义同 Rendered; variants 是这一片段的全部校勘记录
    (render_template 的"校", "另" 和 apply_brackets 的"改", "删", "补"), 按出现先后."""

    display: str
    search: str
    variants: list[Variant] = field(default_factory=list)


_NOTE_TEMPLATES = {"*", "*s", "*e"}


def _render(wikitext: str, variants: list[Variant]) -> tuple[str, str]:
    """递归渲染: 正文片段走 clean_markup, 模板走 render_template, 模板输出里若还有模板就再渲染一遍."""
    display, search = [], []
    for part in scan(wikitext):
        if isinstance(part, str):
            t = clean_markup(part)
            display.append(t)
            search.append(t)
            continue
        args = split_args(part.raw)
        if args[0].strip() in _NOTE_TEMPLATES:
            raise ValueError(f"裴注模板应由 ingest/sanguozhi.py 先分离: {{{{{part.raw[:30]}")
        r = render_template(args)
        if r.variant:
            variants.append(r.variant)
        d, s = _render(r.display, variants)
        if r.search != r.display:  # 只有带公元年的 YL 才不同, 检索文本单独渲染, 校勘记录不重复收
            _, s = _render(r.search, [])
        display.append(d)
        search.append(s)
    return "".join(display), "".join(search)


def clean_fragment(wikitext: str) -> Cleaned:
    """把一段不含裴注模板的 wikitext 清洗成展示文本, 检索文本和校勘记录.

    顺序: (调用方已做 apply_patches, unwrap_links) -> 递归渲染模板与标记 -> 校勘括号.
    校勘括号对 display 和 search 各做一次 (两者只差年号后的公元年, 括号处理结果一致), 记录取 display 的.
    裴注模板 (* / *s / *e) 会 raise ValueError: 它们由 ingest/sanguozhi.py 事先分离出去.
    """
    variants: list[Variant] = []
    display, search = _render(wikitext, variants)
    display, bracket_records = apply_brackets(display)
    search, _ = apply_brackets(search)
    return Cleaned(display, search, variants + bracket_records)
