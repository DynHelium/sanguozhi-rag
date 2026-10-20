# 评测结果

## RAG 效果评测（main 分支 README 和 `docs/EVALUATION.md` 的数字来自这里）

50 道题中英文各一遍（`eval/questions-rag50.jsonl`、`eval/questions-rag50-en.jsonl`），DeepSeek V4.1 Flash（`ds`，思考 low）和
Claude Haiku 5.5（`hk`，思考 medium）各分五组回答，设计见 `docs/EVALUATION.md`：

| 文件 | 内容 | 脚本 |
|---|---|---|
| `rag-<模型>-<组>-rag50.jsonl`、`rag-<模型>-<组>-rag50-en.jsonl` | 回答：组为 `closed`（闭卷）、`static`（检索一次）、`oracle`（gold 原文）、`agentic`（只给工具）、`production`（网页配置）。每行有回答、模型看过的块（`retrieved`）、工具调用（`rounds`）、用量和花费（`usage`）、配置（`group`、`off`、`system_sha` 等） | `eval.run_rag` |
| `rag-<模型>.judged.jsonl` | 判决：每份回答一行，`group`、`lang`、清单逐项的标记（`must` / `must_not`）、编造的史料（`fabricated`）、清单外的事实错误（`extra_errors`）、查不到的确定说法（`unsupported`）、由代码算出的 `verdict` | `eval.judge_v2` |

### 判卷

- 评委：Claude Sonnet（判决行里没有单独的模型字段），按 `eval/judge_prompt_v2.md` 判，可以用 `eval.find` 检索原文核对。
- 同一题同一语言的五组回答放进一个包（一包 2 题），打包前统一格式（去掉 `[n]` 编号，引号换成同一种），每题的标签 g1–g5 随机分配。
  判决行里的 `blind_tag` 是这份回答当时的标签。
- 评委对清单逐项标记，判定（`verdict`）由 `eval/judge_v2.py` 按规则算出：核心项没答到或说错、出现清单列出的错误说法 → 错
  （标了 `core_partial` 的核心项只是没答到时算部分，用于 i-02、f-22、p-01）；
  题目明确要求的其他项漏了或错了、背景细节说错了、有编造的史料或和原文矛盾的事实 → 部分；其余 → 对。
- 事实清单是 `eval/checklists-rag50.jsonl`，每项标了是否核心（`core`）、是否题目明确要求（`asked`）；第一轮判卷后改过一次，
  改动和理由记在 `eval/checklists-rag50.changes.md`。
- 评委标为拿不准的写在 `uncertain`（DeepSeek 12 条，Haiku 24 条），判定按评委的最佳判断计。

### 复现

`eval.run_rag` 的结果写在 `data/processed/eval_runs/answers/`，判卷脚本也从那里读；要重判这里的文件，先把它们复制过去。

```bash
uv run python -m eval.run_rag --group closed --models deepseek-v4.1-flash --effort low \
    --questions eval/questions-rag50.jsonl --name rag-ds-closed-rag50        # 每组 x 每个题集各跑一次
LLM_BACKEND=anthropic uv run python -m eval.run_rag --group production --models claude-haiku-5-5 --effort medium \
    --questions eval/questions-rag50-en.jsonl --name rag-hk-production-rag50-en
uv run python -m eval.judge_v2 prep rag-ds     # 五组回答打包 -> data/processed/eval_runs/judge/rag-ds/batchNN.json
# 评委按 eval/judge_prompt_v2.md 逐包判, 写成同名的 batchNN.out.jsonl (这一步另外调用模型, 脚本本身不调)
uv run python -m eval.judge_v2 merge rag-ds    # 按清单算判定 -> answers/rag-ds.judged.jsonl, 打印各组 对 / 部分 / 错
```

## 较早的一轮评测（已被上面的评测取代）

下面这些文件来自较早的一轮评测：几种配置的 prompt 不同、判卷标准把引文和出处计入对错，没有控制变量，结论以上面的评测为准。
五种配置 × 六套题（`holdout` 留出题 40、`reason` 推断 / 演义 / 官职地理 30、`facts` 事实级错误 24、`persons` 人物 9、`en` 英文 103，同样的 103 道中文题改写成英文；另有 `en-dev` 较早的 20 道英文开发题），每种配置一个回答文件和一个判卷文件：

| 文件 | 配置 | 脚本 |
|---|---|---|
| `agent-<题集>.jsonl` | agent（默认） | `eval.run_agent` |
| `agent-off-<题集>.jsonl` | agent 关掉代码介入（自查、引文修正、编号修正、提示再查） | `eval.run_agent --off nudge nudge_notfound review quote_fix cite_fix` |
| `single-<题集>.jsonl` | 检索一次：原问题检索一次，前 10 组放进上下文，作答一次 | `eval.run_single` |
| `single-nocontext-<题集>.jsonl` | 和检索一次同一个 prompt，资料一栏写 "(没有资料)" | `eval.run_single --no-context` |
| `closed-<题集>.jsonl` | 闭卷：不给工具和资料，prompt 简短、允许凭记忆作答 | `eval.run_closed` |
| `<配置>-<题集>.judged.jsonl` | 判卷结果：每题 对 / 部分 / 错，错在哪（`errors`），原文查不到的说法（`unsupported`），一句说明 | |
| `overrides.jsonl` | 作者改判的记录：哪一组哪一题，从什么改成什么，为什么 | |
| `backends-<后端>-probe62.jsonl` | 两种后端对照：62 题，`haiku-low-nochange`（Haiku，low，三项改动之前）、`haiku-low`、`haiku-medium`、`haiku-high`、`deepseek`（low） | `eval.run_agent`（Haiku 在 `.env` 设 `LLM_BACKEND=anthropic`） |
| `backends-<后端>-en21.jsonl` | 英文 21 题（含三道拼音同音题），较早的 prompt（P8）：`haiku-medium`、`deepseek` | 同上 |
| `prompt-<p8/p9>-<后端>-holdout.jsonl` | 留出题 40 道，较早的 prompt（P8，分事实 / 推断 / 背景写）和现在的 prompt（P9）× DeepSeek（low）/ Haiku（medium）；四组同包盲判，判决行多一个 `style` 字段（自然 / 略僵 / 僵硬） | `eval.run_agent` |
| `prompt-p9-<后端>-new4.jsonl`、`prompt-p9-<后端>-en21.jsonl` | 现在的 prompt：新题 4 道；英文 21 道（含三道同音题，带 lang_fix） | 同上 |
| `human-<p8/p9>-<后端>.jsonl` | 按人平时的问法提问的 16 道开放题（`eval/questions-human.jsonl`、`eval/questions-human-x.jsonl`），两种 prompt × 两种后端；四组同包盲判。没有标准答案，判决行写每个具体说法核对的结果：`verdict`、`errors`、`missing`（漏掉的关键记载）、`novel_misjudged`（把有记载的说成没有或出自演义）、`style` / `style_note` | 同上 |
| `recheck-nocite30.judged.jsonl` | 抽查一：主对照 30 题去掉 [n] 后重判（见下面"判卷的两项抽查"），判决行多 `group`（closed / single / agent）和 `blind_tag` | `eval.recheck nocite` |
| `recheck-cross-gpt-6.1-sol.jsonl` | 抽查二：后端对比 20 题由 OpenAI 的 GPT-6.1 Sol 重判，判决行多 `group`（deepseek / haiku）、`blind_tag`、`call_cost` / `call_usage`（这一题那次调用的花费和 token，两份回答共用） | `eval.recheck cross` |

回答文件的每行是脚本的结果行：问题、回答、模型看过的原文块（`retrieved`）、工具调用（`rounds`）、引文核对（`quotes` / `cites`）、用时、
token 数和花费（`usage`，美元，OpenRouter 返回的）、配置。

### 配置

- 模型：DeepSeek V4.1 Flash（`deepseek/deepseek-v4.1-flash`），经 OpenRouter，思考 low；服务商 InferenceNet 优先、GMICloud 备用。
- agent：main 分支的 `rag/agent.py`（结果行里的 `system_sha` / `review_sha` 是 system prompt 和自查指令全文的哈希）；白话索引，别名扩展 route2，自查 layered。
- 每种配置每套题跑一遍。闭卷有 5 题三次都没给出回答（推理用完 8192 token 的输出上限），判卷文件里按错计（`judge` 为 `code: empty reply`）。
- 两种后端对照的 Haiku 各组：Claude Haiku 5.5（`claude-haiku-5-5`），Claude 原生接口（`rag/claude_llm.py`），花费按 Claude 的价格算；五组 62 题的回答合成一包盲判（g1–g5），英文 21 题两组各自判。`haiku-low-nochange` 里 e-h-15 至 e-h-23 这 9 题是在三项改动写进代码之前跑的（`off` 里没有 agg_nudge / pinyin_who），其余 53 题是之后用 `--off agg_nudge pinyin_who` 跑的，两者配置相同。
- P8 和 P9（P8 是较早的分段式 prompt 的最后一版；五种配置的主对照用的是同一种分段写法的更早一版，后来又加过几条规则）：`rag/agent.py` 的 system prompt 从分段的写法（P8）改成按原则写（P9，两种后端共用规则加各自的最后一段），同时 `who` 给出传末陈寿的评、英文直引号里的文言也核对、"没找到" 的提示改为有针对性的核实并默认开。留出题、新题 4 道和真人问法题的 P9 回答，是在把作答语言那一句挪回 prompt 最后一行之前跑的（中文问题只差这一处顺序）；英文 21 题是挪回之后、加了 lang_fix 再跑的。P8 的真人问法回答里，hx 开头的 8 道是在改动前的代码上补跑的。
- 结果行里的 `sec` 是并发跑题时记的，含排队等待，不代表单题用时；单题用时要用 `eval.bench_latency` 逐题跑。

### 判卷

- 判卷模型 Claude Sonnet（`judge: "claude-sonnet"`），按 `eval/judge_prompt.md` 对照原文判，可以用 `eval.find` 查原文；题目自带的 `judge` 字段说明这一题怎么判。
- 标签打乱：每道题的闭卷、检索一次、agent 三份回答打乱顺序、换成随机标签（g1 / g2 / g3）放进同一个包，
  两组消融（agent-off、single-nocontext）也照此合包；英文 103 题（en）是五种配置的回答合成一包；en-dev 沿用较早一轮的判卷（检索一次和 agent 合包，两组消融合包，闭卷单独判）；判完再按标签还原到各配置。
- 判卷人拿不准的写进 `uncertain`。三组对照里的 46 条由作者逐条看过，并定了几条边界规则（之后的几组按同样的规则判）：
  - 答到了题目的核心，但夹带编造或张冠李戴的引文、出处、细节 → 部分；错误非常小（一个字、出处标错但引文和事实都对）→ 对。
  - 《吴录》"八绝"答全八人即为对；原文没有答案的题只说 "没有记载"、没交代裴注里的说法 → 部分；回答被截断但已给出核心结论 → 对。
  - 按这条规则，另由一个判卷模型盲复查了全部判错的回答。
- 改判的判决行保留原判（`verdict_ai`）和改判原因（`override`），`overrides.jsonl` 汇总了全部改判。

### 判卷的两项抽查

标签打乱不等于完全的盲判，后端对比的评委也和 Haiku 同属一家，所以做了两项抽查（`eval/recheck.py`）：

- **去掉 [n] 后重判**：主对照里闭卷回答没有 [n] 编号，一次检索和 agent 有，判卷人认得出闭卷。在三种配置都有回答的题里按题集比例抽 30 题（留出 12、推断演义官职 8、事实 7、人物 3；闭卷 5 道空回答的题不在池里），
  去掉一次检索和 agent 回答里的 [n]，三份回答重新打乱标签，由 Claude Sonnet 按同样的标准重判。90 份里 83 份和公开的最终判决一致
  （闭卷 27/30、检索一次 27/30、agent 29/30）。只去掉了编号，别的写法差异还在（闭卷用 “” 引原文、另外两种用「」），所以这只说明 [n] 编号本身没有左右判决。对 / 部分 / 错：闭卷 19/9/2（原 16/12/2），检索一次 22/3/5（原 23/4/3），agent 30/0/0（原 29/1/0）。
  和作者改判前的 AI 原判比只有 78 份一致：主对照最初判卷时还没有"核心对、夹带编造细节判部分"这条边界规则，闭卷的不少"错"后来由作者改成了"部分"。
- **第三家评委**：从 `questions-probe62` 抽 20 题，DeepSeek 和 Haiku（medium）的回答一题一次调用、标签打乱，交给 OpenAI 的 GPT-6.1 Sol（思考 medium，经 OpenRouter）判（`judge: "openai/gpt-6.1-sol"`），
  共花 $0.42。它不能检索全书，所以附上 gold 段落和每份回答 [n] 引到的原文块，并说明所附原文里查不到的事只记 `unsupported`、不因此判错。
  40 份里 37 份和 Sonnet 一致；完全正确 DeepSeek 19（Sonnet 20），Haiku 17（Sonnet 19）。三处分歧都是 GPT 判得更严（e-h-35 DeepSeek、e-h-22 和 h-30 Haiku）。

### 复现

`eval.run_agent` 等脚本的结果写在 `data/processed/eval_runs/answers/<名>.jsonl`（`eval/common.py` 的 `OUT_DIR`），判卷脚本也从那里读；
要重判这里的文件，先把它复制过去（如复制成 `answers/ho1.jsonl`）。

```bash
uv run python -m eval.run_agent --models deepseek-v4.1-flash --effort low \
    --questions eval/questions-holdout.jsonl --name ho1
uv run python -m eval.ai_judge prep ho1 --questions eval/questions-holdout.jsonl   # 打包: data/processed/eval_runs/judge/ho1/batchNN.json
# 判卷模型按 eval/judge_prompt.md 逐包判, 写成同名的 batchNN.out.jsonl (这一步另外调用模型, 脚本本身不调)
uv run python -m eval.ai_judge merge ho1 --judge claude-sonnet                     # 汇总对 / 部分 / 错
uv run python -m eval.judge --name ho1                                              # 或者在浏览器里人工判
```
