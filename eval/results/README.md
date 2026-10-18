# 评测结果

main 分支 README "Answer quality"（中文版 "回答效果"）里的数字都来自这里的文件。五种配置 × 六套题（`holdout` 留出题 40、
`reason` 推断 / 演义 / 官职地理 30、`facts` 事实级错误 24、`persons` 人物 9、`en` 英文 103，同样的 103 道中文题改写成英文；另有 `en-dev` 较早的 20 道英文开发题），每种配置一个回答文件和一个判卷文件：

| 文件 | 配置 | 脚本 |
|---|---|---|
| `agent-<题集>.jsonl` | agent（默认） | `eval.run_agent` |
| `agent-off-<题集>.jsonl` | agent 关掉代码介入（自查、引文修正、编号修正、提示再查） | `eval.run_agent --off nudge nudge_notfound review quote_fix cite_fix` |
| `single-<题集>.jsonl` | 检索一次：原问题检索一次，前 10 组放进上下文，作答一次 | `eval.run_single` |
| `single-nocontext-<题集>.jsonl` | 和检索一次同一个 prompt，资料一栏写 "(没有资料)" | `eval.run_single --no-context` |
| `closed-<题集>.jsonl` | 闭卷：不给工具和资料，prompt 简短、允许凭记忆作答 | `eval.run_closed` |
| `<配置>-<题集>.judged.jsonl` | 判卷结果：每题 对 / 部分 / 错，错在哪（`errors`），原文查不到的说法（`unsupported`），一句说明 | |
| `overrides.jsonl` | 作者改判的记录：哪一组哪一题，从什么改成什么，为什么 | |
| `backends-<后端>-probe62.jsonl` | 两种后端对照（README "Two backends"）：62 题，`haiku-low-nochange`（Haiku，low，三项改动之前）、`haiku-low`、`haiku-medium`、`haiku-high`、`deepseek`（low） | `eval.run_agent`（Haiku 在 `.env` 设 `LLM_BACKEND=anthropic`） |
| `backends-<后端>-en21.jsonl` | 英文 21 题（含三道拼音同音题），最终 prompt：`haiku-medium`、`deepseek` | 同上 |

回答文件的每行是脚本的结果行：问题、回答、模型看过的原文块（`retrieved`）、工具调用（`rounds`）、引文核对（`quotes` / `cites`）、用时、
token 数和花费（`usage`，美元，OpenRouter 返回的）、配置。

## 配置

- 模型：DeepSeek V4.1 Flash（`deepseek/deepseek-v4.1-flash`），经 OpenRouter，思考 low；服务商 InferenceNet 优先、GMICloud 备用。
- agent：main 分支的 `rag/agent.py`（结果行里的 `system_sha` / `review_sha` 是 system prompt 和自查指令全文的哈希）；白话索引，别名扩展 route2，自查 layered。
- 每种配置每套题跑一遍。闭卷有 5 题三次都没给出回答（推理用完 8192 token 的输出上限），判卷文件里按错计（`judge` 为 `code: empty reply`）。
- 两种后端对照的 Haiku 各组：Claude Haiku 5.5（`claude-haiku-5-5`），Claude 原生接口（`rag/claude_llm.py`），花费按 Claude 的价格算；五组 62 题的回答合成一包盲判（g1–g5），英文 21 题两组各自判。`haiku-low-nochange` 里 e-h-15 至 e-h-23 这 9 题是在三项改动写进代码之前跑的（`off` 里没有 agg_nudge / pinyin_who），其余 53 题是之后用 `--off agg_nudge pinyin_who` 跑的，两者配置相同。
- 结果行里的 `sec` 是并发跑题时记的，含排队等待，不代表单题用时；README 的用时来自 `eval.bench_latency` 逐题跑。

## 判卷

- 判卷模型 Claude Sonnet（`judge: "claude-sonnet"`），按 `eval/judge_prompt.md` 对照原文判，可以用 `eval.find` 查原文；题目自带的 `judge` 字段说明这一题怎么判。
- 盲判：每道题的闭卷、检索一次、agent 三份回答打乱顺序、换成随机标签（g1 / g2 / g3）放进同一个包，
  两组消融（agent-off、single-nocontext）也照此合包；英文 103 题（en）是五种配置的回答合成一包；en-dev 沿用较早一轮的判卷（检索一次和 agent 合包，两组消融合包，闭卷单独判）；判完再按标签还原到各配置。
- 判卷人拿不准的写进 `uncertain`。三组对照里的 46 条由作者逐条看过，并定了几条边界规则（之后的几组按同样的规则判）：
  - 答到了题目的核心，但夹带编造或张冠李戴的引文、出处、细节 → 部分；错误非常小（一个字、出处标错但引文和事实都对）→ 对。
  - 《吴录》"八绝"答全八人即为对；原文没有答案的题只说 "没有记载"、没交代裴注里的说法 → 部分；回答被截断但已给出核心结论 → 对。
  - 按这条规则，另由一个判卷模型盲复查了全部判错的回答。
- 改判的判决行保留原判（`verdict_ai`）和改判原因（`override`），`overrides.jsonl` 汇总了全部改判。

## 复现

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
