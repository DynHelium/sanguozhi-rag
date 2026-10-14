# 评测结果

main 分支 README "回答效果" 里的数字都来自这里的文件。每套题一个回答文件和一个判卷文件：

| 文件 | 内容 |
|---|---|
| `<名>.jsonl` | `eval.run_agent` 的结果行：问题、回答、模型看过的原文块（`retrieved`）、工具调用、用时、token 数、配置 |
| `<名>.judged.jsonl` | 判卷结果：每题 对 / 部分 / 错，错在哪（`errors`），一句说明 |

## 配置

- 模型：DeepSeek V4.1 Flash（`deepseek/deepseek-v4.1-flash`），经 OpenRouter，思考 low；服务商 InferenceNet 优先、GMICloud 备用。
- agent：本分支的 `rag/agent.py`（与 main 首次公开版本相同）（结果行里的 `system_sha` / `review_sha` 是 system prompt 和自查指令全文的哈希，
  改过 prompt 就会变）；白话索引，别名扩展 route2，自查 layered。
- 判卷：判卷模型 Claude（具体型号未记录）按 `eval/judge_prompt.md` 对照原文判，题目自带的 `judge` 字段说明这一题怎么判。

每套题只跑一遍：同一题每次的回答会不同，单遍结果有波动，只作参考。

## 复现

这里的文件按题集重命名过；复现时 `--name` 任取。`eval.run_agent` 的结果写在 `data/processed/eval_runs/answers/<名>.jsonl`
（`eval/common.py` 的 `OUT_DIR`），判卷脚本也从那里读；要重判这里的文件，先把它复制过去（如复制成 `answers/ho1.jsonl`）。

```bash
uv run python -m eval.run_agent --openrouter --models deepseek-v4.1-flash --effort low \
    --questions eval/questions-holdout.jsonl --name ho1
uv run python -m eval.ai_judge prep ho1 --questions eval/questions-holdout.jsonl   # 打包: data/processed/eval_runs/judge/ho1/batchNN.json
# 判卷模型按 eval/judge_prompt.md 逐批判, 写成同名的 batchNN.out.jsonl (这一步另外调用模型, 脚本本身不调)
uv run python -m eval.ai_judge merge ho1                                            # 汇总对 / 部分 / 错
uv run python -m eval.judge --name ho1                                              # 或者在浏览器里人工判
```
