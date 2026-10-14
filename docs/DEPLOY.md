# 部署网页 demo

网页聊天（`app/chat.py`）和海龟汤（`app/soup.py`）都是 Gradio 应用，本机运行见 README。这里说明放到公网时要注意什么。

## 要准备的东西

| 项目 | 大小 / 要求 | 说明 |
|---|---|---|
| bge-m3 模型 | 约 2.3GB 磁盘，加载后占几 GB 内存 | 第一次加载时从 Hugging Face 下载；查询时每次编码一句话，CPU 也够用 |
| 向量缓存 | 原文、白话两套各约 17MB（4365 块 × 1024 维 float32） | 预先建好（`rag.dense`、`rag.dense --trans`）随应用一起放，不要在启动时现编（约 12 分钟：原文约 5 分钟 + 白话约 7 分钟） |
| 原文和切块 | 几十 MB | `ingest.use_wikisource`、`ingest.trans_chunks` 生成 |
| OpenRouter key | 环境变量 `LLM_API_KEY`（或 `OPENROUTER_API_KEY`） | 只放在服务端（如平台的 Secrets），浏览器拿不到 |

## 控制花费

公开部署后任何人都能提问，花费随提问次数增长；多轮对话（每轮都带上前文）和失败重试会更贵。网页聊天自带这几道限制：

- `--daily-budget <美元>`：按 OpenRouter 返回的实际花费累计，当天用满后拒绝新问题，第二天清零（进程重启也会清零）。
  每题答完才记账，几个问题同时在答时总花费可能略超上限，所以它不是硬上限。
- 每个问题最长 500 字，每次对话最多 20 轮（`app/chat.py` 的 `MAX_QUESTION_CHARS`、`MAX_TURNS`）。
- 用户点停止后，后台不再开始新的一次模型调用；正在进行的那次请求（包括它内部的限流重试）可能还会做完。
- 同时最多 4 个问题在回答（Gradio 的 `concurrency_limit`），其余排队；同一模型同时在途的请求数另有上限（`rag/openrouter.PER_MODEL`）。

另外建议在 OpenRouter 后台给这个 key 设一个总额度上限，作为最后一道保险。

话题限制（只回答三国历史问题）只靠 system prompt，代码不强制，别的请求仍会消耗模型调用。公网部署时可以考虑加检索分数阈值或分类器把关。

## Hugging Face Spaces

Spaces 和 Gradio 配合最直接，但要注意：

- 计划和硬件按 Hugging Face 当前的 Spaces 价格和硬件选择；免费硬件可能放不下模型或会休眠，部署前核实。
  Space 的磁盘不持久（重启后要重新下载模型），所以把模型和向量缓存打进镜像，或放到持久存储里。
- key 放在 Space 的 Secrets，变量名 `LLM_API_KEY`；启动命令用 `--openrouter --daily-budget …`。
- Space 是公开的，任何人都能用，花费上限一定要设。

## 不部署也能展示

README 里的运行示例和截图就能说明功能；需要录屏时，用本机跑网页聊天录一段"提问 → 检索过程 → 回答 → 出处和引文核对"。
