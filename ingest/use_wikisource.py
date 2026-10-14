"""下载维基文库的标点版, 生成检索要用的数据: 下载 -> 解析 -> 切块. 跑检索前需要先运行这一步.

rag/ 和 eval/ 只读这一步生成的 data/processed/sanguozhi.jsonl 和 chunks.jsonl, 从不读白文 baiwen.jsonl
(白文不会被改动或覆盖, 它只是仓库里可以直接查看的原文副本).

    uv run python -m ingest.use_wikisource            # 按 data/sources/sanguozhi/raw/manifest.json 的 revid 下载 (结果可复现)
    uv run python -m ingest.use_wikisource --latest   # 下载各卷最新修订 (manifest.json 随之更新; 补丁可能要调整)

为什么要下载: 仓库里只放白文 (data/sources/sanguozhi/baiwen.jsonl, 见 ingest/baiwen.py). 检索要按句号切块,
回答要引用带标点的原文, 所以要先取得标点版. 维基文库的文字以 CC BY-SA 4.0 提供 (原文本身是公有领域),
下载到本地使用; 再发布这些文字时请按该协议署名.

依次运行 (每一步也可以单独跑):
    1. ingest.fetch_wikisource   下载 65 卷 wikitext -> data/sources/sanguozhi/raw/卷NN.wiki
    2. ingest.sanguozhi          解析 (套用 patches.yaml) -> data/processed/sanguozhi.jsonl
    3. rag.chunk                 切块 -> data/processed/chunks.jsonl
之后建向量缓存: uv run python -m rag.dense (要先 uv sync --extra index).
"""
import argparse
import subprocess
import sys

STEPS = [["ingest.fetch_wikisource"], ["ingest.sanguozhi"], ["rag.chunk"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="下载维基文库标点版并解析, 切块")
    ap.add_argument("--latest", action="store_true", help="下载最新修订, 不按 manifest.json 的 revid")
    args = ap.parse_args()
    for step in STEPS:
        cmd = [sys.executable, "-m", *step] + (["--latest"] if args.latest and step[0] == "ingest.fetch_wikisource" else [])
        print(f"\n== {' '.join(cmd[2:])}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            sys.exit(f"{step[0]} 失败, 后面的步骤没有运行")
    print("\n完成. 下一步建向量缓存: uv run python -m rag.dense")


if __name__ == "__main__":
    main()
