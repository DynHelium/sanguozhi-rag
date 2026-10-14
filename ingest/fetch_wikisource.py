"""从维基文库下载三国志 65 卷的原始 wikitext.

    uv run python -m ingest.fetch_wikisource            # 按 manifest.json 记的 revid 下载 (复现同一版本)
    uv run python -m ingest.fetch_wikisource --latest   # 下载最新修订, 并更新 manifest.json

输出: data/sources/sanguozhi/raw/ 下

    卷01.wiki ... 卷65.wiki      原始 wikitext, 永远保持下载原样 (录入错误写进 patches.yaml, 解析时套用)
    manifest.json               list, 每卷一项, 按卷号排序
    └─ {"juan": 1, "title": "三國志/卷01", "revid": 2585429, "timestamp": "2025-08-03T06:52:52Z"}

manifest.json 是复现的依据: 默认按它的 revid 下载, 不同机器拿到的文本完全一致, patches.yaml 也一定适用.
用 --latest 更新后, 要重跑 tests 与 ingest.sanguozhi: 上游改动可能让某条补丁的 find 失效 (会报错).

走 MediaWiki API, 只读, 不需要账号. 每批最多 50 页, 批之间停 2 秒; User-Agent 只写项目名, 不带个人信息.
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
from collections.abc import Iterator
from typing import Any, TypedDict

from rag import config

API = "https://zh.wikisource.org/w/api.php"
UA = {"User-Agent": "sanguozhi-rag/0.1 (https://github.com/DynHelium/sanguozhi-rag)"}
RAW_DIR = config.DATA_DIR / "sources" / "sanguozhi" / "raw"
JUANS = range(1, 66)


class ManifestEntry(TypedDict):
    """manifest.json 的一项: 一卷的页面和修订."""
    juan: int        # 卷号 1-65
    title: str       # 页面标题, 如 "三國志/卷01"
    revid: int       # 修订 id, 按它下载能复现同一版本
    timestamp: str   # 修订时间 (UTC, ISO 8601)


def _query(params: dict[str, str]) -> dict[str, Any]:
    q = urllib.parse.urlencode({**params, "action": "query", "prop": "revisions",
                                "rvprop": "ids|timestamp|content", "rvslots": "main",
                                "format": "json", "formatversion": "2"})
    with urllib.request.urlopen(urllib.request.Request(f"{API}?{q}", headers=UA), timeout=60) as r:
        return json.load(r)


def _batches(items: list[str], n: int = 50) -> Iterator[list[str]]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


def fetch(latest: bool) -> list[ManifestEntry]:
    """下载 65 卷写进 RAW_DIR, 返回新的 manifest (同时写盘). latest=False 且有 manifest.json 时按其中的 revid 取,
    否则按页面标题取最新修订. 页面缺失或卷数不足 65 时 raise RuntimeError."""
    manifest_path = RAW_DIR / "manifest.json"
    if latest or not manifest_path.exists():
        keys = [f"三國志/卷{j:02d}" for j in JUANS]
        param = "titles"
    else:
        old: list[ManifestEntry] = json.loads(manifest_path.read_text(encoding="utf-8"))
        keys = [str(r["revid"]) for r in old]
        param = "revids"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[ManifestEntry] = []
    for i, batch in enumerate(_batches(keys)):
        if i:
            time.sleep(2)
        for p in _query({param: "|".join(batch)})["query"]["pages"]:
            if p.get("missing"):
                raise RuntimeError(f"页面不存在: {p['title']}")
            rev = p["revisions"][0]
            juan = int(p["title"].split("卷")[1])
            (RAW_DIR / f"卷{juan:02d}.wiki").write_text(rev["slots"]["main"]["content"], encoding="utf-8")
            manifest.append({"juan": juan, "title": p["title"], "revid": rev["revid"], "timestamp": rev["timestamp"]})
    manifest.sort(key=lambda r: r["juan"])
    if len(manifest) != len(JUANS):
        raise RuntimeError(f"只拿到 {len(manifest)} 卷")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--latest", action="store_true", help="下载最新修订并更新 manifest.json")
    args = ap.parse_args()
    m = fetch(args.latest)
    print(f"{len(m)} 卷 -> {RAW_DIR} (最新修订时间 {max(r['timestamp'] for r in m)})")


if __name__ == "__main__":
    main()
