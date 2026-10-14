"""长度分布统计, 用来选切块的 min_len / max_len, 以及检查切出来的结果.

    uv run python -m eval.length_stats                                   # 默认读段落 data/processed/sanguozhi.jsonl
    uv run python -m eval.length_stats data/processed/chunks.jsonl       # 读切块结果
    uv run python -m eval.length_stats --min-len 50 --max-len 400 -n 5   # 改阈值, 列出最短/最长各 5 个

自动识别输入: 有 "tier" 字段的是 chunks.jsonl (按 tier 分 正文块/裴注块);
否则是 sanguozhi.jsonl (正文段 + 每条裴注各算一个, 裴注 id 为 "段落idn序号").
长度一律按 text 的字数算 (含标点与换行).
"""
import argparse
import json
from pathlib import Path

from rag import config

DEFAULT = config.DATA_DIR / "processed" / "sanguozhi.jsonl"


def load_items(path: Path) -> dict[str, list[tuple[str, str]]]:
    """返回 {类别: [(id, text), ...]}."""
    groups: dict[str, list[tuple[str, str]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "tier" in r:
                groups.setdefault("正文块" if r["tier"] == 1 else "裴注块", []).append((r["id"], r["text"]))
            else:
                groups.setdefault("正文段", []).append((r["id"], r["text"]))
                for i, n in enumerate(r["notes"]):
                    groups.setdefault("裴注", []).append((f"{r['id']}n{i}", n["text"]))
    return groups


def percentile(sorted_lens: list[int], p: float) -> int:
    return sorted_lens[min(len(sorted_lens) - 1, int(len(sorted_lens) * p / 100))]


def report(name: str, items: list[tuple[str, str]], min_len: int, max_len: int, n: int) -> None:
    lens = sorted(len(t) for _, t in items)
    pcts = "  ".join(f"p{p}={percentile(lens, p)}" for p in (10, 25, 50, 75, 90, 99))
    print(f"\n[{name}] {len(lens)} 个, 共 {sum(lens)} 字, 均值 {sum(lens) // len(lens)}, 最短 {lens[0]}, 最长 {lens[-1]}")
    print(f"  {pcts}")
    short = sum(x < min_len for x in lens)
    long = sum(x > max_len for x in lens)
    print(f"  < {min_len} 字: {short} ({short / len(lens):.1%})    > {max_len} 字: {long} ({long / len(lens):.1%})")
    buckets = [0, 25, 50, 100, 200, 400, 800, 1600, 3200, 10 ** 9]
    for lo, hi in zip(buckets, buckets[1:]):
        k = sum(lo <= x < hi for x in lens)
        label = f"{lo}-{hi - 1}" if hi < 10 ** 9 else f"{lo}+"
        print(f"  {label:>10} | {'#' * round(60 * k / len(lens)):<60} {k}")
    if n:
        by_len = sorted(items, key=lambda it: len(it[1]))
        print(f"  最短 {n} 个:")
        for i, t in by_len[:n]:
            print(f"    {i:<14} {len(t):>5}  {t[:40]!r}")
        print(f"  最长 {n} 个:")
        for i, t in by_len[-n:]:
            print(f"    {i:<14} {len(t):>5}  {t[:40]!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="长度分布统计")
    ap.add_argument("path", nargs="?", type=Path, default=DEFAULT)
    ap.add_argument("--min-len", type=int, default=50)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("-n", type=int, default=3, help="列出最短/最长各 n 个, 0 不列")
    args = ap.parse_args()
    for name, items in load_items(args.path).items():
        report(name, items, args.min_len, args.max_len, args.n)


if __name__ == "__main__":
    main()
