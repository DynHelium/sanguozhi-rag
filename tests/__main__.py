"""跑全部测试: 每个 tests/test_*.py 单独一个子进程 (各自有 if __name__ == "__main__" 入口), 最后汇总.

    uv run python -m tests            # 全部; 任何一组失败时退出码为 1
    uv run python -m tests bm25 chunk # 只跑名字里含这些词的

都不调大模型; 缺数据或可选依赖的用例自己跳过 (见 tests/_run.py). CI 用的就是这个入口.
"""
import subprocess
import sys
from pathlib import Path


def main() -> None:
    names = sorted(p.stem for p in Path(__file__).parent.glob("test_*.py"))
    if sys.argv[1:]:
        names = [n for n in names if any(k in n for k in sys.argv[1:])]
    failed = []
    for name in names:
        r = subprocess.run([sys.executable, "-m", f"tests.{name}"], capture_output=True, text=True)
        last = (r.stdout.strip().splitlines() or ["(没有输出)"])[-1]
        print(f"{name:22s} {last}")
        if r.returncode != 0:
            failed.append(name)
            print((r.stdout + r.stderr).strip()[-1500:], file=sys.stderr)
    print(f"\n{len(names) - len(failed)}/{len(names)} 组通过" + (f"; 失败: {' '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
