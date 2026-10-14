"""测试文件共用的运行器 (不依赖 pytest): 跑模块里所有 test_ 开头的函数, 分别计数通过 / 跳过 / 失败.

    if __name__ == "__main__":
        from tests._run import run as run_tests
        run_tests(globals())

测试里缺真实数据或可选依赖时 raise Skip("原因"), 记为跳过而不是通过. 有失败时以退出码 1 结束, 方便接 CI.
"""
import sys
from typing import Any


class Skip(Exception):
    """这条测试的前提不满足 (没下载数据, 没装可选依赖), 跳过."""


def run(namespace: dict[str, Any]) -> None:
    tests = [(k, v) for k, v in namespace.items() if k.startswith("test_") and callable(v)]
    passed = skipped = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"PASS {name}")
        except Skip as e:
            skipped += 1
            print(f"SKIP {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} 通过" + (f", {skipped} 跳过" if skipped else "") + (f", {failed} 失败" if failed else ""))
    sys.exit(1 if failed else 0)
