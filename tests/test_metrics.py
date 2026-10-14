"""eval/metrics.py 的测试: is_hit, hit@k, reciprocal rank.

    uv run python -m tests.test_metrics

不依赖 pytest. 包括 "gold 是段落时, 挂在它上面的裴注块也算命中".
"""
from eval import metrics


def _c(*sources, parent=None):
    return {"id": sources[0], "sources": list(sources), "parent": parent}


RANKED = [_c("01-0001"), _c("01-0002", "01-0003"), _c("01-0046"), _c("02-0000n0", parent="02-0000")]


def test_hit_at_k_basic():
    assert metrics.hit_at_k(RANKED, ["01-0046"], 3) == 1.0
    assert metrics.hit_at_k(RANKED, ["01-0046"], 2) == 0.0
    assert metrics.hit_at_k(RANKED, ["01-0001"], 1) == 1.0


def test_hit_at_k_merged_chunk_and_any_gold():
    assert metrics.hit_at_k(RANKED, ["01-0003"], 2) == 1.0, "合并块的任一 source 命中都算"
    assert metrics.hit_at_k(RANKED, ["99-0000", "01-0002"], 2) == 1.0, "gold 有多个时命中任意一个就算"


def test_hit_at_k_edges():
    assert metrics.hit_at_k([], ["01-0001"], 5) == 0.0
    assert metrics.hit_at_k(RANKED, ["01-0046"], 100) == 1.0, "k 超过结果数时看全部"
    assert metrics.hit_at_k(RANKED, ["99-9999"], 4) == 0.0
    assert isinstance(metrics.hit_at_k(RANKED, ["01-0001"], 1), float)


def test_reciprocal_rank():
    assert metrics.reciprocal_rank(RANKED, ["01-0001"]) == 1.0
    assert metrics.reciprocal_rank(RANKED, ["01-0046"]) == 1 / 3
    assert metrics.reciprocal_rank(RANKED, ["99-9999"]) == 0.0
    assert metrics.reciprocal_rank([], ["01-0001"]) == 0.0


def test_reciprocal_rank_first_hit_only():
    assert metrics.reciprocal_rank(RANKED, ["01-0046", "01-0003"]) == 1 / 2, "只看第一个命中"


def test_note_gold():
    assert metrics.hit_at_k(RANKED, ["02-0000n0"], 4) == 1.0
    assert metrics.reciprocal_rank(RANKED, ["02-0000n0"]) == 1 / 4


def test_note_chunk_of_gold_paragraph_counts():
    assert metrics.is_hit(_c("02-0000n0", parent="02-0000"), ["02-0000"]), "gold 段落上挂的裴注块算命中"
    assert metrics.reciprocal_rank(RANKED, ["02-0000"]) == 1 / 4


def test_main_chunk_does_not_hit_note_gold():
    assert not metrics.is_hit(_c("02-0000"), ["02-0000n0"]), "gold 是裴注时, 正文块不算命中"
    assert not metrics.is_hit({"id": "x", "sources": ["01-0001"]}, ["01-0002"]), "没有 parent 键也要能处理"


if __name__ == "__main__":
    from tests._run import run as run_tests
    run_tests(globals())
