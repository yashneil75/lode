"""Reciprocal Rank Fusion is the only place the two retrievers meet; test the math."""

from __future__ import annotations

import numpy as np
import pytest

from lode import search


def test_rrf_scores_are_reciprocal_of_rank_plus_k():
    fused = search.reciprocal_rank_fusion([[(10, 5.0), (11, 4.0)]])
    assert fused[10] == 1 / (search.RRF_K + 1)
    assert fused[11] == 1 / (search.RRF_K + 2)


def test_a_top_hit_in_both_lists_beats_a_solo_first_place():
    lexical = [(1, 9.0), (2, 8.0), (3, 7.0), (4, 6.0), (5, 5.0)]
    semantic = [(2, 0.9), (3, 0.8), (4, 0.7), (5, 0.6), (1, 0.5)]
    fused = search.reciprocal_rank_fusion([lexical, semantic])
    assert max(fused, key=fused.get) == 2, "second place twice beats first place once"


def test_a_chunk_found_by_both_retrievers_outranks_one_found_by_only_one():
    fused = search.reciprocal_rank_fusion([[(1, 9.0), (2, 8.0)], [(2, 0.9), (3, 0.8)]])
    assert fused[2] > fused[1] > fused[3]


def test_weights_scale_contributions():
    unweighted = search.reciprocal_rank_fusion([[(1, 1.0)], [(2, 1.0)]])
    assert unweighted[1] == unweighted[2]
    weighted = search.reciprocal_rank_fusion([[(1, 1.0)], [(2, 1.0)]], weights=(3.0, 1.0))
    assert weighted[1] > weighted[2]


def test_rrf_never_consults_the_raw_scores():
    """Score magnitudes differ by orders of magnitude between retrievers."""
    fused = search.reciprocal_rank_fusion([[(1, 1e9)], [(2, 1e-9)]])
    assert abs(fused[1] - fused[2]) < 1e-12


def test_top_by_score_sorts_descending_and_truncates():
    import numpy as np

    ids = np.array([5, 2, 9, 1], dtype=np.int64)
    sims = np.array([0.1, 0.9, 0.5, 0.9], dtype=np.float32)
    top = search.top_by_score(ids, sims, 3)
    assert len(top) == 3
    assert [score for _, score in top] == sorted([s for _, s in top], reverse=True)
    assert top[0][0] == 1 and top[0][1] == pytest.approx(0.9, abs=1e-6), (
        "ties break on chunk id, so results stay deterministic"
    )


def test_top_by_score_on_nothing():
    import numpy as np

    assert search.top_by_score(np.array([], dtype=np.int64), np.array([], dtype=np.float32), 5) == []


def test_positions_are_one_based():
    assert search._positions([(7, 0.1), (8, 0.2)]) == {7: 1, 8: 2}
