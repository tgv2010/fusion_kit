import itertools
import math

import numpy as np
import pytest

from fusion_kit.association import associate, cost_matrix, iou, mahalanobis_cost
from fusion_kit.kalman import ConstantVelocity


def test_obvious_pairs():
    cost = [[0.1, 5.0],
            [5.0, 0.2]]
    for method in ("optimal", "greedy"):
        r = associate(cost, method=method)
        assert r.matches == [(0, 0), (1, 1)]
        assert r.unmatched_tracks == [] and r.unmatched_detections == []


def test_greedy_vs_optimal():
    # Greedy grabs the cheapest pair (0,0) and is left with (1,1), outside the gate.
    cost = [[1.0, 2.0],
            [2.0, 100.0]]
    assert associate(cost, gate=50, method="optimal").matches == [(0, 1), (1, 0)]
    g = associate(cost, gate=50, method="greedy")
    assert g.matches == [(0, 0)]
    assert g.unmatched_tracks == [1] and g.unmatched_detections == [1]


def test_gate_and_forbidden_costs():
    cost = [[1.0, math.inf],
            [np.nan, 20.0]]
    r = associate(cost, gate=9.0)
    assert r.matches == [(0, 0)]
    assert r.unmatched_tracks == [1]
    assert r.unmatched_detections == [1]


def test_rectangular():
    more_dets = associate([[3.0, 1.0, 2.0]])
    assert more_dets.matches == [(0, 1)] and more_dets.unmatched_detections == [0, 2]
    more_trks = associate([[3.0], [1.0], [2.0]])
    assert more_trks.matches == [(1, 0)] and more_trks.unmatched_tracks == [0, 2]


def test_empty():
    assert associate(np.zeros((0, 3))).unmatched_detections == [0, 1, 2]
    assert associate(np.zeros((2, 0))).unmatched_tracks == [0, 1]
    assert associate(np.full((2, 2), math.inf)).matches == []


def test_bad_input():
    with pytest.raises(ValueError):
        associate([1.0, 2.0])
    with pytest.raises(ValueError):
        associate([[1.0]], method="fastest")


def test_optimal_is_truly_optimal():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = rng.integers(1, 6)
        cost = rng.uniform(0, 10, (n, n))
        best = min(sum(cost[i, p[i]] for i in range(n)) for p in itertools.permutations(range(n)))
        r = associate(cost, method="optimal")
        assert sum(cost[t, d] for t, d in r.matches) == pytest.approx(best)
        g = associate(cost, method="greedy")
        assert sum(cost[t, d] for t, d in g.matches) >= best - 1e-9


def test_optimal_maximizes_matches_before_cost():
    rng = np.random.default_rng(1)
    for _ in range(50):
        cost = rng.uniform(0, 10, (4, 4))
        cost[rng.random((4, 4)) < 0.5] = math.inf
        allowed = np.isfinite(cost)
        most = max(sum(allowed[i, p[i]] for i in range(4)) for p in itertools.permutations(range(4)))
        assert len(associate(cost).matches) == most


def test_iou():
    assert iou((0, 0, 2, 2), (0, 0, 2, 2)) == 1
    assert iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0
    assert iou((0, 0, 2, 2), (1, 0, 3, 2)) == pytest.approx(1 / 3)
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0


def test_cost_matrix_and_mahalanobis():
    tracks = [ConstantVelocity([0.0, 0.0], r=1, q=0.1), ConstantVelocity([10.0, 0.0], r=1, q=0.1)]
    dets = [[10.2, 0.0], [0.1, 0.1]]
    cost = cost_matrix(tracks, dets, mahalanobis_cost)
    assert cost.shape == (2, 2)
    assert cost[0, 1] == pytest.approx(tracks[0].distance(dets[1]) ** 2)
    assert associate(cost, gate=9.0).matches == [(0, 1), (1, 0)]
