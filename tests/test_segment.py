import math

import numpy as np
import pytest

from fusion_kit.segment import beam_gap_rule, centroids, euclidean_clusters, gap_rule, segment_scan
from fusion_kit.transforms import polar_to_xy


def as_lists(groups):
    return [list(g) for g in groups]


def test_segment_scan_basic():
    pts = [[0, 0], [0.1, 0], [0.2, 0], [5, 0], [5.1, 0]]
    assert as_lists(segment_scan(pts, gap_rule(0.5))) == [[0, 1, 2], [3, 4]]


def test_segment_scan_wraps_around():
    pts = [[1, 0], [1, 0.1], [5, 5], [1, -0.1]]     # last point joins the first run
    assert as_lists(segment_scan(pts, gap_rule(0.5), wrap=True)) == [[3, 0, 1], [2]]
    assert len(segment_scan(pts, gap_rule(0.5), wrap=False)) == 3


def test_segment_scan_min_points():
    pts = [[0, 0], [0.1, 0], [5, 0]]
    assert as_lists(segment_scan(pts, gap_rule(0.5), min_points=2)) == [[0, 1]]
    far_ok = lambda run_pts: 1 if np.linalg.norm(run_pts.mean(axis=0)) > 4 else 2
    assert as_lists(segment_scan([[0, 0], [5, 0]], gap_rule(0.5), min_points=far_ok)) == [[1]]


def test_segment_scan_empty_and_single():
    assert segment_scan(np.zeros((0, 2)), gap_rule(1)) == []
    assert as_lists(segment_scan([[1, 2]], gap_rule(1))) == [[0]]


def test_beam_gap_rule_scales_with_range():
    step = math.radians(1.0)
    rule = beam_gap_rule(step, k=2.0, floor=0.15)
    near_a, near_b = (np.array(polar_to_xy(5.0, a)) for a in (0, 3 * step))
    far_a, far_b = (np.array(polar_to_xy(20.0, a)) for a in (0, 2 * step))
    assert rule(far_a, far_b)                         # 0.7 m apart but at 20 m: same object
    assert not rule(near_a, near_b)                   # 3 beams apart at 5 m (0.26 m): different
    assert rule(np.array([0.1, 0.0]), np.array([0.2, 0.0]))   # floor applies near the sensor
    with pytest.raises(ValueError):
        beam_gap_rule(0)


def test_euclidean_clusters():
    rng = np.random.default_rng(0)
    a = rng.normal([0, 0, 0], 0.05, (20, 3))
    b = rng.normal([3, 0, 1], 0.05, (10, 3))
    noise = np.array([[10.0, 10.0, 10.0]])
    pts = np.vstack([a, b, noise])
    groups = euclidean_clusters(pts, max_gap=0.5, min_points=2)
    assert [len(g) for g in groups] == [20, 10]
    assert centroids(pts, groups)[1] == pytest.approx([3, 0, 1], abs=0.05)


def test_euclidean_clusters_chain():
    line = [[0.4 * i, 0.0] for i in range(10)]           # each 0.4 apart, ends 3.6 apart
    assert len(euclidean_clusters(line, max_gap=0.5)) == 1


def test_euclidean_clusters_edge_cases():
    assert euclidean_clusters(np.zeros((0, 2)), 1.0) == []
    with pytest.raises(ValueError):
        euclidean_clusters([[0, 0]], 0)
