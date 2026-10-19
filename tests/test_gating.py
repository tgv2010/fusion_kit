import pytest

from fusion_kit.gating import Group, gate_tracks, merge_intervals

# Cases from the puzzle (coding_exercise/gating.py). tracks are (center, uncertainty).
PUZZLE_CASES = [
    # no overlap
    ([(2, 0.5), (4, 0.5), (8, 1)],
     [(1.5, 2.5, [0]), (3.5, 4.5, [1]), (7, 9, [2])]),
    # three overlap
    ([(1, 0.5), (2, 1), (3, 0.5), (10, 1)],
     [(0.5, 3.5, [0, 1, 2]), (9, 11, [3])]),
    # two and two
    ([(2, 1), (3, 0.5), (10, 2), (9, 1)],
     [(1, 3.5, [0, 1]), (8, 12, [2, 3])]),
    # containment: one wide gate swallows the rest
    ([(2, 0.5), (4, 0.5), (8, 1), (10, 10)],
     [(0, 20, [0, 1, 2, 3])]),
    # gates that touch exactly merge
    ([(4, 0.5), (5, 0.5)],
     [(3.5, 5.5, [0, 1])]),
    # sort by lower edge, not center
    ([(3, 0.5), (5, 4), (1, 0.5)],
     [(0.5, 9, [0, 1, 2])]),
    # original indices survive sorting; output is ordered by lower edge
    ([(10, 1), (2, 1), (3, 0.5)],
     [(1, 3.5, [1, 2]), (9, 11, [0])]),
    # empty
    ([], []),
    # single
    ([(5, 1)], [(4, 6, [0])]),
    # zero-width gates (perfectly certain), duplicates
    ([(5, 0), (5, 0)], [(5, 5, [0, 1])]),
    # touch at 0.3 on paper, but 0.45 - 0.15 rounds to 0.30000000000000004
    ([(0.15, 0.15), (0.45, 0.15)], [(0, 0.6, [0, 1])]),
]


def assert_groups(actual, expected):
    assert len(actual) == len(expected)
    for g, (lo, hi, members) in zip(actual, expected):
        assert g.lo == pytest.approx(lo)
        assert g.hi == pytest.approx(hi)
        assert g.members == members


@pytest.mark.parametrize("tracks, expected", PUZZLE_CASES)
def test_puzzle_cases(tracks, expected):
    assert_groups(gate_tracks(tracks), expected)


def test_float_touch_needs_tolerance():
    tracks = [(0.15, 0.15), (0.45, 0.15)]
    assert len(gate_tracks(tracks, tol=0)) == 2
    assert len(gate_tracks(tracks)) == 1


def test_chaining_through_middle_track():
    # 0 and 2 don't overlap, but both overlap 1
    tracks = [(0, 1), (1.5, 1), (3, 1)]
    assert_groups(gate_tracks(tracks), [(-1, 4, [0, 1, 2])])


def test_small_gap_stays_separate():
    assert len(gate_tracks([(0, 1), (2.001, 1)])) == 2


def test_negative_positions():
    assert_groups(gate_tracks([(-5, 1), (-3.5, 1)]), [(-6, -2.5, [0, 1])])


def test_negative_uncertainty_rejected():
    with pytest.raises(ValueError):
        gate_tracks([(1, -0.5)])


def test_returns_named_groups():
    (g,) = gate_tracks([(5, 1)])
    assert isinstance(g, Group)
    lo, hi, members = g
    assert (lo, hi, members) == (4, 6, [0])


def test_input_not_modified():
    tracks = [(10, 1), (2, 1)]
    gate_tracks(tracks)
    assert tracks == [(10, 1), (2, 1)]


# Plain interval merge (coding_exercise/merge_intervals_v*.py).
INTERVAL_CASES = [
    ([[1, 3], [2, 6], [8, 10], [15, 18]], [(1, 6), (8, 10), (15, 18)]),
    ([[8, 10], [1, 3], [2, 6]], [(1, 6), (8, 10)]),
    ([[1, 10], [2, 3]], [(1, 10)]),
    ([], []),
]


@pytest.mark.parametrize("intervals, expected", INTERVAL_CASES)
def test_merge_intervals(intervals, expected):
    assert merge_intervals(intervals) == expected


def test_merge_intervals_tolerance():
    assert merge_intervals([(0, 1), (1.5, 2)]) == [(0, 1), (1.5, 2)]
    assert merge_intervals([(0, 1), (1.5, 2)], tol=0.5) == [(0, 2)]


def test_merge_intervals_does_not_mutate():
    intervals = [[8, 10], [1, 3]]
    merge_intervals(intervals)
    assert intervals == [[8, 10], [1, 3]]


def test_reversed_interval_rejected():
    with pytest.raises(ValueError):
        merge_intervals([(3, 1)])
