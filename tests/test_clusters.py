import pytest

from fusion_kit.clusters import Clusters, cluster, count_clusters

# Cases from the puzzle (coding_exercise/object_count*.py).
PUZZLE_CASES = [
    (6, [[4, 0], [2, 5]], 4),          # labels aren't positions; 1 and 3 match nothing
    (4, [[0, 2], [1, 3]], 2),          # 0-2 doesn't include 1
    (5, [[3, 1], [0, 4], [1, 0]], 2),  # either order, chained across the list
    (3, [[0, 1], [2, 1]], 1),
    (2, [[0, 1], [0, 1]], 1),          # duplicate pair
    (5, [[0, 1], [1, 2], [3, 4]], 2),
    (4, [[0, 1], [2, 3], [1, 2]], 1),  # late pair bridges two clusters
    (3, [], 3),                        # no matches
]


@pytest.mark.parametrize("n, pairs, expected", PUZZLE_CASES)
def test_puzzle_cases(n, pairs, expected):
    assert count_clusters(n, pairs) == expected


def test_zero_items():
    assert count_clusters(0, []) == 0
    assert cluster(0, []) == []


def test_self_pair_is_noop():
    assert count_clusters(3, [[1, 1]]) == 3


def test_groups():
    assert cluster(6, [[4, 0], [2, 5]]) == [{0, 4}, {1}, {2, 5}, {3}]


def test_pairs_can_be_tuples():
    assert count_clusters(4, [(0, 1), (2, 3)]) == 2


def test_out_of_range_item():
    with pytest.raises(IndexError):
        count_clusters(3, [[0, 3]])
    with pytest.raises(IndexError):
        count_clusters(3, [[-1, 0]])


def test_negative_n():
    with pytest.raises(ValueError):
        Clusters(-1)


def test_incremental_use():
    c = Clusters(4)
    assert c.union(0, 1) is True
    assert c.union(1, 0) is False   # already merged
    assert c.find(0) == c.find(1)
    assert c.find(2) != c.find(3)
    assert c.count == 3


def test_long_chain():
    n = 10_000
    pairs = [[i, i + 1] for i in range(n - 1)]
    assert count_clusters(n, pairs) == 1
