import pytest

from fusion_kit.blobs import count_blobs, find_blobs

# Cases from the puzzle (coding_exercise/number_of_islands_v2.py).
PUZZLE_CASES = [
    ([["1", "1", "0", "0", "0"],
      ["1", "1", "0", "0", "0"],
      ["0", "0", "1", "0", "0"],
      ["0", "0", "0", "1", "1"]], 3),
    ([["1", "0", "1"]], 2),                     # one row, two blobs
    ([["1", "1"],
      ["1", "1"]], 1),                          # two rows, one blob
    ([["1", "0", "1"],
      ["1", "1", "1"]], 1),                     # joined by the row below
    ([["0"]], 0),                               # no foreground
]


@pytest.mark.parametrize("grid, expected", PUZZLE_CASES)
def test_puzzle_cases(grid, expected):
    assert count_blobs(grid) == expected


def test_empty_grid():
    assert count_blobs([]) == 0
    assert count_blobs([[]]) == 0


def test_diagonal_cells_are_separate_with_4_connectivity():
    grid = [[1, 0],
            [0, 1]]
    assert count_blobs(grid, connectivity=4) == 2


def test_diagonal_cells_join_with_8_connectivity():
    grid = [[1, 0],
            [0, 1]]
    assert count_blobs(grid, connectivity=8) == 1


def test_int_and_bool_grids():
    assert count_blobs([[1, 0, 1]]) == 2
    assert count_blobs([[True, True, False]]) == 1


def test_custom_foreground_threshold():
    # e.g. an intensity image thresholded at 128
    image = [[200, 50, 130],
             [10, 10, 255]]
    assert count_blobs(image, is_foreground=lambda v: v >= 128) == 2


def test_ring_is_one_blob():
    grid = ["111",
            "101",
            "111"]
    assert count_blobs(grid) == 1


def test_ragged_rows():
    grid = [["1", "1"],
            ["0"],
            ["1", "0", "1"]]
    assert count_blobs(grid) == 3


def test_find_blobs_returns_cells():
    grid = [["1", "1", "0"],
            ["0", "0", "1"]]
    blobs = sorted(find_blobs(grid), key=min)
    assert blobs == [{(0, 0), (0, 1)}, {(1, 2)}]


def test_large_snake_does_not_hit_recursion_limit():
    # One long winding blob; a recursive flood fill would overflow here.
    rows, cols = 200, 200
    grid = [["1" if r % 2 == 0 or c == (cols - 1 if r % 4 == 1 else 0) else "0"
             for c in range(cols)] for r in range(rows)]
    assert count_blobs(grid) == 1


def test_bad_connectivity():
    with pytest.raises(ValueError):
        count_blobs([[1]], connectivity=6)
