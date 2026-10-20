import pytest

from fusion_kit.windows import MofN, m_of_n

F, T = False, True

# Cases from the puzzle (coding_exercise/windows.py), m=3, n=5.
PUZZLE_CASES = [
    ([], []),
    ([1], [F]),
    ([1, 1], [F, F]),
    ([1, 1, 1], [F, F, T]),                        # confirms before window is full
    ([1, 1, 1, 1], [F, F, T, T]),
    ([1, 1, 1, 1, 1], [F, F, T, T, T]),
    ([1, 1, 1, 1, 0], [F, F, T, T, T]),            # one miss doesn't drop it
    ([0, 1, 1, 0, 1], [F, F, F, F, T]),
    ([0, 1, 0, 0, 1], [F, F, F, F, F]),
    ([0, 1, 0, 0, 1, 1], [F, F, F, F, F, T]),
    ([0, 1, 0, 0, 1, 1, 1], [F, F, F, F, F, T, T]),
    ([1, 1, 0, 1, 0, 0, 1, 1], [F, F, F, T, T, F, F, T]),  # old hits slide out
]


@pytest.mark.parametrize("hits, expected", PUZZLE_CASES)
def test_puzzle_cases(hits, expected):
    assert m_of_n(hits, 3, 5) == expected


def test_1_of_1_passes_hits_through():
    hits = [1, 1, 0, 1, 0, 0, 1, 1]
    assert m_of_n(hits, 1, 1) == [bool(h) for h in hits]


@pytest.mark.parametrize("m, n", [(6, 5), (0, 5), (1, 0), (-1, 3)])
def test_bad_parameters(m, n):
    with pytest.raises(ValueError):
        m_of_n([1, 0, 1], m, n)


def test_bad_parameters_rejected_even_with_no_hits():
    with pytest.raises(ValueError):
        m_of_n([], 6, 5)


def test_n_of_n_needs_every_scan():
    assert m_of_n([1, 1, 1, 0, 1, 1, 1], 3, 3) == [F, F, T, F, F, F, T]


def test_bool_and_truthy_inputs():
    assert m_of_n([True, False, True], 2, 3) == [F, F, T]
    assert m_of_n([5, 0, 0.1], 2, 3) == [F, F, T]


def test_streaming_matches_batch():
    hits = [1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0]
    det = MofN(2, 4)
    assert [det.update(h) for h in hits] == m_of_n(hits, 2, 4)


def test_running_count_and_reset():
    det = MofN(2, 3)
    for h in [1, 1, 1, 0]:
        det.update(h)
    assert det.hits == 2              # window holds [1, 1, 0]
    det.reset()
    assert det.hits == 0
    assert det.update(1) is False     # starts fresh


def test_long_stream_count_stays_exact():
    hits = [(i * 7) % 3 == 0 for i in range(10_000)]
    det = MofN(2, 5)
    for i, h in enumerate(hits):
        det.update(h)
        assert det.hits == sum(hits[max(0, i - 4): i + 1])
