import math
import random

import pytest

from fusion_kit.gating import gate_tracks
from fusion_kit.kalman import Kalman1D


def run(z_series, r, q):
    """Start on the first measurement, step through the rest. Return the filter."""
    kf = Kalman1D(z_series[0], r, q)
    for z in z_series[1:]:
        kf.step(z)
    return kf


# --- Cases from the puzzle (coding_exercise/kalman.py) ---

def test_case1_no_missed_frames():
    kf = Kalman1D(10, r=1, q=0)
    assert kf.step(12) == pytest.approx((11.0, 1 / 2))
    assert kf.step(11) == pytest.approx((11.0, 1 / 3))


def test_case2_one_missed_frame():
    kf = Kalman1D(10, r=1, q=0.5)
    assert kf.step(None) == pytest.approx((10, 1.5))      # coast: P grows by q
    assert kf.step(12) == pytest.approx((34 / 3, 2 / 3))  # K = 2/3


def test_case3_noisy_constant_target():
    random.seed(0)
    true_range, r, n = 10.0, 1.0, 50
    z_series = [random.gauss(true_range, math.sqrt(r)) for _ in range(n)]
    kf = run(z_series, r, q=0)
    assert kf.P == pytest.approx(r / n)                   # q=0: variance falls as r/N
    assert abs(kf.x - true_range) <= 3 * math.sqrt(kf.P)
    assert kf.x == pytest.approx(sum(z_series) / n)       # q=0 is a plain running average


@pytest.mark.parametrize("r, q", [(0, 0), (-1, 0), (0.01, -1)])
def test_case4_5_bad_noise(r, q):
    with pytest.raises(ValueError):
        Kalman1D(10, r, q)


def test_case6_none_on_init():
    with pytest.raises(ValueError):
        Kalman1D(None, r=1, q=0.5)


# --- Edge cases ---

def test_update_none_rejected():
    kf = Kalman1D(10, r=1)
    with pytest.raises(ValueError):
        kf.update(None)


def test_gain_is_none_before_first_update():
    assert Kalman1D(10, r=1).K is None


def test_q0_gain_shrinks_as_1_over_n():
    kf = Kalman1D(0, r=2, q=0)
    for n in range(1, 6):
        kf.step(0)
        assert kf.K == pytest.approx(1 / (n + 1))


def test_coasting_grows_P_by_q_per_step():
    kf = Kalman1D(10, r=1, q=0.5)
    for _ in range(4):
        kf.step(None)
    assert kf.P == pytest.approx(1 + 4 * 0.5)
    assert kf.x == 10


def test_huge_q_follows_sensor():
    kf = Kalman1D(10, r=1, q=1e9)
    x, _ = kf.step(50)
    assert kf.K == pytest.approx(1, abs=1e-6)
    assert x == pytest.approx(50, abs=1e-4)


def test_steady_state_variance():
    # With q > 0, P stops shrinking. Before each update it settles at
    # (q + sqrt(q^2 + 4qr)) / 2; for q=0.5, r=1 that's 1.0, and 0.5 after update.
    kf = Kalman1D(10, r=1, q=0.5)
    for _ in range(100):
        kf.step(10)
    assert kf.P == pytest.approx(0.5)
    assert kf.predict()[1] == pytest.approx(1.0)


def test_moving_target_lags():
    # The documented limit: a target at constant speed is followed with a lag.
    # Bigger q shrinks the lag but never removes it.
    truth = [float(t) for t in range(50)]      # 1 m per step, no noise
    lag_slow = truth[-1] - run(truth, r=1, q=0).x
    lag_fast = truth[-1] - run(truth, r=1, q=1).x
    assert lag_slow > 20                       # q=0 averages: ends near the middle
    assert 0 < lag_fast < lag_slow


# --- gate() ---

def test_gate_width():
    kf = Kalman1D(10, r=1, q=0)
    assert kf.gate() == pytest.approx((10, 3 * math.sqrt(2)))   # P + r = 2
    assert kf.gate(k=1) == pytest.approx((10, math.sqrt(2)))


def test_gate_tightens_as_filter_settles():
    kf = Kalman1D(10, r=1, q=0)
    widths = []
    for _ in range(5):
        kf.step(10)
        widths.append(kf.gate()[1])
    assert widths == sorted(widths, reverse=True)
    assert widths[-1] > 3 * math.sqrt(kf.r)    # never narrower than the sensor alone


def test_gate_widens_while_coasting():
    kf = Kalman1D(10, r=1, q=0.5)
    before = kf.gate()[1]
    kf.predict()
    assert kf.gate()[1] > before


def test_gate_bad_k():
    with pytest.raises(ValueError):
        Kalman1D(10, r=1).gate(k=0)


def test_gates_feed_gate_tracks():
    near_a, near_b, far = Kalman1D(10, r=1), Kalman1D(11, r=1), Kalman1D(40, r=1)
    groups = gate_tracks([kf.gate() for kf in (near_a, near_b, far)])
    assert [g.members for g in groups] == [[0, 1], [2]]
