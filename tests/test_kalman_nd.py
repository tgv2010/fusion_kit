import math
import random

import numpy as np
import pytest

from fusion_kit.gating import gate_tracks
from fusion_kit.kalman import ConstantVelocity, Kalman1D, KalmanFilter


def scalar_kf(z0, r, q):
    """The general filter configured as the 1D puzzle filter."""
    return KalmanFilter(x0=z0, P0=r, F=1, H=1, Q=q, R=r)


# --- General filter reduces to the 1D puzzle filter ---

def test_matches_puzzle_case1():
    kf = scalar_kf(10, r=1, q=0)
    x, P = kf.step(12)
    assert (x[0], P[0, 0]) == pytest.approx((11.0, 1 / 2))
    x, P = kf.step(11)
    assert (x[0], P[0, 0]) == pytest.approx((11.0, 1 / 3))


def test_matches_puzzle_case2():
    kf = scalar_kf(10, r=1, q=0.5)
    kf.step(None)
    assert kf.P[0, 0] == pytest.approx(1.5)
    x, P = kf.step(12)
    assert (x[0], P[0, 0]) == pytest.approx((34 / 3, 2 / 3))


def test_matches_kalman1d_on_noisy_run():
    rng = random.Random(1)
    zs = [rng.gauss(5, 1) if rng.random() < 0.8 else None for _ in range(100)]
    a, b = Kalman1D(5.0, r=1, q=0.05), scalar_kf(5.0, r=1, q=0.05)
    for z in zs:
        a.step(z)
        b.step(z)
        assert b.x[0] == pytest.approx(a.x)
        assert b.P[0, 0] == pytest.approx(a.P)
    assert b.gate() == pytest.approx(a.gate())


# --- General filter: input checks ---

@pytest.mark.parametrize("kwargs", [
    dict(F=np.eye(3)),                    # wrong size
    dict(H=[[1, 0, 0]]),                  # wrong number of columns
    dict(R=[[1, 0], [0, 1]]),             # R doesn't match H rows
    dict(R=0),                            # zero sensor noise
    dict(R=-1),
])
def test_bad_shapes_and_noise(kwargs):
    base = dict(x0=[0, 0], P0=np.eye(2), F=np.eye(2), H=[[1, 0]], Q=np.zeros((2, 2)), R=1)
    base.update(kwargs)
    with pytest.raises(ValueError):
        KalmanFilter(**base)


def test_wrong_measurement_size():
    kf = ConstantVelocity([0, 0], r=1, q=0.1)
    with pytest.raises(ValueError):
        kf.update([1, 2, 3])
    with pytest.raises(ValueError):
        kf.update(None)


def test_input_arrays_not_aliased():
    x0 = np.array([1.0, 2.0])
    kf = KalmanFilter(x0=x0, P0=np.eye(2), F=np.eye(2), H=np.eye(2), Q=np.zeros((2, 2)), R=np.eye(2))
    kf.update([5, 5])
    assert list(x0) == [1.0, 2.0]


# --- distance() and gate() ---

def test_distance_in_standard_deviations():
    kf = scalar_kf(10, r=1, q=0)                 # S = P + R = 2
    assert kf.distance(10) == pytest.approx(0)
    assert kf.distance(10 + math.sqrt(2)) == pytest.approx(1)
    assert kf.distance(10 - 3 * math.sqrt(2)) == pytest.approx(3)


def test_distance_uses_shape_of_uncertainty():
    # 2D sensor that's much noisier in x than y: the same 1 m miss counts
    # for less in x than in y.
    kf = KalmanFilter(x0=[0, 0], P0=np.zeros((2, 2)), F=np.eye(2), H=np.eye(2),
                      Q=np.zeros((2, 2)), R=np.diag([4.0, 0.25]))
    assert kf.distance([1, 0]) == pytest.approx(0.5)
    assert kf.distance([0, 1]) == pytest.approx(2.0)


def test_gate_needs_1_number_measurement():
    with pytest.raises(ValueError):
        ConstantVelocity([0, 0], r=1, q=0.1).gate()
    with pytest.raises(ValueError):
        ConstantVelocity(0, r=1, q=0.1).gate(k=0)


def test_cv_gates_feed_gate_tracks():
    tracks = [ConstantVelocity(z, r=1, q=0.1) for z in (10, 11, 40)]
    groups = gate_tracks([kf.gate() for kf in tracks])
    assert [g.members for g in groups] == [[0, 1], [2]]


# --- Constant velocity ---

def run_cv(zs, r=1.0, q=1e-4, dt=1.0, **kw):
    kf = ConstantVelocity(zs[0], r=r, q=q, dt=dt, **kw)
    for z in zs[1:]:
        kf.step(z)
    return kf


def test_cv_no_lag_on_steady_speed():
    # The 1D filter lags >20 m on this target (see test_kalman.py); CV doesn't.
    truth = [2.0 * t for t in range(50)]
    kf = run_cv(truth)
    assert kf.position[0] == pytest.approx(truth[-1], abs=0.1)
    assert kf.velocity[0] == pytest.approx(2.0, abs=0.05)


def test_cv_noisy_estimate_within_3_sigma():
    rng = random.Random(0)
    r = 0.25
    truth = [5 + 1.5 * t for t in range(100)]
    zs = [p + rng.gauss(0, math.sqrt(r)) for p in truth]
    kf = run_cv(zs, r=r, q=0.01)
    assert abs(kf.position[0] - truth[-1]) <= 3 * math.sqrt(kf.P[0, 0])
    assert abs(kf.velocity[0] - 1.5) <= 3 * math.sqrt(kf.P[1, 1])


def test_cv_2d_diagonal_path():
    truth = [(1.0 * t, 10 - 2.0 * t) for t in range(30)]
    kf = run_cv(truth)
    assert kf.position == pytest.approx(truth[-1], abs=0.1)
    assert kf.velocity == pytest.approx([1.0, -2.0], abs=0.05)
    assert kf.x.shape == (4,)


def test_cv_3d():
    truth = [(t, 2.0 * t, -0.5 * t) for t in range(30)]
    kf = run_cv(truth)
    assert kf.velocity == pytest.approx([1.0, 2.0, -0.5], abs=0.05)


def test_cv_coasting_keeps_moving():
    kf = run_cv([2.0 * t for t in range(30)])
    p_before, var_before = kf.position[0], kf.P[0, 0]
    for _ in range(5):
        kf.step(None)
    assert kf.position[0] == pytest.approx(p_before + 10, abs=0.2)
    assert kf.P[0, 0] > var_before


def test_cv_irregular_time_steps():
    rng = random.Random(3)
    dts = [rng.uniform(0.1, 2.0) for _ in range(40)]
    t, kf = 0.0, ConstantVelocity(0.0, r=1e-4, q=1e-4)
    for dt in dts:
        t += dt
        kf.step(3.0 * t, dt=dt)
    assert kf.velocity[0] == pytest.approx(3.0, abs=0.01)
    assert kf.dt == dts[-1]


def test_cv_slow_start_with_small_speed_std():
    truth = [20.0 * t for t in range(6)]
    fast = run_cv(truth, speed_std=50.0)
    slow = run_cv(truth, speed_std=0.1)
    assert abs(fast.velocity[0] - 20) < abs(slow.velocity[0] - 20)


def test_cv_covariance_stays_symmetric_and_positive():
    rng = random.Random(4)
    kf = ConstantVelocity([0.0, 0.0], r=0.5, q=0.2)
    for t in range(1, 500):
        z = None if rng.random() < 0.2 else [t + rng.gauss(0, 0.7), -t + rng.gauss(0, 0.7)]
        kf.step(z)
    assert np.allclose(kf.P, kf.P.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(kf.P) > 0)


@pytest.mark.parametrize("kwargs", [
    dict(r=0), dict(r=-1), dict(q=-0.1), dict(dt=0), dict(speed_std=0),
])
def test_cv_bad_parameters(kwargs):
    base = dict(z0=0, r=1, q=0.1)
    base.update(kwargs)
    with pytest.raises(ValueError):
        ConstantVelocity(**base)


def test_cv_none_on_init():
    with pytest.raises(ValueError):
        ConstantVelocity(None, r=1, q=0.1)


def test_cv_uncertainty_is_honest():
    # Simulate targets that really do follow the CV model (random acceleration
    # with variance q), then check the filter's P matches its actual errors.
    # Average of e^T P^-1 e over many runs should be ~2 (number of states);
    # with 200 runs, 95% of the time it lands in about 1.72..2.28.
    # Too high = filter is overconfident (e.g. Q formula wrong), too low = too timid.
    rng = np.random.default_rng(0)
    r, q, dt = 0.5, 0.3, 0.7
    scores = []
    for _ in range(200):
        p, v = 0.0, rng.normal(0, 2.0)
        kf = ConstantVelocity(p + rng.normal(0, math.sqrt(r)), r=r, q=q, dt=dt, speed_std=2.0)
        for _ in range(60):
            a = rng.normal(0, math.sqrt(q))
            p, v = p + v * dt + a * dt ** 2 / 2, v + a * dt
            kf.step(p + rng.normal(0, math.sqrt(r)))
        e = np.array([p, v]) - kf.x
        scores.append(e @ np.linalg.solve(kf.P, e))
    assert 1.72 < np.mean(scores) < 2.28
