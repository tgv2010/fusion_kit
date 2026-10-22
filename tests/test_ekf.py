import math

import numpy as np
import pytest

from fusion_kit.kalman import ConstantVelocity, ExtendedKalmanFilter
from fusion_kit.models import (Motion, Sensor, bearing_only, constant_velocity, initial_state,
                               position_sensor, range_bearing)
from fusion_kit.transforms import numerical_jacobian, polar_to_cartesian_cov, polar_to_xy


@pytest.mark.parametrize("convention", ["math", "compass"])
@pytest.mark.parametrize("doppler", [False, True])
def test_range_bearing_jacobian(convention, doppler):
    s = range_bearing(0.1, 0.01, doppler=doppler, sigma_rdot=0.2, convention=convention)
    x = np.array([3.0, -4.0, 1.5, 0.7])
    assert s.H_jac(x) == pytest.approx(numerical_jacobian(s.h, x), abs=1e-7)


def test_range_bearing_values():
    s = range_bearing(0.1, 0.01, doppler=True, sigma_rdot=0.2)
    r, th, rdot = s.h(np.array([3.0, 4.0, 3.0, 4.0]))       # moving straight away
    assert (r, th, rdot) == pytest.approx((5, math.atan2(4, 3), 5))
    assert s.R == pytest.approx(np.diag([0.01, 1e-4, 0.04]))


def test_bearing_only_jacobian():
    s = bearing_only(0.01, convention="compass")
    x = np.array([2.0, 5.0, 0.0, 0.0])
    assert s.H_jac(x) == pytest.approx(numerical_jacobian(s.h, x))


@pytest.mark.parametrize("kwargs", [
    dict(sigma_r=0, sigma_theta=0.1), dict(sigma_r=0.1, sigma_theta=0.1, doppler=True),
    dict(sigma_r=0.1, sigma_theta=0.1, convention="degrees"),
])
def test_range_bearing_bad_args(kwargs):
    with pytest.raises(ValueError):
        range_bearing(**kwargs)


def test_initial_state():
    x0, P0 = initial_state([1, 2], 0.25, speed_std=3)
    assert list(x0) == [1, 2, 0, 0]
    assert P0 == pytest.approx(np.diag([0.25, 0.25, 9, 9]))
    with pytest.raises(ValueError):
        initial_state([1, 2], np.eye(3), 1)


def test_ekf_with_linear_models_equals_kalman_filter():
    rng = np.random.default_rng(0)
    kf = ConstantVelocity([0.0, 0.0], r=0.5, q=0.3, dt=0.1, speed_std=2.0)
    x0, P0 = initial_state([0.0, 0.0], 0.5, speed_std=2.0)
    ekf = ExtendedKalmanFilter(x0, P0, constant_velocity(2, 0.3), position_sensor(2, 0.5), dt=0.1)
    for t in range(1, 60):
        z = None if rng.random() < 0.2 else [0.3 * t + rng.normal(0, 0.7), rng.normal(0, 0.7)]
        kf.step(z)
        ekf.step(z)
        assert ekf.x == pytest.approx(kf.x)
        assert ekf.P == pytest.approx(kf.P)


def simulate_radar(rng, steps=80, dt=0.1, sr=0.1, sth=0.01, doppler=False):
    p, v = np.array([20.0, -5.0]), np.array([-3.0, 4.0])
    truth, zs = [], []
    for _ in range(steps):
        p = p + v * dt
        truth.append((p.copy(), v.copy()))
        r, th = math.hypot(*p), math.atan2(p[1], p[0])
        z = [r + rng.normal(0, sr), th + rng.normal(0, sth)]
        if doppler:
            z.append(p @ v / r + rng.normal(0, 0.1))
        zs.append(np.array(z))
    return truth, zs


def radar_ekf(z0, doppler=False, sr=0.1, sth=0.01):
    xy = np.array(polar_to_xy(z0[0], z0[1]))
    x0, P0 = initial_state(xy, polar_to_cartesian_cov(z0[0], z0[1], sr, sth), speed_std=5.0)
    sensor = range_bearing(sr, sth, doppler=doppler, sigma_rdot=0.1)
    return ExtendedKalmanFilter(x0, P0, constant_velocity(2, q=0.5), sensor, dt=0.1)


def test_ekf_tracks_radar_target():
    rng = np.random.default_rng(1)
    truth, zs = simulate_radar(rng)
    ekf = radar_ekf(zs[0])
    for z in zs[1:]:
        ekf.step(z)
    p, v = truth[-1]
    assert np.linalg.norm(ekf.position - p) < 3 * math.sqrt(np.trace(ekf.P[:2, :2]))
    assert np.linalg.norm(ekf.velocity - v) < 1.0


def test_doppler_pins_down_velocity_faster():
    rng = np.random.default_rng(2)
    _, zs = simulate_radar(rng, steps=6, doppler=True)
    with_d, without = radar_ekf(zs[0], doppler=True), radar_ekf(zs[0])
    for z in zs[1:]:
        with_d.step(z)
        without.step(z[:2])
    assert np.trace(with_d.P[2:, 2:]) < np.trace(without.P[2:, 2:])


def test_bearing_wraps_across_180_degrees():
    # Target behind the sensor crossing the -x axis: bearing jumps from +179 to -179 deg.
    sr, sth = 0.05, 0.005
    sensor = range_bearing(sr, sth)
    x0, P0 = initial_state([-10.0, 0.5], 0.01, speed_std=1.0)
    x0[3] = -1.0                                      # moving in -y
    ekf = ExtendedKalmanFilter(x0, P0, constant_velocity(2, q=0.01), sensor, dt=0.1)
    for k in range(1, 11):
        p = np.array([-10.0, 0.5 - 0.1 * k])
        z = [math.hypot(*p), math.atan2(p[1], p[0])]
        ekf.predict()
        assert ekf.distance(z) < 3
        ekf.update(z)
    assert ekf.position == pytest.approx([-10.0, -0.5], abs=0.05)


def test_numerical_jacobian_fallback_matches_analytic():
    rng = np.random.default_rng(3)
    _, zs = simulate_radar(rng, steps=20)
    a = radar_ekf(zs[0])
    b = radar_ekf(zs[0])
    s = b.sensor
    b.sensor = Sensor(h=s.h, R=s.R, residual=s.residual)               # no H_jac
    m = b.motion
    b.motion = Motion(f=m.f, Q=m.Q, dims=m.dims)                        # no F_jac
    for z in zs[1:]:
        a.step(z)
        b.step(z)
    assert b.x == pytest.approx(a.x, rel=1e-5)
    assert b.P == pytest.approx(a.P, rel=1e-4)


def test_fuse_radar_and_camera_bearing():
    rng = np.random.default_rng(4)
    _, zs = simulate_radar(rng, steps=10)
    ekf = radar_ekf(zs[0])
    for z in zs[1:]:
        ekf.step(z)
    camera = bearing_only(0.002)                      # sharper angle than the radar
    ekf.predict()
    before = ekf.P.copy()
    bearing = math.atan2(ekf.position[1], ekf.position[0])
    ekf.update([bearing], sensor=camera)
    assert np.trace(ekf.P[:2, :2]) < np.trace(before[:2, :2])
    assert ekf.distance([bearing], sensor=camera) < 1


def test_ekf_r_override_and_gate():
    rng = np.random.default_rng(5)
    _, zs = simulate_radar(rng, steps=3)
    ekf = radar_ekf(zs[0])
    ekf.predict()
    z = zs[1]
    loose = np.diag([1.0, 0.1])
    assert ekf.distance(z, R=loose) < ekf.distance(z)
    with pytest.raises(ValueError):
        ekf.gate()                                    # 2-number sensor
    rng_only = Sensor(h=lambda x: np.array([math.hypot(x[0], x[1])]), R=[[0.01]])
    center, half = ekf.gate(sensor=rng_only)
    assert center == pytest.approx(math.hypot(*ekf.position))
    assert half > 0


def test_ekf_bad_input():
    x0, P0 = initial_state([1.0, 1.0], 0.1, 1.0)
    with pytest.raises(ValueError):
        ExtendedKalmanFilter(x0, P0, constant_velocity(2, 0.1), range_bearing(0.1, 0.01), dt=0)
    ekf = ExtendedKalmanFilter(x0, P0, constant_velocity(2, 0.1), range_bearing(0.1, 0.01))
    with pytest.raises(ValueError):
        ekf.update(None)
    with pytest.raises(ValueError):
        ekf.update([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        ekf.predict(dt=-1)


def test_position_needs_dims():
    motion = Motion(f=lambda x, dt: x, Q=lambda dt: np.zeros((2, 2)))
    ekf = ExtendedKalmanFilter([1.0, 2.0], np.eye(2), motion, Sensor(h=lambda x: x, R=np.eye(2)))
    with pytest.raises(AttributeError):
        ekf.position
