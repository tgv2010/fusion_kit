import math

import numpy as np
import pytest

from fusion_kit.transforms import (covariance_ellipse, numerical_jacobian, polar_jacobian,
                                   polar_to_cartesian_cov, polar_to_xy, propagate_cov,
                                   spherical_to_cartesian_cov, spherical_to_xyz, wrap_angle)


def test_wrap_angle():
    assert wrap_angle(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    assert wrap_angle(math.radians(359) - math.radians(1)) == pytest.approx(math.radians(-2))
    assert isinstance(wrap_angle(1.0), float)
    assert wrap_angle([0, 2 * math.pi, -3 * math.pi]) == pytest.approx([0, 0, -math.pi])


def test_polar_to_xy_conventions():
    assert polar_to_xy(2, 0) == pytest.approx((2, 0))
    assert polar_to_xy(2, math.pi / 2) == pytest.approx((0, 2))
    assert polar_to_xy(2, 0, "compass") == pytest.approx((0, 2))             # 0 = +y
    assert polar_to_xy(2, math.pi / 2, "compass") == pytest.approx((2, 0))   # clockwise to +x


def test_polar_to_xy_vectorized():
    x, y = polar_to_xy(np.array([1.0, 2.0]), np.array([0.0, math.pi]))
    assert x == pytest.approx([1, -2])
    assert y == pytest.approx([0, 0], abs=1e-12)


def test_bad_convention():
    with pytest.raises(ValueError):
        polar_to_xy(1, 0, "degrees")


@pytest.mark.parametrize("convention", ["math", "compass"])
def test_polar_jacobian_matches_numerical(convention):
    r, th = 7.0, 0.8
    f = lambda v: np.array(polar_to_xy(v[0], v[1], convention))
    assert polar_jacobian(r, th, convention) == pytest.approx(numerical_jacobian(f, [r, th]))


def test_polar_cov_is_thin_ellipse_across_beam():
    # Target straight down +x: range error along x, angle error (r * sigma_theta) along y.
    cov = polar_to_cartesian_cov(10.0, 0.0, sigma_r=0.02, sigma_theta=0.01)
    assert cov == pytest.approx(np.diag([0.02 ** 2, (10 * 0.01) ** 2]))


def test_polar_cov_matches_monte_carlo():
    rng = np.random.default_rng(0)
    r, th, sr, sth = 8.0, 1.1, 0.05, 0.01
    rs = r + rng.normal(0, sr, 200_000)
    ths = th + rng.normal(0, sth, 200_000)
    sampled = np.cov(np.vstack(polar_to_xy(rs, ths)))
    assert polar_to_cartesian_cov(r, th, sr, sth) == pytest.approx(sampled, rel=0.03, abs=1e-5)


def test_spherical():
    assert spherical_to_xyz(2, 0, 0) == pytest.approx((2, 0, 0))
    assert spherical_to_xyz(2, math.pi / 2, 0) == pytest.approx((0, 2, 0), abs=1e-12)
    assert spherical_to_xyz(2, 0, math.pi / 2) == pytest.approx((0, 0, 2), abs=1e-12)
    r, az, el = 5.0, 0.7, 0.3
    f = lambda v: np.array(spherical_to_xyz(*v))
    J = numerical_jacobian(f, [r, az, el])
    sig = np.diag([0.1 ** 2, 0.02 ** 2, 0.03 ** 2])
    assert spherical_to_cartesian_cov(r, az, el, 0.1, 0.02, 0.03) == pytest.approx(J @ sig @ J.T)


def test_negative_sigma_rejected():
    with pytest.raises(ValueError):
        polar_to_cartesian_cov(1, 0, -0.1, 0.1)
    with pytest.raises(ValueError):
        spherical_to_cartesian_cov(1, 0, 0, 0.1, -0.1, 0.1)


def test_propagate_cov_shape_check():
    with pytest.raises(ValueError):
        propagate_cov(np.eye(2), np.eye(3))


def test_covariance_ellipse():
    a, b, ang = covariance_ellipse(np.diag([4.0, 1.0]))
    assert (a, b, ang) == pytest.approx((2, 1, 0))
    a, b, ang = covariance_ellipse(np.diag([1.0, 9.0]), k=2)
    assert (a, b) == pytest.approx((6, 2))
    assert abs(ang) == pytest.approx(math.pi / 2)


def test_covariance_ellipse_rotated():
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    Rot = np.array([[c, -s], [s, c]])
    cov = Rot @ np.diag([9.0, 1.0]) @ Rot.T
    a, b, ang = covariance_ellipse(cov)
    assert (a, b, ang) == pytest.approx((3, 1, math.radians(30)))


def test_covariance_ellipse_bad_input():
    with pytest.raises(ValueError):
        covariance_ellipse(np.eye(3))
    with pytest.raises(ValueError):
        covariance_ellipse(np.diag([1.0, -1.0]))
    with pytest.raises(ValueError):
        covariance_ellipse(np.eye(2), k=0)


def test_numerical_jacobian_linear_is_exact():
    A = np.array([[1.0, 2.0, 0.0], [0.0, -1.0, 3.0]])
    assert numerical_jacobian(lambda v: A @ v, [1.0, 2.0, 3.0]) == pytest.approx(A)
