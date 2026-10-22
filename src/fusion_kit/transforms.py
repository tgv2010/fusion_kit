"""Coordinate conversions, and how measurement noise changes shape through them.

The core idea (first-order error propagation): if out = f(in) and the input
has covariance C, the output covariance is about  J C J^T, where J is the
Jacobian of f: J[i][j] = how much output i moves per unit of input j.
propagate_cov() does that for any J; the polar/spherical helpers below
are the common sensor cases already worked out.

Why it matters: a lidar/radar/sonar measures range and angle. Range error
stays the same size at any distance, but angle error turns into a sideways
position error that grows with range (arc length = r * dtheta). In x/y the
noise is a thin ellipse lying across the beam, not a circle. Feeding the
right ellipse to a Kalman filter as R gives tighter, better shaped gates.

Angle conventions (pick the one your sensor uses):
    "math"     theta = 0 along +x, increasing counter-clockwise toward +y
               x = r cos(theta), y = r sin(theta)
    "compass"  theta = 0 along +y, increasing clockwise toward +x
               x = r sin(theta), y = r cos(theta)   (common for spinning lidar)
All angles are radians.

Quick reference:
    polar_to_xy(r, theta)                       vectorized point conversion
    polar_to_cartesian_cov(r, theta, sr, sth)   2x2 x/y covariance -> use as R
    spherical_to_xyz / spherical_to_cartesian_cov   same in 3D
    covariance_ellipse(cov, k)                  axes + angle, for plotting gates
    propagate_cov(J, cov), numerical_jacobian(f, x)
    wrap_angle(a)                               into [-pi, pi)
"""

import math

import numpy as np

_CONVENTIONS = ("math", "compass")


def wrap_angle(a):
    """Wrap an angle (or array of angles) into [-pi, pi).

    Use on angle differences: 359 deg - 1 deg is a 2 deg miss, not 358.
    """
    wrapped = np.mod(np.asarray(a, dtype=float) + math.pi, 2 * math.pi) - math.pi
    return float(wrapped) if wrapped.ndim == 0 else wrapped


def propagate_cov(J, cov):
    """Return J @ cov @ J^T: the covariance after a change of coordinates with Jacobian J."""
    J = np.atleast_2d(np.asarray(J, dtype=float))
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    if cov.shape != (J.shape[1], J.shape[1]):
        raise ValueError(f"cov must be {J.shape[1]}x{J.shape[1]} to match J, got {cov.shape}")
    return J @ cov @ J.T


def numerical_jacobian(f, x, eps=1e-6):
    """Jacobian of f at x by central differences. Use when the math is tedious.

    Accurate to roughly eps**2 relative error for smooth f.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    f0 = np.atleast_1d(np.asarray(f(x), dtype=float))
    J = np.empty((len(f0), len(x)))
    for j in range(len(x)):
        h = eps * max(1.0, abs(x[j]))
        up, down = x.copy(), x.copy()
        up[j] += h
        down[j] -= h
        J[:, j] = (np.atleast_1d(f(up)) - np.atleast_1d(f(down))) / (2 * h)
    return J


def _check_convention(convention):
    if convention not in _CONVENTIONS:
        raise ValueError(f"convention must be one of {_CONVENTIONS}, got {convention!r}")


def polar_to_xy(r, theta, convention="math"):
    """Convert range/angle to x, y. Works on single values or whole arrays (a full sweep)."""
    _check_convention(convention)
    r, theta = np.asarray(r, dtype=float), np.asarray(theta, dtype=float)
    if convention == "math":
        return r * np.cos(theta), r * np.sin(theta)
    return r * np.sin(theta), r * np.cos(theta)


def polar_jacobian(r, theta, convention="math"):
    """2x2 Jacobian of (x, y) with respect to (r, theta)."""
    _check_convention(convention)
    c, s = math.cos(theta), math.sin(theta)
    if convention == "math":
        return np.array([[c, -r * s],
                         [s, r * c]])
    return np.array([[s, r * c],
                     [c, -r * s]])


def polar_to_cartesian_cov(r, theta, sigma_r, sigma_theta, convention="math"):
    """x/y covariance of one range/angle measurement. Use as R in a position filter.

    sigma_r      range noise std (m)
    sigma_theta  angle noise std (rad). For a scanning sensor with no better
                 spec, about half the beam spacing is a reasonable start.
    """
    if sigma_r < 0 or sigma_theta < 0:
        raise ValueError("sigma_r and sigma_theta must be >= 0")
    J = polar_jacobian(r, theta, convention)
    return propagate_cov(J, np.diag([sigma_r ** 2, sigma_theta ** 2]))


def spherical_to_xyz(r, azimuth, elevation):
    """Range/azimuth/elevation to x, y, z (vectorized).

    azimuth: from +x toward +y (math convention). elevation: up from the x-y plane.
    """
    r, az, el = (np.asarray(v, dtype=float) for v in (r, azimuth, elevation))
    return r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)


def spherical_to_cartesian_cov(r, azimuth, elevation, sigma_r, sigma_az, sigma_el):
    """3x3 x/y/z covariance of one range/azimuth/elevation measurement (3D lidar, radar)."""
    if min(sigma_r, sigma_az, sigma_el) < 0:
        raise ValueError("sigmas must be >= 0")
    ca, sa = math.cos(azimuth), math.sin(azimuth)
    ce, se = math.cos(elevation), math.sin(elevation)
    J = np.array([[ce * ca, -r * ce * sa, -r * se * ca],
                  [ce * sa, r * ce * ca, -r * se * sa],
                  [se, 0.0, r * ce]])
    return propagate_cov(J, np.diag([sigma_r ** 2, sigma_az ** 2, sigma_el ** 2]))


def covariance_ellipse(cov, k=1.0):
    """Return (semi_major, semi_minor, angle) of the k-sigma ellipse of a 2x2 covariance.

    angle is the major axis direction in radians, from +x toward +y.
    For matplotlib: Ellipse(center, 2*semi_major, 2*semi_minor, angle=math.degrees(angle)).
    A Kalman gate is covariance_ellipse(S, k) centered on the expected measurement,
    where S comes from kf.expected().
    """
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (2, 2):
        raise ValueError(f"cov must be 2x2, got {cov.shape}")
    if k <= 0:
        raise ValueError(f"k({k}) must be > 0")
    vals, vecs = np.linalg.eigh(cov)          # ascending eigenvalues
    if vals[0] < -1e-12 * max(1.0, abs(vals[1])):
        raise ValueError("cov must be positive semi-definite")
    vals = np.clip(vals, 0, None)
    major = vecs[:, 1]
    angle = math.atan2(major[1], major[0])
    if angle <= -math.pi / 2:                 # report in (-pi/2, pi/2]
        angle += math.pi
    elif angle > math.pi / 2:
        angle -= math.pi
    return k * math.sqrt(vals[1]), k * math.sqrt(vals[0]), angle
