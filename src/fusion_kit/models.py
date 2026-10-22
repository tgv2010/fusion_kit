"""Ready-made motion and sensor models for the Kalman filters.

A Motion says how the state moves between scans; a Sensor says what a
measurement of that state looks like. ExtendedKalmanFilter takes one
Motion and a default Sensor; update() can take a different Sensor per call,
which is how you fuse several sensors into one track.

State layout for the constant-velocity models: [positions..., velocities...],
e.g. 2D is [px, py, vx, vy].

Quick reference:
    constant_velocity(dims, q)          Motion: steady speed, random acceleration q
    initial_state(pos, pos_cov, speed_std)   x0, P0 for a new CV track
    position_sensor(dims, R)            Sensor: measures position directly (camera
                                        ground plane, GPS, lidar centroid in x/y)
    range_bearing(sigma_r, sigma_theta, doppler=False, sigma_rdot=None)
                                        Sensor: radar/sonar/lidar in polar, 2D,
                                        optionally with Doppler (range rate)
    bearing_only(sigma_theta)           Sensor: angle only (camera azimuth, passive sonar)
"""

import math
from typing import Callable, NamedTuple, Optional

import numpy as np

from fusion_kit.transforms import wrap_angle


class Motion(NamedTuple):
    f: Callable                       # f(x, dt) -> next state
    Q: Callable                       # Q(dt) -> process noise covariance
    F_jac: Optional[Callable] = None  # F_jac(x, dt) -> Jacobian of f; numerical if None
    dims: Optional[int] = None        # set for CV models: gives filters .position/.velocity


class Sensor(NamedTuple):
    h: Callable                          # h(x) -> expected measurement
    R: object                            # measurement noise covariance
    H_jac: Optional[Callable] = None     # H_jac(x) -> Jacobian of h; numerical if None
    residual: Optional[Callable] = None  # residual(z, z_expected); plain subtraction if None


def cv_matrices(dims, dt, q):
    """F and Q for constant velocity over one step of dt seconds.

    q is the acceleration variance, (m/s^2)^2. A random acceleration a held
    over dt moves position by a*dt^2/2 and velocity by a*dt, and those two
    errors are correlated; that's where the dt^4/4, dt^3/2, dt^2 terms come from.
    """
    if dt <= 0:
        raise ValueError(f"dt({dt}) must be > 0")
    if q < 0:
        raise ValueError(f"process noise q({q}) must be >= 0")
    I, Z = np.eye(dims), np.zeros((dims, dims))
    F = np.block([[I, dt * I], [Z, I]])
    Q = q * np.block([[dt ** 4 / 4 * I, dt ** 3 / 2 * I],
                      [dt ** 3 / 2 * I, dt ** 2 * I]])
    return F, Q


def constant_velocity(dims, q):
    """Motion model: position moves at steady velocity, nudged by random acceleration q."""
    return Motion(
        f=lambda x, dt: cv_matrices(dims, dt, q)[0] @ x,
        Q=lambda dt: cv_matrices(dims, dt, q)[1],
        F_jac=lambda x, dt: cv_matrices(dims, dt, q)[0],
        dims=dims,
    )


def initial_state(position, position_cov, speed_std):
    """x0, P0 for a new constant-velocity track: known position, unknown velocity.

    position_cov  covariance of the first position fix: a number (same variance
                  on every axis) or a matrix, e.g. from polar_to_cartesian_cov.
    speed_std     rough speed you expect (m/s); velocity starts at 0 with this spread.
    """
    if speed_std <= 0:
        raise ValueError(f"speed_std({speed_std}) must be > 0")
    p = np.atleast_1d(np.asarray(position, dtype=float))
    d = len(p)
    C = np.asarray(position_cov, dtype=float)
    C = C * np.eye(d) if C.ndim == 0 else C
    if C.shape != (d, d):
        raise ValueError(f"position_cov must be a number or {d}x{d}, got {C.shape}")
    x0 = np.concatenate([p, np.zeros(d)])
    P0 = np.block([[C, np.zeros((d, d))], [np.zeros((d, d)), speed_std ** 2 * np.eye(d)]])
    return x0, P0


def position_sensor(dims, R):
    """Sensor that measures position directly (state layout [positions, velocities])."""
    H = np.hstack([np.eye(dims), np.zeros((dims, dims))])
    R = np.asarray(R, dtype=float)
    R = R * np.eye(dims) if R.ndim == 0 else R
    return Sensor(h=lambda x: H @ x, R=R, H_jac=lambda x: H)


def _angle_and_grad(px, py, convention):
    """theta and d(theta)/d(px, py) for the chosen angle convention."""
    r2 = max(px * px + py * py, 1e-18)
    if convention == "math":
        return math.atan2(py, px), (-py / r2, px / r2)
    if convention == "compass":
        return math.atan2(px, py), (py / r2, -px / r2)
    raise ValueError(f"convention must be 'math' or 'compass', got {convention!r}")


def range_bearing(sigma_r, sigma_theta, doppler=False, sigma_rdot=None, convention="math"):
    """Sensor for a 2D CV state measured as [range, bearing] or [range, bearing, range_rate].

    Range rate (Doppler) is how fast the target closes or opens range, the
    part of velocity along the beam. Bearing residuals are wrapped, so a
    target crossing the +/-180 deg line doesn't look like a 360 deg miss.
    """
    if doppler and sigma_rdot is None:
        raise ValueError("doppler=True needs sigma_rdot")
    _angle_and_grad(1.0, 0.0, convention)     # validate convention now
    sigmas = [sigma_r, sigma_theta] + ([sigma_rdot] if doppler else [])
    if min(sigmas) <= 0:
        raise ValueError("sensor sigmas must be > 0")

    def h(x):
        px, py, vx, vy = x
        r = math.hypot(px, py)
        theta, _ = _angle_and_grad(px, py, convention)
        z = [r, theta]
        if doppler:
            z.append((px * vx + py * vy) / max(r, 1e-9))
        return np.array(z)

    def H_jac(x):
        px, py, vx, vy = x
        r = max(math.hypot(px, py), 1e-9)
        _, (dth_dpx, dth_dpy) = _angle_and_grad(px, py, convention)
        rows = [[px / r, py / r, 0.0, 0.0],
                [dth_dpx, dth_dpy, 0.0, 0.0]]
        if doppler:
            rdot = (px * vx + py * vy) / r
            rows.append([vx / r - rdot * px / r ** 2, vy / r - rdot * py / r ** 2, px / r, py / r])
        return np.array(rows)

    def residual(z, z_expected):
        y = np.asarray(z, dtype=float) - z_expected
        y[1] = wrap_angle(y[1])
        return y

    return Sensor(h=h, R=np.diag(np.square(sigmas)), H_jac=H_jac, residual=residual)


def bearing_only(sigma_theta, convention="math"):
    """Sensor for a 2D CV state that measures only the angle to the target.

    One bearing can't fix range; the filter learns it over time as the
    geometry changes (the observer or target must move across the line of sight).
    """
    if sigma_theta <= 0:
        raise ValueError("sigma_theta must be > 0")
    _angle_and_grad(1.0, 0.0, convention)

    def h(x):
        return np.array([_angle_and_grad(x[0], x[1], convention)[0]])

    def H_jac(x):
        _, (dpx, dpy) = _angle_and_grad(x[0], x[1], convention)
        return np.array([[dpx, dpy, 0.0, 0.0]])

    def residual(z, z_expected):
        return np.atleast_1d(wrap_angle(np.asarray(z, dtype=float) - z_expected))

    return Sensor(h=h, R=np.array([[sigma_theta ** 2]]), H_jac=H_jac, residual=residual)
