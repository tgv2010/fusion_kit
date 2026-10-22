"""Kalman filters.

Which one to grab:
    Kalman1D              one number that should hold steady (no numpy needed)
    ConstantVelocity      a position (1D, 2D or 3D) moving at roughly steady speed,
                          measured in x/y(/z). The usual choice for tracking.
    KalmanFilter          any linear model: you supply the matrices F, H, Q, R
    ExtendedKalmanFilter  the sensor (or motion) isn't linear in the state:
                          radar range/bearing/Doppler, camera bearing, etc.
                          Pair it with fusion_kit.models.

All of them: predict() advances time, update(z) blends in a measurement,
step(z) does both (z=None for a missed scan), distance(z) / gate() say
whether a measurement belongs to this track.

Per-measurement noise and multi-sensor fusion: update(z, R=...) and
distance(z, R=...) take this measurement's own noise, so a detection at
long range or with low confidence can count for less. KalmanFilter also
takes H=..., and ExtendedKalmanFilter takes sensor=..., so one track can be
updated by different sensors on different scans (e.g. radar, then camera).

Kalman1D (from the "1D Kalman" puzzle)
--------------------------------------
Use it to smooth a noisy sensor reading of something that is roughly
constant, e.g. range to a parked car, a temperature, a bias. It's a
low-pass filter whose bandwidth tunes itself: it trusts new measurements a
lot at first, then less as its own estimate firms up.

The numbers (all variances, i.e. sigma squared, not sigma):
    x  current estimate
    P  variance of the estimate: how unsure the filter is about x
    r  sensor noise variance, from the datasheet (sigma_sensor ** 2)
    q  process noise: how much the true value can drift per step.
       q = 0 means "truly constant": the filter averages forever.
       Bigger q = forgets old data faster = follows changes, but noisier.
    K  gain, 0..1: how much of each new measurement to take in.
       K = P / (P + r). Unsure filter (big P) -> K near 1 -> follow the sensor.

Each time step:
    predict()   P grows by q (time passed, so we know less). x stays put.
    update(z)   blend in measurement z:  x += K * (z - x),  P shrinks.
A missed scan is just predict() with no update: P keeps growing.

Typical loops:
    Smoothing only:   x, P = kf.step(z)        # z may be None for a miss
    Tracking:         kf.predict()
                      center, half = kf.gate()  # is this z ours?
                      if abs(z - center) <= half: kf.update(z)

Limit: the model assumes the value doesn't move. A target moving at
constant speed will be tracked with a lag. Use ConstantVelocity for that.

KalmanFilter (general, N-dimensional)
-------------------------------------
Same two steps as Kalman1D, with numbers replaced by vectors and matrices:
    x  state vector, e.g. [position, velocity]
    P  state covariance: variances on the diagonal, how errors in
       different states move together off the diagonal
    F  state transition: how the state evolves in one step (x_next = F x)
    H  measurement matrix: which part of the state the sensor sees (z = H x)
    Q  process noise covariance: how much the model is wrong per step
    R  sensor noise covariance (diagonal = each channel's sigma ** 2)
    K  gain matrix: P H^T S^-1, where S = H P H^T + R is the spread the next
       measurement is expected to have. Same idea as P / (P + r).

ConstantVelocity
----------------
A KalmanFilter set up for "position moves at a roughly steady speed".
The state is [positions..., velocities...] and the sensor measures position
only; the filter works out velocity from how position changes. It has no
lag on a steady-speed target, and it keeps moving the estimate forward
through missed scans.
    q  how hard the target can accelerate: variance of acceleration per
       step, (m/s^2)^2. Small q = smooth, slow to react to turns; big q = jumpy.
    r  sensor position noise variance (same as Kalman1D), or a full
       covariance matrix, e.g. from transforms.polar_to_cartesian_cov.

ExtendedKalmanFilter
--------------------
Same steps, but the motion f(x, dt) and measurement h(x) can be any
smooth functions. Each step it straightens them out locally (uses their
Jacobians, the matrix of slopes) and runs the ordinary Kalman math on that.
Works well when the functions are close to straight over the size of the
uncertainty, which is true for typical radar/camera tracking. Build it from
fusion_kit.models:
    from fusion_kit.models import constant_velocity, initial_state, range_bearing
    x0, P0 = initial_state(first_xy, first_xy_cov, speed_std=2.0)
    ekf = ExtendedKalmanFilter(x0, P0, constant_velocity(2, q=1.0),
                               range_bearing(sigma_r=0.1, sigma_theta=0.02))

Gating with N-D filters: distance(z) is how many standard deviations z is
from where the filter expected it, taking the shape of S into account
(Mahalanobis distance). Accept z if distance(z) <= k. The keep rate for
k=3 depends on how many numbers the sensor gives:
    1 number (range)            99.7%
    2 numbers (x, y)            98.9%
    3 numbers (x, y, z)         97.1%
gate() still returns (center, half_width) when the sensor gives 1 number,
for use with fusion_kit.gating.gate_tracks. To draw the 2D gate ellipse:
transforms.covariance_ellipse(S, k) with S from expected().
"""

import math

import numpy as np

from fusion_kit.models import cv_matrices, initial_state
from fusion_kit.transforms import numerical_jacobian


class Kalman1D:
    def __init__(self, z0, r, q=0.0):
        """Start the filter at the first measurement z0.

        The initial variance is r: we know x as well as one sensor reading.
        """
        if r <= 0:
            raise ValueError(f"sensor noise r({r}) must be > 0")
        if q < 0:
            raise ValueError(f"process noise q({q}) must be >= 0")
        if z0 is None:
            raise ValueError("first measurement z0 must not be None")
        self.r = r
        self.q = q
        self.x = z0      # estimate
        self.P = r       # variance of the estimate
        self.K = None    # gain used by the last update (None until the first one)

    def _r(self, r):
        if r is None:
            return self.r
        if r <= 0:
            raise ValueError(f"sensor noise r({r}) must be > 0")
        return r

    def predict(self):
        """Advance one time step with no measurement. Return (x, P)."""
        self.P = self.P + self.q
        return self.x, self.P

    def update(self, z, r=None):
        """Blend in measurement z. Call predict() first for each new time step. Return (x, P).

        r: this measurement's noise variance, if different from the default.
        """
        if z is None:
            raise ValueError("update() needs a measurement; for a missed scan just call predict()")
        r = self._r(r)
        self.K = self.P / (self.P + r)
        self.x = self.x + self.K * (z - self.x)
        self.P = (1 - self.K) * self.P
        return self.x, self.P

    def step(self, z=None, r=None):
        """One full time step: predict, then update if z is not None. Return (x, P)."""
        self.predict()
        if z is not None:
            self.update(z, r)
        return self.x, self.P

    def gate(self, k=3.0, r=None):
        """Return (center, half_width) of the window the next measurement should land in.

        The next measurement is expected at x, with variance P + r (filter
        uncertainty plus sensor noise). k is how many standard deviations to
        allow: k=3 keeps ~99.7% of this target's real measurements.

        Call after predict() and before update(). The result is the
        (center, uncertainty) pair that fusion_kit.gating.gate_tracks takes.
        """
        if k <= 0:
            raise ValueError(f"k({k}) must be > 0")
        return self.x, k * math.sqrt(self.P + self._r(r))


# ---------------------------------------------------------------- shared helpers

def _matrix(a, shape, name):
    """Turn a number or nested list into a float matrix and check its shape."""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    if a.shape != shape:
        raise ValueError(f"{name} must be {shape[0]}x{shape[1]}, got {a.shape[0]}x{a.shape[1]}")
    return a


def _covariance(R, m, name="R"):
    """Check R is an m x m valid covariance (symmetric, positive variances)."""
    R = _matrix(R, (m, m), name)
    try:
        np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        raise ValueError(f"{name} must be symmetric with positive variances") from None
    return R


def _measurement(z, m):
    if z is None:
        raise ValueError("update() needs a measurement; for a missed scan just call predict()")
    z = np.atleast_1d(np.asarray(z, dtype=float))
    if z.shape != (m,):
        raise ValueError(f"measurement must have {m} value(s), got {z.shape}")
    return z


def _kalman_update(x, P, y, H, R):
    """The update math shared by every filter. y is the residual z - expected."""
    S = H @ P @ H.T + R
    K = np.linalg.solve(S, H @ P).T      # K = P H^T S^-1, without an explicit inverse
    x = x + K @ y
    # Joseph form: same answer as (I - K H) P on paper, but keeps P
    # symmetric and positive under rounding over long runs.
    I_KH = np.eye(len(x)) - K @ H
    P = I_KH @ P @ I_KH.T + K @ R @ K.T
    return x, P, K


def _mahalanobis(y, S):
    return float(math.sqrt(y @ np.linalg.solve(S, y)))


def _gate(center, S, k):
    if k <= 0:
        raise ValueError(f"k({k}) must be > 0")
    if len(center) != 1:
        raise ValueError("gate() needs a 1-number measurement; use distance(z) <= k instead")
    return float(center[0]), k * math.sqrt(S[0, 0])


# ---------------------------------------------------------------- linear filter

class KalmanFilter:
    """General linear Kalman filter. Scalars are fine for 1-state models."""

    def __init__(self, x0, P0, F, H, Q, R):
        self.x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
        n = len(self.x)
        self.P = _matrix(P0, (n, n), "P0")
        self.F = _matrix(F, (n, n), "F")
        self.Q = _matrix(Q, (n, n), "Q")
        self.H = self._H(H)
        self.R = _covariance(R, self.H.shape[0])
        self.K = None

    def _H(self, H):
        H = np.atleast_2d(np.asarray(H, dtype=float))
        if H.ndim != 2 or H.shape[1] != len(self.x):
            raise ValueError(f"H must have {len(self.x)} columns (one per state), got shape {H.shape}")
        return H

    def _model(self, H, R):
        """The (H, R) to use for one measurement: defaults unless overridden."""
        H = self.H if H is None else self._H(H)
        if R is None:
            if H is not self.H and H.shape[0] != self.H.shape[0]:
                raise ValueError("a different-size H needs its own R")
            R = self.R
        else:
            R = _covariance(R, H.shape[0])
        return H, R

    @property
    def dim_z(self):
        """How many numbers each measurement has (default sensor)."""
        return self.H.shape[0]

    def predict(self, dt=None):
        """Advance one step with no measurement. Return (x, P)."""
        if dt is not None:
            raise ValueError("this filter has a fixed F; use ConstantVelocity or "
                             "ExtendedKalmanFilter for a variable time step")
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P

    def expected(self, H=None, R=None):
        """Return (z_center, S): where the next measurement should land and its covariance."""
        H, R = self._model(H, R)
        return H @ self.x, H @ self.P @ H.T + R

    def _residual(self, z, H, R):
        H, R = self._model(H, R)
        z = _measurement(z, H.shape[0])
        return z - H @ self.x, H, R

    def update(self, z, R=None, H=None):
        """Blend in measurement z. Call predict() first for each new time step. Return (x, P).

        R: this measurement's noise covariance, if different from the default.
        H: this measurement's sensor matrix, for a sensor that sees a different
           part of the state (multi-sensor fusion). Give R too if its size differs.
        """
        y, H, R = self._residual(z, H, R)
        self.x, self.P, self.K = _kalman_update(self.x, self.P, y, H, R)
        return self.x, self.P

    def step(self, z=None, R=None, H=None):
        """One full time step: predict, then update if z is not None. Return (x, P)."""
        self.predict()
        if z is not None:
            self.update(z, R, H)
        return self.x, self.P

    def distance(self, z, R=None, H=None):
        """How many standard deviations z is from the expected measurement (Mahalanobis)."""
        y, H, R = self._residual(z, H, R)
        return _mahalanobis(y, H @ self.P @ H.T + R)

    def gate(self, k=3.0, R=None, H=None):
        """(center, half_width) for a 1-number measurement; feeds gating.gate_tracks.

        For multi-number measurements there's no single half-width; use distance(z) <= k.
        """
        return _gate(*self.expected(H, R), k)


class ConstantVelocity(KalmanFilter):
    """Track a position that moves at a roughly steady speed, from position measurements.

    z0         first position measurement: a number (1D) or [x, y] / [x, y, z]
    r          sensor position noise: a variance (same on every axis) or a covariance matrix
    q          acceleration variance per step, (m/s^2)^2
    dt         default time between scans, seconds
    speed_std  rough speed you expect at the start (m/s). The first velocity is
               unknown, so it starts at 0 with this much uncertainty. Too small
               and the filter is slow to pick up a fast target.
    """

    def __init__(self, z0, r, q, dt=1.0, speed_std=10.0):
        if z0 is None:
            raise ValueError("first measurement z0 must not be None")
        if np.ndim(r) == 0 and r <= 0:
            raise ValueError(f"sensor noise r({r}) must be > 0")
        if q < 0:
            raise ValueError(f"process noise q({q}) must be >= 0")
        z0 = np.atleast_1d(np.asarray(z0, dtype=float))
        d = len(z0)
        R = _covariance(r * np.eye(d) if np.ndim(r) == 0 else r, d, "r")
        self.dims = d
        self.q = q
        x0, P0 = initial_state(z0, R, speed_std)
        F, Q = cv_matrices(d, dt, q)
        super().__init__(x0=x0, P0=P0, F=F, H=np.hstack([np.eye(d), np.zeros((d, d))]), Q=Q, R=R)
        self.dt = dt

    def predict(self, dt=None):
        """Advance by dt seconds (default: the dt from the constructor). Return (x, P)."""
        if dt is not None and dt != self.dt:
            self.F, self.Q = cv_matrices(self.dims, dt, self.q)
            self.dt = dt
        return super().predict()

    def step(self, z=None, dt=None, R=None):
        """Predict by dt, then update if z is not None. Return (x, P)."""
        self.predict(dt)
        if z is not None:
            self.update(z, R)
        return self.x, self.P

    @property
    def position(self):
        return self.x[:self.dims]

    @property
    def velocity(self):
        return self.x[self.dims:]


# ---------------------------------------------------------------- extended filter

class ExtendedKalmanFilter:
    """Kalman filter for nonlinear motion and/or measurement models.

    motion  fusion_kit.models.Motion: f(x, dt), Q(dt), optional F_jac(x, dt)
    sensor  fusion_kit.models.Sensor: h(x), R, optional H_jac(x), residual(z, z_exp)
            (the default sensor; update(z, sensor=...) can use others)
    dt      default time step for predict()
    Missing Jacobians are computed numerically.
    """

    def __init__(self, x0, P0, motion, sensor, dt=1.0):
        self.x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
        n = len(self.x)
        self.P = _matrix(P0, (n, n), "P0")
        if dt <= 0:
            raise ValueError(f"dt({dt}) must be > 0")
        self.motion = motion
        self.sensor = sensor
        self.dt = dt
        self.K = None
        _covariance(sensor.R, len(np.atleast_1d(sensor.h(self.x))), "sensor.R")

    def predict(self, dt=None):
        """Advance by dt seconds (default: the constructor's dt). Return (x, P)."""
        dt = self.dt if dt is None else dt
        if dt <= 0:
            raise ValueError(f"dt({dt}) must be > 0")
        m = self.motion
        F = (m.F_jac(self.x, dt) if m.F_jac is not None
             else numerical_jacobian(lambda s: m.f(s, dt), self.x))
        Q = m.Q(dt) if callable(m.Q) else m.Q
        self.x = np.asarray(m.f(self.x, dt), dtype=float)
        self.P = F @ self.P @ F.T + _matrix(Q, self.P.shape, "Q")
        return self.x, self.P

    def _linearize(self, sensor, R):
        sensor = self.sensor if sensor is None else sensor
        z_exp = np.atleast_1d(np.asarray(sensor.h(self.x), dtype=float))
        H = (np.atleast_2d(sensor.H_jac(self.x)) if sensor.H_jac is not None
             else numerical_jacobian(sensor.h, self.x))
        R = _covariance(sensor.R if R is None else R, len(z_exp))
        return sensor, z_exp, H, R

    def expected(self, sensor=None, R=None):
        """Return (z_center, S) for the next measurement from this sensor."""
        _, z_exp, H, R = self._linearize(sensor, R)
        return z_exp, H @ self.P @ H.T + R

    def _residual(self, z, sensor, R):
        sensor, z_exp, H, R = self._linearize(sensor, R)
        z = _measurement(z, len(z_exp))
        y = sensor.residual(z, z_exp) if sensor.residual is not None else z - z_exp
        return np.asarray(y, dtype=float), H, R

    def update(self, z, R=None, sensor=None):
        """Blend in measurement z from sensor (default: the constructor's). Return (x, P)."""
        y, H, R = self._residual(z, sensor, R)
        self.x, self.P, self.K = _kalman_update(self.x, self.P, y, H, R)
        return self.x, self.P

    def step(self, z=None, dt=None, R=None, sensor=None):
        """Predict by dt, then update if z is not None. Return (x, P)."""
        self.predict(dt)
        if z is not None:
            self.update(z, R, sensor)
        return self.x, self.P

    def distance(self, z, R=None, sensor=None):
        """How many standard deviations z is from the expected measurement (Mahalanobis)."""
        y, H, R = self._residual(z, sensor, R)
        return _mahalanobis(y, H @ self.P @ H.T + R)

    def gate(self, k=3.0, R=None, sensor=None):
        """(center, half_width) for a 1-number measurement; see KalmanFilter.gate."""
        return _gate(*self.expected(sensor, R), k)

    @property
    def position(self):
        """First `dims` states, when the motion model defines dims (CV models do)."""
        if self.motion.dims is None:
            raise AttributeError("motion model has no dims; read self.x directly")
        return self.x[:self.motion.dims]

    @property
    def velocity(self):
        if self.motion.dims is None:
            raise AttributeError("motion model has no dims; read self.x directly")
        d = self.motion.dims
        return self.x[d:2 * d]
