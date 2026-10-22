"""Kalman filters.

Which one to grab:
    Kalman1D          one number that should hold steady (no numpy needed)
    ConstantVelocity  a position (1D, 2D or 3D) that moves at roughly steady speed.
                      The usual choice for tracking.
    KalmanFilter      any linear model: you supply the matrices F, H, Q, R

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
    r  sensor position noise variance (same as Kalman1D).

Gating with N-D filters: distance(z) is how many standard deviations z is
from where the filter expected it, taking the shape of S into account
(Mahalanobis distance). Accept z if distance(z) <= k. The keep rate for
k=3 depends on how many numbers the sensor gives:
    1 number (range)            99.7%
    2 numbers (x, y)            98.9%
    3 numbers (x, y, z)         97.1%
gate() still returns (center, half_width) when the sensor gives 1 number,
for use with fusion_kit.gating.gate_tracks.
"""

import math

import numpy as np


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

    def predict(self):
        """Advance one time step with no measurement. Return (x, P)."""
        self.P = self.P + self.q
        return self.x, self.P

    def update(self, z):
        """Blend in measurement z. Call predict() first for each new time step. Return (x, P)."""
        if z is None:
            raise ValueError("update() needs a measurement; for a missed scan just call predict()")
        self.K = self.P / (self.P + self.r)
        self.x = self.x + self.K * (z - self.x)
        self.P = (1 - self.K) * self.P
        return self.x, self.P

    def step(self, z=None):
        """One full time step: predict, then update if z is not None. Return (x, P)."""
        self.predict()
        if z is not None:
            self.update(z)
        return self.x, self.P

    def gate(self, k=3.0):
        """Return (center, half_width) of the window the next measurement should land in.

        The next measurement is expected at x, with variance P + r (filter
        uncertainty plus sensor noise). k is how many standard deviations to
        allow: k=3 keeps ~99.7% of this target's real measurements.

        Call after predict() and before update(). The result is the
        (center, uncertainty) pair that fusion_kit.gating.gate_tracks takes.
        """
        if k <= 0:
            raise ValueError(f"k({k}) must be > 0")
        return self.x, k * math.sqrt(self.P + self.r)


def _matrix(a, shape, name):
    """Turn a number or nested list into a float matrix and check its shape."""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    if a.shape != shape:
        raise ValueError(f"{name} must be {shape[0]}x{shape[1]}, got {a.shape[0]}x{a.shape[1]}")
    return a


class KalmanFilter:
    """General linear Kalman filter. Scalars are fine for 1-state models."""

    def __init__(self, x0, P0, F, H, Q, R):
        self.x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
        n = len(self.x)
        self.P = _matrix(P0, (n, n), "P0")
        self.F = _matrix(F, (n, n), "F")
        self.Q = _matrix(Q, (n, n), "Q")
        self.H = np.atleast_2d(np.asarray(H, dtype=float))
        if self.H.ndim != 2 or self.H.shape[1] != n:
            raise ValueError(f"H must have {n} columns (one per state), got shape {self.H.shape}")
        m = self.H.shape[0]
        self.R = _matrix(R, (m, m), "R")
        try:
            np.linalg.cholesky(self.R)   # fails unless R is a valid (positive) covariance
        except np.linalg.LinAlgError:
            raise ValueError("R must be symmetric with positive variances") from None
        self.K = None

    @property
    def dim_z(self):
        """How many numbers each measurement has."""
        return self.H.shape[0]

    def predict(self):
        """Advance one step with no measurement. Return (x, P)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P

    def expected(self):
        """Return (z_center, S): where the next measurement should land and its covariance."""
        return self.H @ self.x, self.H @ self.P @ self.H.T + self.R

    def _residual(self, z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        if z.shape != (self.dim_z,):
            raise ValueError(f"measurement must have {self.dim_z} value(s), got {z.shape}")
        center, S = self.expected()
        return z - center, S

    def update(self, z):
        """Blend in measurement z. Call predict() first for each new time step. Return (x, P)."""
        if z is None:
            raise ValueError("update() needs a measurement; for a missed scan just call predict()")
        y, S = self._residual(z)
        # K = P H^T S^-1, computed with solve() rather than an explicit inverse.
        self.K = np.linalg.solve(S, self.H @ self.P).T
        self.x = self.x + self.K @ y
        # Joseph form: same answer as (I - K H) P on paper, but keeps P
        # symmetric and positive under rounding over long runs.
        I_KH = np.eye(len(self.x)) - self.K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + self.K @ self.R @ self.K.T
        return self.x, self.P

    def step(self, z=None):
        """One full time step: predict, then update if z is not None. Return (x, P)."""
        self.predict()
        if z is not None:
            self.update(z)
        return self.x, self.P

    def distance(self, z):
        """How many standard deviations z is from the expected measurement (Mahalanobis)."""
        y, S = self._residual(z)
        return float(math.sqrt(y @ np.linalg.solve(S, y)))

    def gate(self, k=3.0):
        """(center, half_width) for a 1-number measurement; feeds gating.gate_tracks.

        For multi-number measurements there's no single half-width; use distance(z) <= k.
        """
        if k <= 0:
            raise ValueError(f"k({k}) must be > 0")
        if self.dim_z != 1:
            raise ValueError("gate() needs a 1-number measurement; use distance(z) <= k instead")
        center, S = self.expected()
        return float(center[0]), k * math.sqrt(S[0, 0])


class ConstantVelocity(KalmanFilter):
    """Track a position that moves at a roughly steady speed, from position measurements.

    z0         first position measurement: a number (1D) or [x, y] / [x, y, z]
    r          sensor position noise variance (per axis)
    q          acceleration variance per step, (m/s^2)^2
    dt         default time between scans, seconds
    speed_std  rough speed you expect at the start (m/s). The first velocity is
               unknown, so it starts at 0 with this much uncertainty. Too small
               and the filter is slow to pick up a fast target.
    """

    def __init__(self, z0, r, q, dt=1.0, speed_std=10.0):
        if r <= 0:
            raise ValueError(f"sensor noise r({r}) must be > 0")
        if q < 0:
            raise ValueError(f"process noise q({q}) must be >= 0")
        if speed_std <= 0:
            raise ValueError(f"speed_std({speed_std}) must be > 0")
        if z0 is None:
            raise ValueError("first measurement z0 must not be None")
        z0 = np.atleast_1d(np.asarray(z0, dtype=float))
        d = len(z0)
        self.dims = d
        self.q = q
        super().__init__(
            x0=np.concatenate([z0, np.zeros(d)]),     # [positions, velocities]
            P0=np.diag([r] * d + [speed_std ** 2] * d),
            F=np.eye(2 * d),                          # set for real by _set_dt
            H=np.hstack([np.eye(d), np.zeros((d, d))]),
            Q=np.zeros((2 * d, 2 * d)),
            R=r * np.eye(d),
        )
        self._set_dt(dt)

    def _set_dt(self, dt):
        if dt <= 0:
            raise ValueError(f"dt({dt}) must be > 0")
        I, Z = np.eye(self.dims), np.zeros((self.dims, self.dims))
        self.dt = dt
        # position += velocity * dt; velocity unchanged
        self.F = np.block([[I, dt * I], [Z, I]])
        # Random acceleration with variance q over one step pushes position
        # by a*dt^2/2 and velocity by a*dt, and those two errors move together.
        self.Q = self.q * np.block([[dt ** 4 / 4 * I, dt ** 3 / 2 * I],
                                    [dt ** 3 / 2 * I, dt ** 2 * I]])

    def predict(self, dt=None):
        """Advance by dt seconds (default: the dt from the constructor). Return (x, P)."""
        if dt is not None and dt != self.dt:
            self._set_dt(dt)
        return super().predict()

    def step(self, z=None, dt=None):
        """Predict by dt, then update if z is not None. Return (x, P)."""
        self.predict(dt)
        if z is not None:
            self.update(z)
        return self.x, self.P

    @property
    def position(self):
        return self.x[:self.dims]

    @property
    def velocity(self):
        return self.x[self.dims:]
