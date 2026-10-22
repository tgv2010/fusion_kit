"""1D Kalman filter for a value that should hold steady (from the "1D Kalman" puzzle).

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
constant speed will be tracked with a lag. That needs a position+velocity
(2D) filter.
"""

import math


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
