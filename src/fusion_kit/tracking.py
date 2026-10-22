"""Multi-target track management: predict, associate, update, birth, confirm, delete.

The loop every multi-target tracker runs, sensor-agnostic:

    each scan at time t with detections [z0, z1, ...]:
      1. predict every live track forward to t
      2. associate detections to tracks (fusion_kit.association, gated)
      3. update matched tracks; tracks with no detection coast
      4. unmatched detections start new tentative tracks
      5. M-of-N (fusion_kit.windows) confirms tentative tracks
      6. delete tentative tracks that miss too often, and confirmed
         tracks that coast longer than coast_time

You supply new_filter(z, R) -> a filter for a fresh track. Any fusion_kit
filter works as long as its predict() accepts dt (ConstantVelocity,
ExtendedKalmanFilter).

Example (2D lidar/radar centroids):
    from fusion_kit.kalman import ConstantVelocity
    from fusion_kit.tracking import TrackManager

    tm = TrackManager(lambda z, R: ConstantVelocity(z, r=R if R is not None else 0.04,
                                                    q=1.0, speed_std=2.0))
    for t, detections in scans:
        tm.step(t, detections)
    good = [tr for tr in tm.all_tracks() if tr.confirmed]

Knobs:
    gate                  cost threshold; default cost is squared Mahalanobis,
                          so 9.0 = 3 sigma. Must match your cost's scale.
    confirm=(m, n)        confirm after m hits in the last n scans
    tentative_max_misses  unconfirmed track dies after more misses in a row than this
    coast_time            confirmed track dies after this many seconds without a hit
    cost(track, z, R)     custom match cost (e.g. add a hard distance cap, or IoU)
    method                'optimal' or 'greedy' association

Per-scan options on step(): R (one noise covariance per detection, e.g. range
dependent), birth=False (this sensor can confirm tracks but not start them,
e.g. a bearing-only camera), and extra keywords (H=..., sensor=...) passed to
the filters' update/distance, which is how a second sensor updates the same tracks.
"""

from dataclasses import dataclass, field

import numpy as np

from fusion_kit.association import associate, cost_matrix, mahalanobis_cost
from fusion_kit.windows import MofN


@dataclass
class Track:
    id: int
    kf: object                     # the track's filter
    mofn: MofN
    born_t: float
    last_t: float                  # time of the last hit
    confirmed: bool = False        # passed M-of-N at least once (stays True)
    hits: int = 1
    misses: int = 0                # consecutive misses
    scans: int = 1                 # scans since birth, including the birth scan
    scans_at_last_hit: int = 1
    history: list = field(default_factory=list)   # [(t, position), ...] at each hit

    @property
    def hit_ratio(self):
        """Hits / scans between birth and last hit. People in view ~0.8-0.9; clutter lower."""
        return self.hits / self.scans_at_last_hit

    @property
    def positions(self):
        return np.array([p for _, p in self.history])


def path_length(points):
    """Total distance travelled along a sequence of positions."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def max_displacement(points):
    """Farthest any position gets from the first one. Jitter adds path length, not this."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return 0.0
    return float(np.linalg.norm(pts - pts[0], axis=1).max())


def _default_position(kf):
    """What a track records in its history: kf.position if the filter has it, else kf.x."""
    try:
        return np.array(kf.position, dtype=float)
    except AttributeError:
        return np.array(kf.x, dtype=float)


class TrackManager:
    def __init__(self, new_filter, gate=9.0, confirm=(3, 5), tentative_max_misses=2,
                 coast_time=1.5, cost=None, method="optimal", position=_default_position):
        if coast_time < 0:
            raise ValueError(f"coast_time({coast_time}) must be >= 0")
        if tentative_max_misses < 0:
            raise ValueError(f"tentative_max_misses({tentative_max_misses}) must be >= 0")
        MofN(*confirm)                       # validate m, n now
        self.new_filter = new_filter
        self.gate = gate
        self.confirm = confirm
        self.tentative_max_misses = tentative_max_misses
        self.coast_time = coast_time
        self.cost = cost
        self.method = method
        self.position = position
        self.live = []
        self.dead = []
        self.t = None
        self._next_id = 0

    def all_tracks(self):
        return self.live + self.dead

    def step(self, t, detections, R=None, birth=True, **sensor):
        """Process one scan. Return the live tracks.

        detections  list of measurements z (arrays or numbers)
        R           None, or one noise covariance per detection
        birth       start new tracks from unmatched detections
        sensor      extra keywords for the filters (H=..., sensor=...)
        """
        detections = list(detections)
        Rs = [None] * len(detections) if R is None else list(R)
        if len(Rs) != len(detections):
            raise ValueError(f"got {len(Rs)} R's for {len(detections)} detections")

        # 1. predict
        if self.t is not None:
            dt = t - self.t
            if dt < 0:
                raise ValueError(f"time went backwards: {self.t} -> {t}")
            if dt > 0:
                for tr in self.live:
                    tr.kf.predict(dt)
        self.t = t

        # 2. associate
        cost_fn = self.cost or (lambda tr, d: mahalanobis_cost(tr.kf, d[0], d[1], **sensor))
        cost = cost_matrix(self.live, list(zip(detections, Rs)), cost_fn)
        result = associate(cost, self.gate, self.method)

        # 3. update matched, coast the rest
        for tr in self.live:
            tr.scans += 1
        for ti, di in result.matches:
            tr = self.live[ti]
            tr.kf.update(detections[di], R=Rs[di], **sensor)
            tr.hits += 1
            tr.misses = 0
            tr.last_t = t
            tr.scans_at_last_hit = tr.scans
            tr.history.append((t, self.position(tr.kf)))
            if tr.mofn.update(True):
                tr.confirmed = True

        survivors = []
        unmatched = set(result.unmatched_tracks)
        for ti, tr in enumerate(self.live):
            if ti in unmatched:
                tr.misses += 1
                tr.mofn.update(False)
                expired = (t - tr.last_t > self.coast_time) if tr.confirmed \
                    else (tr.misses > self.tentative_max_misses)
                if expired:
                    self.dead.append(tr)
                    continue
            survivors.append(tr)
        self.live = survivors

        # 4. birth
        if birth:
            for di in result.unmatched_detections:
                self._birth(t, detections[di], Rs[di])
        return self.live

    def _birth(self, t, z, R):
        kf = self.new_filter(z, R)
        mofn = MofN(*self.confirm)
        tr = Track(id=self._next_id, kf=kf, mofn=mofn, born_t=t, last_t=t)
        tr.confirmed = mofn.update(True)     # 1-of-n confirms immediately
        tr.history.append((t, self.position(kf)))
        self.live.append(tr)
        self._next_id += 1


def summarize(track):
    """Handy numbers for filtering/reporting a finished track."""
    pts = track.positions
    return {
        "id": track.id,
        "confirmed": track.confirmed,
        "duration": track.last_t - track.born_t,
        "hits": track.hits,
        "hit_ratio": track.hit_ratio,
        "path_length": path_length(pts),
        "max_displacement": max_displacement(pts),
    }
