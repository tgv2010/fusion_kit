"""End-to-end example: several targets, clutter and missed detections, one 2D sensor.

The sensor reports range and bearing (think radar): precise in range, poor
in angle. Each scan:
    transforms   turn each (range, bearing) into x/y plus its noise ellipse R
    tracking     TrackManager predicts, associates (Hungarian, 3-sigma gate),
                 updates, starts tentative tracks, confirms with 3-of-5, deletes
    kalman       one ConstantVelocity filter per track, fed each detection's own R

Run:  uv run python examples/multi_target_2d.py
"""

import math

import numpy as np

from fusion_kit.kalman import ConstantVelocity
from fusion_kit.tracking import TrackManager, summarize
from fusion_kit.transforms import polar_to_cartesian_cov, polar_to_xy

DT = 0.1           # 10 Hz scans
SIGMA_R = 0.05     # range noise, m
SIGMA_TH = 0.02    # bearing noise, rad (0.4 m sideways at 20 m)
P_DETECT = 0.9
CLUTTER_PER_SCAN = 2


def simulate(rng, scans=150):
    """Three targets: two walking past each other, one driving across. Yields (t, detections)."""
    starts = np.array([[5.0, 12.0], [15.0, 12.6], [-10.0, 25.0]])
    vels = np.array([[1.2, 0.0], [-1.2, 0.0], [4.0, -1.0]])
    for k in range(scans):
        t = k * DT
        truth = starts + vels * t
        dets = []
        for p in truth:
            if rng.random() < P_DETECT:
                r, th = math.hypot(*p), math.atan2(p[1], p[0])
                dets.append((r + rng.normal(0, SIGMA_R), th + rng.normal(0, SIGMA_TH)))
        for _ in range(rng.poisson(CLUTTER_PER_SCAN)):
            dets.append((rng.uniform(2, 40), rng.uniform(0, math.pi)))
        yield t, dets, truth


def run(seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    tm = TrackManager(
        lambda z, R: ConstantVelocity(z, r=R, q=1.0, dt=DT, speed_std=5.0),
        gate=9.0, confirm=(3, 5), coast_time=1.0)
    for t, polar_dets, truth in simulate(rng):
        zs = [np.array(polar_to_xy(r, th)) for r, th in polar_dets]
        Rs = [polar_to_cartesian_cov(r, th, SIGMA_R, SIGMA_TH) for r, th in polar_dets]
        tm.step(t, zs, R=Rs)
    real = [tr for tr in tm.all_tracks() if tr.confirmed and summarize(tr)["max_displacement"] > 3]
    if verbose:
        print(f"tracks started: {len(tm.all_tracks())}, confirmed and moving: {len(real)}")
        for tr in real:
            s = summarize(tr)
            print(f"  id {s['id']:3d}: {s['duration']:4.1f} s, hit ratio {s['hit_ratio']:.2f}, "
                  f"ends at {np.round(tr.positions[-1], 1)}, velocity {np.round(tr.kf.velocity, 1)}")
        print(f"truth at end: {np.round(truth, 1).tolist()}")
    return tm, real, truth


if __name__ == "__main__":
    run(verbose=True)
