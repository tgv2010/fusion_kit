import math

import numpy as np
import pytest

from fusion_kit.association import iou
from fusion_kit.kalman import ConstantVelocity
from fusion_kit.tracking import TrackManager, max_displacement, path_length, summarize

DT = 0.1


def cv(z, R):
    return ConstantVelocity(z, r=0.01 if R is None else R, q=1.0, dt=DT, speed_std=2.0)


def run(scans, **kw):
    tm = TrackManager(cv, **kw)
    for k, dets in enumerate(scans):
        tm.step(k * DT, dets)
    return tm


def test_single_target_confirms_after_m_of_n():
    tm = TrackManager(cv)
    states = []
    for k in range(5):
        tm.step(k * DT, [[k * 0.1, 0.0]])
        states.append(tm.live[0].confirmed)
    assert states == [False, False, True, True, True]
    assert len(tm.live) == 1 and tm.live[0].hits == 5


def test_crossing_targets_keep_identity():
    # A walks +x, B walks -x, 0.4 m apart in y; they pass each other mid-run.
    rng = np.random.default_rng(0)
    scans = []
    for k in range(60):
        t = k * DT
        a = [-3 + 1.0 * t, 0.2]
        b = [3 - 1.0 * t, -0.2]
        scans.append([np.add(p, rng.normal(0, 0.03, 2)) for p in (a, b)])
    tm = run(scans)
    assert len(tm.all_tracks()) == 2
    first = {tr.id: tr.positions[0] for tr in tm.live}
    last = {tr.id: tr.positions[-1] for tr in tm.live}
    for tid in first:
        assert np.sign(last[tid][1]) == np.sign(first[tid][1])      # stayed in its lane
        assert abs(last[tid][0] - first[tid][0]) > 5                 # crossed over


def test_clutter_never_confirms():
    rng = np.random.default_rng(1)
    scans = [[rng.uniform(-10, 10, 2) for _ in range(3)] for _ in range(50)]
    tm = run(scans)
    assert not any(tr.confirmed for tr in tm.all_tracks())
    assert len(tm.live) <= 3 * 3               # tentative tracks die within 3 scans


def test_confirmed_track_coasts_then_dies():
    tm = TrackManager(cv, coast_time=0.5)
    t = 0.0
    for k in range(10):
        t = k * DT
        tm.step(t, [[0.5 * t, 0.0]])
    tr = tm.live[0]
    for k in range(1, 6):                        # 0.5 s of misses: still alive
        tm.step(t + k * DT, [])
    assert tm.live == [tr]
    assert tr.misses == 5
    tm.step(t + 0.6, [])                         # past coast_time
    assert tm.live == [] and tm.dead == [tr]
    tm2 = TrackManager(cv, coast_time=0.5)
    for k in range(10):
        tm2.step(k * DT, [[0.5 * k * DT, 0.0]])
    for k in range(1, 4):                        # gap, then reacquired
        tm2.step(0.9 + k * DT, [])
    tm2.step(0.9 + 4 * DT, [[0.5 * 1.3, 0.0]])
    assert len(tm2.all_tracks()) == 1 and tm2.live[0].misses == 0


def test_tentative_dies_after_misses():
    tm = TrackManager(cv, tentative_max_misses=2)
    tm.step(0.0, [[0.0, 0.0]])
    tm.step(0.1, [])
    tm.step(0.2, [])
    assert len(tm.live) == 1
    tm.step(0.3, [])
    assert tm.live == [] and len(tm.dead) == 1


def test_confirm_1_of_1_is_immediate():
    tm = TrackManager(cv, confirm=(1, 1))
    tm.step(0.0, [[0.0, 0.0]])
    assert tm.live[0].confirmed


def test_per_detection_r_reaches_filter():
    seen = []

    class Spy:
        position = np.zeros(2)

        def predict(self, dt):
            pass

        def distance(self, z, R=None):
            return 0.0

        def update(self, z, R=None):
            seen.append(R)

    tm = TrackManager(lambda z, R: Spy())
    tm.step(0.0, [[0, 0]], R=["birth"])
    tm.step(0.1, [[0, 0]], R=["first update"])
    assert seen == ["first update"]
    with pytest.raises(ValueError):
        tm.step(0.2, [[0, 0]], R=[])


def test_extra_sensor_keywords_and_birth_false():
    tm = TrackManager(cv)
    for k in range(3):
        tm.step(k * DT, [[0.0, 0.0]])
    tm.step(0.3, [[0.0], [50.0]], birth=False, H=[[1, 0, 0, 0]], R=[[[0.01]], [[0.01]]])
    assert len(tm.live) == 1                     # the far x-only detection didn't start a track
    assert tm.live[0].hits == 4


def test_custom_cost_iou():
    # Track boxes: filter state is box center; cost compares boxes of fixed size.
    size = 1.0

    def box(c):
        return (c[0] - size / 2, c[1] - size / 2, c[0] + size / 2, c[1] + size / 2)

    def cost(tr, det):
        z, _ = det
        return 1 - iou(box(tr.kf.position), box(z))

    tm = TrackManager(cv, cost=cost, gate=0.7)
    for k in range(10):
        tm.step(k * DT, [[0.05 * k, 0.0], [5.0, 5.0 + 0.05 * k]])
    assert len(tm.all_tracks()) == 2 and all(tr.confirmed for tr in tm.live)


def test_time_handling():
    tm = TrackManager(cv)
    tm.step(1.0, [[0.0, 0.0]])
    p_before = tm.live[0].kf.P.copy()
    tm.step(1.0, [])                             # same timestamp: no predict
    assert tm.live[0].kf.P == pytest.approx(p_before)
    with pytest.raises(ValueError):
        tm.step(0.5, [])


def test_bad_settings():
    with pytest.raises(ValueError):
        TrackManager(cv, confirm=(4, 3))
    with pytest.raises(ValueError):
        TrackManager(cv, coast_time=-1)


def test_metrics():
    pts = [[0, 0], [3, 4], [0, 0]]
    assert path_length(pts) == 10
    assert max_displacement(pts) == 5
    assert path_length([[1, 1]]) == 0 and max_displacement([]) == 0
    tm = run([[[0.1 * k, 0.0]] if k % 2 == 0 else [] for k in range(9)],
             tentative_max_misses=5)
    s = summarize(tm.live[0])
    assert s["hits"] == 5 and s["hit_ratio"] == pytest.approx(5 / 9)
    assert s["max_displacement"] == pytest.approx(0.8, abs=0.1)
