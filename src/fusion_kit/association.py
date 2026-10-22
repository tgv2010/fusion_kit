"""Match detections to tracks, one-to-one.

Every scan a multi-target tracker must decide which detection belongs to
which track. Build a cost matrix (rows = tracks, columns = detections,
lower = better match), forbid pairs outside the gate, then pick pairs:

    optimal  Hungarian algorithm: the pairing with the lowest total cost.
             Matches as many pairs as possible first, then minimizes cost.
             Use this by default.
    greedy   repeatedly take the cheapest remaining pair. Simpler and fast,
             but can make a choice that forces a worse one later, e.g. when
             two targets cross.

Unmatched tracks coasted (missed this scan); unmatched detections are
candidates for new tracks.

Pick the cost to fit the sensor:
    Position sensors (lidar, radar, GPS):  squared Mahalanobis distance,
        kf.distance(z) ** 2, gate = k**2 (k=3 -> gate 9). mahalanobis_cost does this.
    Camera boxes:  1 - iou(box_a, box_b), gate e.g. 0.7 (needs >= 30% overlap).
    Anything else: any function where lower is better. Return math.inf to forbid.

Quick reference:
    cost = cost_matrix(tracks, detections, lambda trk, det: ...)
    result = associate(cost, gate=9.0)
    result.matches                 [(track_index, detection_index), ...]
    result.unmatched_tracks        [track_index, ...]
    result.unmatched_detections    [detection_index, ...]
"""

from typing import NamedTuple

import numpy as np
from scipy.optimize import linear_sum_assignment


class Association(NamedTuple):
    matches: list               # [(track_index, detection_index)], sorted by track
    unmatched_tracks: list
    unmatched_detections: list


def cost_matrix(tracks, detections, cost_fn):
    """cost[i, j] = cost_fn(tracks[i], detections[j])."""
    cost = np.empty((len(tracks), len(detections)))
    for i, trk in enumerate(tracks):
        for j, det in enumerate(detections):
            cost[i, j] = cost_fn(trk, det)
    return cost


def mahalanobis_cost(kf, z, R=None, **sensor):
    """Squared Mahalanobis distance from a filter's expected measurement to z.

    Pair with gate = k**2. Extra keywords (H=..., sensor=...) pass through to kf.distance.
    """
    return kf.distance(z, R=R, **sensor) ** 2


def iou(box_a, box_b):
    """Intersection over union of two (x1, y1, x2, y2) boxes: 0 = no overlap, 1 = identical."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def associate(cost, gate=np.inf, method="optimal"):
    """Pick one-to-one (track, detection) pairs from a cost matrix.

    Pairs with cost > gate, or a non-finite cost, are never matched.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2:
        raise ValueError(f"cost must be a 2D matrix (tracks x detections), got shape {cost.shape}")
    n_trk, n_det = cost.shape
    allowed = np.isfinite(cost) & (cost <= gate)

    if method == "optimal":
        pairs = _optimal(cost, allowed)
    elif method == "greedy":
        pairs = _greedy(cost, allowed)
    else:
        raise ValueError(f"method must be 'optimal' or 'greedy', got {method!r}")

    pairs.sort()
    used_t = {t for t, _ in pairs}
    used_d = {d for _, d in pairs}
    return Association(
        matches=pairs,
        unmatched_tracks=[t for t in range(n_trk) if t not in used_t],
        unmatched_detections=[d for d in range(n_det) if d not in used_d],
    )


def _optimal(cost, allowed):
    if not allowed.any():
        return []
    # Forbidden pairs get a cost so large that the solver only uses one when
    # it has no choice, so it maximizes valid matches first; we drop those after.
    big = 2 * np.abs(cost[allowed]).sum() + 1
    work = np.where(allowed, cost, big)
    rows, cols = linear_sum_assignment(work)
    return [(int(t), int(d)) for t, d in zip(rows, cols) if allowed[t, d]]


def _greedy(cost, allowed):
    t_idx, d_idx = np.nonzero(allowed)
    order = sorted(range(len(t_idx)), key=lambda k: (cost[t_idx[k], d_idx[k]], t_idx[k], d_idx[k]))
    used_t, used_d, pairs = set(), set(), []
    for k in order:
        t, d = int(t_idx[k]), int(d_idx[k])
        if t not in used_t and d not in used_d:
            pairs.append((t, d))
            used_t.add(t)
            used_d.add(d)
    return pairs
