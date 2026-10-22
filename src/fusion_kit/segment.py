"""Turn raw points into detections (object-sized groups of points).

Two cases:
    segment_scan        points arrive in firing order around a sweep (2D spinning
                        lidar, scanning sonar). Walk them in order and cut wherever
                        two neighbors aren't the same object. Fast and simple.
    euclidean_clusters  points in no particular order (3D lidar, imaging radar).
                        Group any points closer than max_gap, chaining through
                        neighbors (built on fusion_kit.clusters).
For camera masks or occupancy grids, use fusion_kit.blobs instead.

The same-object rule for scanning sensors: adjacent beams spread apart with
range (gap = range * beam_step), so a fixed distance threshold splits far
objects or merges near ones. beam_gap_rule scales the threshold with range.

Quick reference:
    rule = beam_gap_rule(beam_step=math.radians(1.0), k=2.0, floor=0.15)
    segments = segment_scan(xy, rule, wrap=True)   # list of index arrays
    groups = euclidean_clusters(xyz, max_gap=0.3, min_points=3)
    centroids(points, groups)                      # one mean point per group
"""

import numpy as np
from scipy.spatial import cKDTree

from fusion_kit.clusters import cluster


def beam_gap_rule(beam_step, k=2.0, floor=0.0):
    """Same-object rule for a scanning sensor with angular beam spacing beam_step (radians).

    Two neighbors are the same object if their gap <= max(floor, k * r * beam_step),
    with r the nearer point's range from the sensor (at the origin).
    k = 2 bridges one missing return; floor stops near-range points splitting on noise.
    """
    if beam_step <= 0 or k <= 0 or floor < 0:
        raise ValueError("need beam_step > 0, k > 0, floor >= 0")

    def same_object(a, b):
        r = min(np.linalg.norm(a), np.linalg.norm(b))
        return np.linalg.norm(a - b) <= max(floor, k * r * beam_step)
    return same_object


def gap_rule(max_gap):
    """Same-object rule with a fixed distance threshold."""
    return lambda a, b: np.linalg.norm(a - b) <= max_gap


def segment_scan(points, same_object, wrap=False, min_points=1):
    """Split points (in firing order) into runs of neighbors that are the same object.

    points       (N, 2) or (N, 3) array, in the order the sensor fired
    same_object  function(point_a, point_b) -> bool for neighbors
    wrap         the scan is a full circle: join the last run to the first if they match
    min_points   drop runs with fewer points (a callable(points_of_run) -> int also works,
                 e.g. to allow single returns only at long range)
    Returns a list of index arrays into points.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return []
    runs = [[0]]
    for i in range(1, len(pts)):
        if same_object(pts[i - 1], pts[i]):
            runs[-1].append(i)
        else:
            runs.append([i])
    if wrap and len(runs) > 1 and same_object(pts[runs[-1][-1]], pts[runs[0][0]]):
        runs[0] = runs.pop() + runs[0]
    keep = []
    for run in runs:
        need = min_points(pts[run]) if callable(min_points) else min_points
        if len(run) >= need:
            keep.append(np.array(run))
    return keep


def euclidean_clusters(points, max_gap, min_points=1):
    """Group unordered points: any two closer than max_gap are in the same group (chained).

    Returns a list of index arrays, largest group first.
    """
    if max_gap <= 0:
        raise ValueError(f"max_gap({max_gap}) must be > 0")
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return []
    pairs = cKDTree(pts).query_pairs(max_gap)
    groups = [np.array(sorted(g)) for g in cluster(len(pts), pairs) if len(g) >= min_points]
    return sorted(groups, key=len, reverse=True)


def centroids(points, groups):
    """Mean point of each group: the usual detection position."""
    pts = np.asarray(points, dtype=float)
    return [pts[g].mean(axis=0) for g in groups]

