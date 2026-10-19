"""Group 1D measurements whose uncertainty gates overlap (from the "merge intervals" puzzle).

Each track is a measurement with a +/- uncertainty, e.g. a range of
10.0 m +/- 0.5 m. Its "gate" is the interval [center - unc, center + unc].
Tracks whose gates overlap could be the same object, so they are grouped.

Grouping chains: if gate A overlaps B and B overlaps C, then A, B and C
form one group even if A and C don't touch. A wide gate can swallow
several narrow ones.

Method: sort the gates by their lower edge, then sweep left to right.
Each gate either overlaps the group being built (extend that group) or
starts a new one. Sorting dominates the cost.
"""

from typing import NamedTuple


class Group(NamedTuple):
    lo: float        # lower edge of the merged gate
    hi: float        # upper edge of the merged gate
    members: list    # indices into the input, ascending


def merge_intervals(intervals, tol=0.0):
    """Merge overlapping (lo, hi) intervals. Return a sorted list of (lo, hi) tuples.

    Intervals that only touch (one's hi == the next's lo) are merged.
    tol widens that check, so gaps up to tol also merge. The input is not modified.
    """
    return [(g.lo, g.hi) for g in _merge(list(intervals), tol)]


def gate_tracks(tracks, tol=1e-9):
    """Group (center, uncertainty) tracks whose gates overlap.

    Return a list of Group(lo, hi, members), sorted by lo.
    tol absorbs floating point rounding, so gates that touch on paper
    (e.g. 0.15+0.15 vs 0.45-0.15) still merge. Set tol=0 for an exact check.
    """
    intervals = []
    for center, unc in tracks:
        if unc < 0:
            raise ValueError(f"uncertainty must be >= 0, got {unc}")
        intervals.append((center - unc, center + unc))
    return _merge(intervals, tol)


def _merge(intervals, tol):
    """Sweep merge. Each input interval keeps its original index in members."""
    for lo, hi in intervals:
        if lo > hi:
            raise ValueError(f"interval has lo > hi: ({lo}, {hi})")
    order = sorted(range(len(intervals)), key=lambda i: intervals[i])

    groups = []
    for i in order:
        lo, hi = intervals[i]
        if groups and lo <= groups[-1].hi + tol:
            last = groups[-1]
            groups[-1] = Group(last.lo, max(last.hi, hi), last.members + [i])
        else:
            groups.append(Group(lo, hi, [i]))

    for g in groups:
        g.members.sort()
    return groups
