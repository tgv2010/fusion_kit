"""Sliding-window detection logic (from the "M-of-N" puzzle).

M-of-N confirmation: declare a target present when at least m of the
last n scans had a hit. A single noisy hit doesn't confirm a track, and a
single missed scan doesn't drop it. It's the digital cousin of a
debounce filter.

Method: keep the last n hits in a fixed-length queue and a running count
of how many are hits. Each new scan adds one entry and, once the window is
full, pushes the oldest one out, so the count updates in constant time
instead of re-summing the window.

Before n scans have arrived the window is just shorter: the output can
go True as soon as m hits are in, it doesn't wait for a full window.
"""

from collections import deque


class MofN:
    """Streaming M-of-N detector: call update() once per scan."""

    def __init__(self, m, n):
        if n < 1:
            raise ValueError(f"n({n}) cannot be less than 1")
        if m < 1:
            raise ValueError(f"m({m}) cannot be less than 1")
        if m > n:
            raise ValueError(f"m({m}) cannot be greater than n({n})")
        self.m = m
        self.n = n
        self._window = deque(maxlen=n)  # oldest entry drops out automatically
        self.hits = 0                   # number of hits currently in the window

    def update(self, hit):
        """Add one scan (truthy = hit). Return True if at least m of the last n were hits."""
        if len(self._window) == self.n and self._window[0]:
            self.hits -= 1              # the oldest entry is about to fall out
        hit = bool(hit)
        self._window.append(hit)
        if hit:
            self.hits += 1
        return self.hits >= self.m

    def reset(self):
        """Forget all past scans."""
        self._window.clear()
        self.hits = 0


def m_of_n(hits, m, n):
    """Run M-of-N over a whole sequence. Return one bool per scan."""
    detector = MofN(m, n)
    return [detector.update(h) for h in hits]
