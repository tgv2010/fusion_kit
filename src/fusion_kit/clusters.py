"""Merge detections into objects (from the "track merge" puzzle).

You have n detections labeled 0..n-1 and a list of pairs, each meaning
"these two detections are the same object". Matches chain: if A~B and B~C,
then A, B and C are one object.

Method (union-find): every detection points to a "parent". Following the
parents up from any detection ends at a single root, and that root names
the cluster. Merging two clusters means pointing one root at the other.
This works like the puzzle's relabeling solution, but a merge no longer
relabels every member, so it stays fast with thousands of detections.
"""


class Clusters:
    """Incrementally merge items 0..n-1 into clusters."""

    def __init__(self, n):
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        self._parent = list(range(n))  # each item starts as its own cluster
        self.count = n                 # number of distinct clusters

    def _check(self, item):
        if not 0 <= item < len(self._parent):
            raise IndexError(f"item {item} out of range 0..{len(self._parent) - 1}")

    def find(self, item):
        """Return the root (cluster id) of item."""
        self._check(item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        # Path shortcut: point everything we passed directly at the root,
        # so the next lookup is one step.
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a, b):
        """Mark a and b as the same object. Return True if two clusters merged."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self._parent[rb] = ra
        self.count -= 1
        return True

    def groups(self):
        """Return the clusters as a list of sets, sorted by smallest member."""
        by_root = {}
        for item in range(len(self._parent)):
            by_root.setdefault(self.find(item), set()).add(item)
        return sorted(by_root.values(), key=min)


def cluster(n, pairs):
    """Return the clusters formed by n items and matching pairs, as a list of sets."""
    c = Clusters(n)
    for a, b in pairs:
        c.union(a, b)
    return c.groups()


def count_clusters(n, pairs):
    """Return the number of distinct objects among n items given matching pairs."""
    c = Clusters(n)
    for a, b in pairs:
        c.union(a, b)
    return c.count
