"""Find connected blobs in a 2D grid (from the "number of islands" puzzle).

A blob is a group of foreground cells that touch each other. In a camera
mask, an occupancy grid, or a thresholded radar map, each blob is usually
one candidate object.

Method (flood fill): put every foreground cell in a set, then repeatedly
pick one, and remove everything reachable from it. Each pick is one blob.
"""

# Neighbor offsets (row, col).
_NEIGHBORS = {
    4: [(0, -1), (0, 1), (-1, 0), (1, 0)],                    # left, right, up, down
    8: [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)],
}


def _default_is_foreground(value):
    """Treat "1", 1 and True as foreground; everything else is background."""
    return value == "1" or value == 1


def find_blobs(grid, connectivity=4, is_foreground=_default_is_foreground):
    """Return a list of blobs, each a set of (row, col) cells.

    grid           -- list of rows (rows may have different lengths).
    connectivity   -- 4 (edges only) or 8 (edges and corners).
    is_foreground  -- function deciding whether a cell value counts as foreground.

    Everything outside the grid counts as background.
    """
    if connectivity not in _NEIGHBORS:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")
    offsets = _NEIGHBORS[connectivity]

    # Step 1: collect every foreground cell.
    unvisited = {
        (r, c)
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
        if is_foreground(value)
    }

    # Step 2: flood fill until nothing is left. Each leftover cell starts a new blob.
    blobs = []
    while unvisited:
        start = unvisited.pop()
        blob = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in offsets:
                nbr = (r + dr, c + dc)
                if nbr in unvisited:      # off-grid coords are never in the set
                    unvisited.remove(nbr)  # mark visited by removing it
                    blob.add(nbr)
                    stack.append(nbr)
        blobs.append(blob)
    return blobs


def count_blobs(grid, connectivity=4, is_foreground=_default_is_foreground):
    """Return the number of blobs in the grid. See find_blobs."""
    return len(find_blobs(grid, connectivity, is_foreground))
