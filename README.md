# fusion_kit

Reusable, tested building blocks for sensor fusion and computer vision. Each
module comes from a solved puzzle in `../coding_exercise`.

| Module | Puzzle | What it does |
|---|---|---|
| `blobs` | number of islands | Find connected blobs in a 2D grid or mask (4- or 8-connected) |
| `clusters` | track merge | Merge "same object" detection pairs into objects (union-find) |
| `gating` | merge intervals / gating | Group (center, ±uncertainty) tracks whose gates overlap; plain interval merge |
| `windows` | M-of-N | Confirm a detection when m of the last n scans hit (batch or streaming) |
| `kalman` | — | *planned* |

## Usage

```python
from fusion_kit.blobs import count_blobs, find_blobs
from fusion_kit.clusters import cluster, count_clusters
from fusion_kit.gating import gate_tracks, merge_intervals
from fusion_kit.windows import MofN, m_of_n

count_blobs(mask, connectivity=8, is_foreground=lambda v: v >= 128)
cluster(6, [[4, 0], [2, 5]])   # [{0, 4}, {1}, {2, 5}, {3}]
gate_tracks([(10, 1), (2, 1), (3, 0.5)])
# [Group(lo=1, hi=3.5, members=[1, 2]), Group(lo=9, hi=11, members=[0])]
m_of_n([1, 1, 0, 1, 0, 0, 1, 1], m=3, n=5)
# [False, False, False, True, True, False, False, True]
```

## Development

```sh
uv sync          # set up the environment
uv run pytest    # run all tests
```

## Adding a solved puzzle

1. Copy the final solution into `src/fusion_kit/<module>.py`. Remove the debug prints and add docstrings.
2. Move the puzzle's test cases into `tests/test_<module>.py` and add edge cases
   (empty input, bad input, large input).
3. Run `uv run pytest`, then add the module to the table above.
