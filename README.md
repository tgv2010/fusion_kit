# fusion_kit

Reusable, tested building blocks for sensor fusion and computer vision. Each
module comes from a solved puzzle in `../coding_exercise`.

| Module | Puzzle | What it does |
|---|---|---|
| `blobs` | number of islands | Find connected blobs in a 2D grid or mask (4- or 8-connected) |
| `clusters` | track merge | Merge "same object" detection pairs into objects (union-find) |
| `gating` | merge intervals | *planned* |
| `windows` | longest substring | *planned* |
| `kalman` | — | *planned* |

## Usage

```python
from fusion_kit.blobs import count_blobs, find_blobs
from fusion_kit.clusters import cluster, count_clusters

count_blobs(mask, connectivity=8, is_foreground=lambda v: v >= 128)
cluster(6, [[4, 0], [2, 5]])   # [{0, 4}, {1}, {2, 5}, {3}]
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
