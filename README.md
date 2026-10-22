# fusion_kit

Reusable, tested building blocks for sensor fusion and computer vision. Each
module comes from a solved puzzle in `../coding_exercise`.

| Module | Puzzle | What it does |
|---|---|---|
| `blobs` | number of islands | Find connected blobs in a 2D grid or mask (4- or 8-connected) |
| `clusters` | track merge | Merge "same object" detection pairs into objects (union-find) |
| `gating` | merge intervals / gating | Group (center, ±uncertainty) tracks whose gates overlap; plain interval merge |
| `windows` | M-of-N | Confirm a detection when m of the last n scans hit (batch or streaming) |
| `kalman` | 1D Kalman (+ N-D, not from a puzzle) | `Kalman1D` smooths a steady value; `ConstantVelocity` tracks a moving position in 1D/2D/3D; `KalmanFilter` takes any F, H, Q, R. `gate()` / `distance()` say whether a measurement belongs to the track |

## Usage

```python
from fusion_kit.blobs import count_blobs, find_blobs
from fusion_kit.clusters import cluster, count_clusters
from fusion_kit.gating import gate_tracks, merge_intervals
from fusion_kit.windows import MofN, m_of_n
from fusion_kit.kalman import ConstantVelocity, Kalman1D

count_blobs(mask, connectivity=8, is_foreground=lambda v: v >= 128)
cluster(6, [[4, 0], [2, 5]])   # [{0, 4}, {1}, {2, 5}, {3}]
gate_tracks([(10, 1), (2, 1), (3, 0.5)])
# [Group(lo=1, hi=3.5, members=[1, 2]), Group(lo=9, hi=11, members=[0])]
m_of_n([1, 1, 0, 1, 0, 0, 1, 1], m=3, n=5)
# [False, False, False, True, True, False, False, True]
kf = Kalman1D(z0=10.2, r=0.25, q=0.01)   # r, q are variances
kf.step(9.8)                              # -> (x, P); step(None) for a missed scan

cv = ConstantVelocity(z0=[0, 0], r=0.25, q=0.1, dt=0.1)   # 2D position track
cv.step([0.1, 0.2])                       # step(z, dt=...) if scans are irregular
cv.position, cv.velocity
cv.distance([0.3, 0.4]) <= 3              # does this measurement belong to the track?
```

## Putting it together

[examples/track_1d.py](examples/track_1d.py) tracks two targets with one noisy sensor. Each scan it:

1. **kalman**: `predict()` every track, then `gate()` gives the window the next hit should land in.
2. **gating**: `gate_tracks(gates)` flags tracks whose windows overlap, where one measurement could belong to either.
3. **kalman**: `update(z)` each track with the nearest measurement inside its window.
4. **windows**: an `MofN(3, 5)` per track turns hit/miss into confirmed / not confirmed.

```sh
uv run python examples/track_1d.py
```

Each module's docstring explains its knobs (what r, q, k, m, n, tol mean and how to pick them).

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
