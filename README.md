# fusion_kit

Reusable, tested building blocks for sensor fusion and computer vision. The
core modules come from solved puzzles in `../coding_exercise`; the rest were
added so a full tracker can be assembled off the shelf.

## I need to...

| Task | Reach for |
|---|---|
| Track several moving targets | `tracking.TrackManager` + `kalman.ConstantVelocity` ([example](examples/multi_target_2d.py)) |
| Smooth one noisy, steady value | `kalman.Kalman1D` |
| Track with radar (range/bearing/Doppler) or camera bearings | `kalman.ExtendedKalmanFilter` + `models.range_bearing` / `models.bearing_only` |
| Fuse two sensors into one track | `update(z, H=..., R=...)` (linear) or `update(z, sensor=...)` (EKF) |
| Turn a 2D lidar sweep into detections | `segment.segment_scan` + `segment.beam_gap_rule` |
| Turn a 3D point cloud into detections | `segment.euclidean_clusters` |
| Turn a camera mask into detections | `blobs.find_blobs` |
| Match detections to tracks | `association.associate` (Hungarian or greedy; Mahalanobis or IoU cost) |
| Confirm a track / debounce detections | `windows.MofN` |
| Get x/y noise from a range/angle sensor | `transforms.polar_to_cartesian_cov` |
| Draw a gate or covariance ellipse | `transforms.covariance_ellipse` |

Each module's docstring opens with what it's for, what every knob means, and a quick-reference call pattern.

## Modules

| Module | Source | What it does |
|---|---|---|
| `blobs` | puzzle: number of islands | Connected blobs in a 2D grid or mask (4- or 8-connected) |
| `clusters` | puzzle: track merge | Merge "same object" pairs into objects (union-find) |
| `gating` | puzzle: merge intervals / gating | Group (center, ±uncertainty) tracks whose gates overlap |
| `windows` | puzzle: M-of-N | Confirm when m of the last n scans hit (batch or streaming) |
| `kalman` | puzzle: 1D Kalman; rest added | `Kalman1D`, `KalmanFilter` (any F, H, Q, R), `ConstantVelocity` (1D/2D/3D), `ExtendedKalmanFilter`. Per-measurement `R`, per-sensor `H`/`sensor`, `distance()` / `gate()` |
| `models` | added | Motion and sensor models for the EKF: constant velocity, position, range/bearing(/Doppler), bearing-only |
| `association` | added | One-to-one track/detection matching with gating; Mahalanobis and IoU costs |
| `tracking` | added | `TrackManager`: predict, associate, update, birth, M-of-N confirm, coast, delete; track metrics |
| `transforms` | added | Polar/spherical to Cartesian with noise propagation, covariance ellipses, angle wrapping, numerical Jacobians |
| `segment` | added | Scan segmentation for spinning sensors, Euclidean clustering for point clouds |

## Usage

```python
from fusion_kit.kalman import ConstantVelocity, Kalman1D
from fusion_kit.tracking import TrackManager

# One target, 2D position sensor
cv = ConstantVelocity(z0=[0, 0], r=0.25, q=0.1, dt=0.1)   # r, q are variances
cv.step([0.1, 0.2])                       # predict + update; step(None) for a missed scan
cv.position, cv.velocity
cv.distance([0.3, 0.4]) <= 3              # does this measurement belong to the track?

# Many targets
tm = TrackManager(lambda z, R: ConstantVelocity(z, r=R if R is not None else 0.04,
                                                q=1.0, dt=0.1, speed_std=2.0))
for t, detections in scans:
    tm.step(t, detections)                # optional: R=[one covariance per detection]
confirmed = [tr for tr in tm.all_tracks() if tr.confirmed]

# Radar with an EKF
from fusion_kit.kalman import ExtendedKalmanFilter
from fusion_kit.models import constant_velocity, initial_state, range_bearing
from fusion_kit.transforms import polar_to_cartesian_cov, polar_to_xy

xy = polar_to_xy(r0, th0)
x0, P0 = initial_state(xy, polar_to_cartesian_cov(r0, th0, 0.1, 0.01), speed_std=5.0)
ekf = ExtendedKalmanFilter(x0, P0, constant_velocity(2, q=1.0), range_bearing(0.1, 0.01), dt=0.1)
ekf.step([r, th])
```

## Examples

Both run as tests, so they stay working.

- [examples/multi_target_2d.py](examples/multi_target_2d.py): three targets (two crossing), clutter and misses from a range/bearing sensor, using `TrackManager`, `ConstantVelocity` and per-detection polar noise.
- [examples/track_1d.py](examples/track_1d.py): the puzzle modules wired by hand: `Kalman1D` → `gate()` → `gating.gate_tracks` → update → `MofN`.

```sh
uv run python examples/multi_target_2d.py
```

## Development

```sh
uv sync          # set up the environment (numpy, scipy)
uv run pytest    # run all tests
```

## Adding a solved puzzle

1. Copy the final solution into `src/fusion_kit/<module>.py`. Remove the debug prints and add docstrings.
2. Move the puzzle's test cases into `tests/test_<module>.py` and add edge cases
   (empty input, bad input, large input).
3. Run `uv run pytest`, then add the module to the tables above.
