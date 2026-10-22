import runpy
from pathlib import Path

EXAMPLES = Path(__file__).parent.parent / "examples"


def test_track_1d_example():
    run = runpy.run_path(str(EXAMPLES / "track_1d.py"))["run"]
    truths = (10.0, 30.0)
    tracks = run(seed=0, truths=truths)
    for tr, truth in zip(tracks, truths):
        # 80% detection means a run of misses can briefly drop confirmation,
        # so check it holds on most scans rather than on the last one.
        settled = tr["history"][5:]
        assert sum(settled) / len(settled) >= 0.6
        assert abs(tr["kf"].x - truth) < 1.0      # settled sigma is ~0.2 m


def test_multi_target_2d_example():
    import numpy as np
    run = runpy.run_path(str(EXAMPLES / "multi_target_2d.py"))["run"]
    for seed in range(3):
        _, real, truth = run(seed=seed)
        assert len(real) == 3
        ends = np.array([tr.positions[-1] for tr in real])
        for p in truth:                              # every target has a track ending near it
            assert np.min(np.linalg.norm(ends - p, axis=1)) < 1.0
