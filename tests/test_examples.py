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
