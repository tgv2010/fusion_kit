"""End-to-end example: track two targets in range with one noisy sensor.

Shows how the modules fit together each scan:
    kalman   predict each track, then gate() says where its next hit should land
    gating   flag tracks whose gates overlap (a measurement could belong to either)
    kalman   update each track with the measurement inside its gate
    windows  M-of-N on hit/miss decides when a track counts as confirmed

Run:  uv run python examples/track_1d.py
"""

import random

from fusion_kit.gating import gate_tracks
from fusion_kit.kalman import Kalman1D
from fusion_kit.windows import MofN

R = 0.25           # sensor noise variance (sigma = 0.5 m)
Q = 0.01           # targets drift a little per scan
P_DETECT = 0.8     # sensor sees each target on 80% of scans


def run(seed=0, scans=30, truths=(10.0, 30.0), verbose=False):
    rng = random.Random(seed)

    # One filter + one confirmer per track, started from a first look at each target.
    tracks = [{"kf": Kalman1D(rng.gauss(t, R ** 0.5), R, Q), "confirm": MofN(3, 5), "history": []}
              for t in truths]

    for scan in range(scans):
        measurements = [rng.gauss(t, R ** 0.5) for t in truths if rng.random() < P_DETECT]

        gates = []
        for tr in tracks:
            tr["kf"].predict()
            gates.append(tr["kf"].gate(k=3))

        ambiguous = [g.members for g in gate_tracks(gates) if len(g.members) > 1]

        for tr, (center, half) in zip(tracks, gates):
            inside = [z for z in measurements if abs(z - center) <= half]
            if inside:
                tr["kf"].update(min(inside, key=lambda z: abs(z - center)))
            tr["confirmed"] = tr["confirm"].update(bool(inside))
            tr["history"].append(tr["confirmed"])

        if verbose:
            states = "  ".join(
                f"x={tr['kf'].x:6.2f} P={tr['kf'].P:.3f} {'CONF' if tr['confirmed'] else '....'}"
                for tr in tracks)
            flag = f"  overlap {ambiguous}" if ambiguous else ""
            print(f"scan {scan:2d}: {states}{flag}")

    return tracks


if __name__ == "__main__":
    run(verbose=True)
