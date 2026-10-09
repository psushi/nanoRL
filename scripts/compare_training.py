"""Time and iterations to reach lift-success thresholds, from TensorBoard logs.

    python -m scripts.compare_training artifacts/ppo-compare
"""

import argparse
import glob
import statistics
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

THRESHOLDS = (0.5, 0.7, 0.85, 0.9)


def load(run):
    ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
    ea.Reload()
    success = {e.step: (e.value, e.wall_time) for e in ea.Scalars("Metrics/lift_height/episode_success")}
    fps = [e.value for e in ea.Scalars("Perf/total_fps")]
    start = min(w for _, w in success.values())
    hits = {}
    for threshold in THRESHOLDS:
        step = next((s for s in sorted(success) if success[s][0] >= threshold), None)
        hits[threshold] = None if step is None else (step, (success[step][1] - start) / 60)
    return hits, statistics.median(fps), max(v for v, _ in success.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    groups = {}
    for run in sorted(glob.glob(str(args.root / "*" / "*"))):
        name = Path(run).name.split("_", 2)[-1]
        groups.setdefault(name.rsplit("-s", 1)[0], []).append((name, *load(run)))
    for group, runs in groups.items():
        print(f"== {group}")
        for name, hits, fps, best in runs:
            cells = [f"{t:.2f}: " + (f"it {h[0]:4d} / {h[1]:4.1f} min" if h else "not reached")
                     for t, h in hits.items()]
            print(f"  {name:14s} median SPS {fps:9,.0f} | best {best:.3f} | " + " | ".join(cells))
        for t in THRESHOLDS:
            minutes = [h[t][1] for _, h, _, _ in runs if h[t]]
            iters = [h[t][0] for _, h, _, _ in runs if h[t]]
            if minutes:
                print(f"  median to {t:.2f}: {statistics.median(minutes):4.1f} min, "
                      f"{statistics.median(iters):.0f} iterations ({len(minutes)}/{len(runs)} runs reached)")


if __name__ == "__main__":
    main()
