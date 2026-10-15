#!/usr/bin/env python3
"""Precompute the temporal stage table used by the auxiliary stage head (see docs/TRAINING_GUIDE.md, "Stage head").

The 2025 winner's "System 2 stage tracking": split every episode into N equal-time stages and have the model
predict which one it is in. N is per task, `clip(mean_episode_length / 900, 5, 15)`, and the label for a frame is
`floor(frame / (episode_length / N))`. It needs no annotation -- only episode lengths, which are in the dataset's
own metadata -- and, unlike a gripper-state label, it is NOT recoverable from the observation. That is the whole
point: "the robot holding a radio at the start of a task looks identical to holding it at the end."

Writes {"n_stages": {task_index: N}, "length": {episode_index: frames}} next to the dataset's norm stats, so the
data transform can look it up without touching dataset metadata at training time.

Usage: compute_stage_table.py --root <dataset root> --out <assets dir>/stage_table.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="training dataset root (only meta/episodes is read)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    lengths: dict[int, int] = {}
    per_task: dict[int, list[int]] = collections.defaultdict(list)
    files = sorted(pathlib.Path(a.root, "meta/episodes").rglob("*.parquet"))
    for f in files:
        t = pq.read_table(f, columns=["episode_index", "length", "task_index"]).to_pydict()
        for ep, ln, tk in zip(t["episode_index"], t["length"], t["task_index"], strict=True):
            lengths[int(ep)] = int(ln)
            per_task[int(tk)].append(int(ln))

    n_stages = {tk: max(5, min(15, round(sum(v) / len(v) / 900))) for tk, v in per_task.items()}
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_stages": {str(k): v for k, v in sorted(n_stages.items())},
                               "length": {str(k): v for k, v in sorted(lengths.items())}}))

    hist = collections.Counter(n_stages.values())
    print(f"{len(files)} metadata files -> {len(lengths):,} episodes, {len(n_stages)} tasks")
    print(f"episode length: min {min(lengths.values()):,}  mean {sum(lengths.values()) // len(lengths):,}  "
          f"max {max(lengths.values()):,} frames")
    print(f"stages per task: {dict(sorted(hist.items()))}  (N -> how many tasks)")
    print(f"written: {out}  ({out.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
