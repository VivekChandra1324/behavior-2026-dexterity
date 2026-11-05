#!/usr/bin/env python3
"""Derive the gripper-closure table for all 100 tasks -- the data the inference-time gripper re-open rule reads.

THE RULE (from the 2025 winning solution; implemented in openpi/overlay/src/openpi/shared/b1k_correction_rules.py):
if a gripper is closed at a (task, stage) where the demonstrations never close it, the grasp has failed -> overwrite
the predicted chunk with hold-position and command that gripper fully open.

The 2025 winner hand-built the gating tables for 50 tasks. We derive them mechanically from the demonstrations, for
all 100 tasks.

Definitions:
  gripper width  = state[24]+state[25] (left), state[49]+state[50] (right); each finger in [0, 0.05] -> width [0, 0.1]
  normalized     = width*20 - 1   (so -1 = fully closed, +1 = fully open)
  CLOSED         = normalized < -0.98  <=>  width < 0.001
  stage          = clamp(int(frame_index / (episode_length / num_stages)), 0, num_stages-1)
  num_stages     = clip(ceil(mean_episode_length / 900), 5, 15) per task -- the 2025 winner's stage count (this ceil
                   form reproduces their 50-task table exactly). Note: compute_stage_table.py, which labels the
                   training data for the stage head, uses round() instead, so 33 tasks get one stage fewer there.
                   The table published with our model was built with ceil, as here. As served, the rule gets no
                   stage signal and uses only the task-level test ("never closed in this task"), which does not
                   depend on the stage count.

Reads only meta/ and the low-dimensional parquet files under data/ (no video).

Usage: derive_gripper_table.py --root <dataset root> --out gripper_table.json   (~3 min on 4 cores)
"""
import argparse, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

CLOSED_THRESHOLD = -0.98
L0, L1, R0, R1 = 24, 25, 49, 50


def episode_lengths(root: Path):
    out, per_task = {}, {}
    for f in sorted(glob.glob(str(root / "meta/episodes/**/*.parquet"), recursive=True)):
        t = pq.read_table(f, columns=["episode_index", "length", "task_index"]).to_pydict()
        for e, l, k in zip(t["episode_index"], t["length"], t["task_index"]):
            out[int(e)] = int(l)
            per_task.setdefault(int(k), []).append(int(l))
    return out, per_task


def process_file(args):
    path, ep_len, stages_by_idx = args
    t = pq.read_table(path, columns=["observation.state", "frame_index", "episode_index", "task_index"])
    st = np.asarray(t["observation.state"].combine_chunks().flatten()).reshape(len(t), -1)
    fi = np.asarray(t["frame_index"]); ei = np.asarray(t["episode_index"]); ti = np.asarray(t["task_index"])
    lw = (st[:, L0] + st[:, L1]) * 20.0 - 1.0
    rw = (st[:, R0] + st[:, R1]) * 20.0 - 1.0
    lens = np.array([ep_len.get(int(e), 1) for e in ei], dtype=np.float64)
    nst = np.array([stages_by_idx[int(x)] for x in ti], dtype=np.float64)
    stage = np.clip((fi / np.maximum(lens / nst, 1e-9)).astype(np.int64), 0, (nst - 1).astype(np.int64))
    acc = {}
    for task in np.unique(ti):
        m = ti == task
        for s in np.unique(stage[m]):
            k = (int(task), int(s))
            sel = m & (stage == s)
            acc[k] = (int(sel.sum()),
                      int((lw[sel] < CLOSED_THRESHOLD).sum()), int((rw[sel] < CLOSED_THRESHOLD).sum()),
                      float(lw[sel].min()), float(rw[sel].min()))
    return acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root (meta/ and the low-dim data/ parquet files are read)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    root = Path(a.root)

    tasks = {json.loads(l)["task_index"]: json.loads(l)["task_name"] for l in open(root / "meta/tasks.jsonl")}
    ep_len, per_task_len = episode_lengths(root)
    stages_by_idx = {i: int(np.clip(np.ceil(np.mean(per_task_len[i]) / 900.0), 5, 15)) for i in tasks}
    files = sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True))
    print(f"{len(files)} parquet files, {len(ep_len)} episodes, {len(tasks)} tasks", flush=True)

    total = {}
    workers = max(1, min(a.workers, (os.cpu_count() or 4)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, acc in enumerate(ex.map(process_file, [(f, ep_len, stages_by_idx) for f in files], chunksize=1), 1):
            for k, v in acc.items():
                if k in total:
                    t = total[k]
                    total[k] = (t[0]+v[0], t[1]+v[1], t[2]+v[2], min(t[3], v[3]), min(t[4], v[4]))
                else:
                    total[k] = v
            if n % 25 == 0 or n == len(files):
                print(f"  {n}/{len(files)} files", flush=True)

    # ---- fold into the rule's format, for 100 tasks
    per_task = {}
    for (task, stage), (frames, lc, rc, lmin, rmin) in sorted(total.items()):
        d = per_task.setdefault(task, {"stages": {}})
        d["stages"][stage] = {"frames": frames, "left_closed": lc, "right_closed": rc,
                              "left_min": round(lmin, 4), "right_min": round(rmin, 4)}
    always_open_left, always_open_right, min_stage, always_allowed = [], [], {}, {"left": [], "right": []}
    for task, d in per_task.items():
        ls = sorted(int(s) for s, v in d["stages"].items() if v["left_closed"] > 0)
        rs = sorted(int(s) for s, v in d["stages"].items() if v["right_closed"] > 0)
        if not ls:
            always_open_left.append(task)
        elif ls[0] > 0:
            min_stage.setdefault(task, {})["left"] = ls[0]
        else:
            always_allowed["left"].append(task)
        if not rs:
            always_open_right.append(task)
        elif rs[0] > 0:
            min_stage.setdefault(task, {})["right"] = rs[0]
        else:
            always_allowed["right"].append(task)

    out = {"closed_threshold_normalized": CLOSED_THRESHOLD,
           "closed_threshold_raw_width": round(0.1 * (1 + CLOSED_THRESHOLD) / 2, 6),
           "task_names": {str(i): n for i, n in tasks.items()},
           "always_open_left_tasks": sorted(always_open_left),
           "always_open_right_tasks": sorted(always_open_right),
           "min_stage_for_closure": {str(k): v for k, v in sorted(min_stage.items())},
           "always_allowed": {k: sorted(v) for k, v in always_allowed.items()},
           "per_task": {str(k): v for k, v in sorted(per_task.items())}}
    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out, indent=1))
    print(f"\nalways-open LEFT: {len(always_open_left)}/100 tasks · always-open RIGHT: {len(always_open_right)}/100")
    print(f"stage-restricted: {len(min_stage)} tasks · closure from stage 0: "
          f"left {len(always_allowed['left'])}, right {len(always_allowed['right'])}")
    print(f"wrote {p}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
