#!/usr/bin/env python3
"""Create the training dataset: the three RGB camera videos re-encoded at 224x224, WITHOUT B-frames, depth dropped.

    python3 scripts/transcode_rgb224.py --src $WORK/data/2026-challenge-demos --dst $WORK/data/2026-challenge-rgb224 --workers 32

Why 224: pi0.5 consumes 224x224 images (openpi resizes every frame to that before augmentation), so storing them at
224 makes that resize a no-op and cuts the bytes the loader reads per sample by ~10x (the 2025 winning team did the
same). The source frames are square (720x720 head, 480x480 wrists), so this is a pure scale with no padding.

Why NO B-frames (the single most important setting here): with B-frames the stored frame order differs from the
display order, and the video decoder must then use its "exact" seek, which scans the file's whole packet index on
every open. We measured 0.82 s per file open with B-frames vs 0.003 s without -- about 2.5 s per training sample once
three cameras are opened -- and the loader fell to ~37 samples/s against the ~88 the GPUs consumed. Without B-frames
the cheap seek is correct and the loader keeps up.

What it writes: <dst>/videos/<rgb camera>/chunk-XXX/file-YYY.mp4 (HEVC, yuv420p, CRF 28, GOP 8, no B-frames);
<dst>/data and <dst>/meta/{episodes,tasks.*,stats.json} as symlinks to the source (low-dim data is unchanged); and a
rewritten <dst>/meta/info.json describing 224x224 RGB-only video features. The source is never modified.
Resumable: files already written are skipped; each file is written to .tmp and renamed only when complete.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

RGB_STREAMS = ["observation.rgb.zed_link_camera_0",
               "observation.rgb.left_realsense_link_camera_0",
               "observation.rgb.right_realsense_link_camera_0"]


def frame_count(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else 0


def transcode_one(job: tuple[str, str, int, int, int, str, int]) -> tuple[str, bool, float, str]:
    src, dst, size, crf, gop, preset, pools = job
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".mp4.tmp")
    t0 = time.time()
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-threads", "2", "-i", str(src),                   # -threads BEFORE -i caps the decoder's threads
           "-vf", f"scale={size}:{size}:flags=area",           # area filter: the right one for downscaling
           "-c:v", "libx265", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p",
           # pools: x265 otherwise sizes its thread pool to the whole machine (many workers -> massive oversubscription)
           # bframes=0: see the module docstring -- this is what keeps the training loader fast
           "-x265-params", f"log-level=none:keyint={gop}:min-keyint={gop}:scenecut=0:pools={pools}:bframes=0",
           "-threads", "2", "-f", "mp4", str(tmp)]               # -f mp4: the .tmp suffix would hide the format
    # timeout scales with the file: wrist-camera files hold ~4x the frames of head-camera files
    timeout = max(1800, 3 * frame_count(src) / 100)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, "OMP_NUM_THREADS": "1"})
        if r.returncode != 0:
            tmp.unlink(missing_ok=True)
            return str(dst), False, time.time() - t0, r.stderr.strip()[-300:]
        tmp.rename(dst)
        return str(dst), True, time.time() - t0, ""
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return str(dst), False, time.time() - t0, f"timeout after {timeout:.0f} s"


def build_root(src_root: Path, dst_root: Path, size: int) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    if not (dst_root / "data").exists():
        (dst_root / "data").symlink_to(src_root.resolve() / "data")
    meta = dst_root / "meta"
    meta.mkdir(exist_ok=True)
    for name in ("episodes", "tasks.jsonl", "tasks.parquet", "stats.json"):
        if (src_root / "meta" / name).exists() and not (meta / name).exists():
            (meta / name).symlink_to(src_root.resolve() / "meta" / name)
    info = json.loads((src_root / "meta/info.json").read_text())
    feats = info["features"]
    for key in list(feats):
        if feats[key].get("dtype") == "video":
            if key in RGB_STREAMS:
                feats[key]["shape"] = [size, size, 3]
                feats[key].setdefault("info", {}).update({"video.height": size, "video.width": size,
                                                          "video.codec": "hevc", "video.pix_fmt": "yuv420p"})
            else:
                del feats[key]                                 # depth streams are not part of the training root
    (meta / "info.json").write_text(json.dumps(info, indent=4))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="the downloaded 2026-challenge-demos root")
    ap.add_argument("--dst", required=True, help="the new training root, e.g. .../2026-challenge-rgb224")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--crf", type=int, default=28, help="28: at 224 px, CRF 20 costs 2.2x the bytes for 0.8/255 less error")
    ap.add_argument("--gop", type=int, default=8, help="keyframe interval; 8 matches the source's random-access cost")
    ap.add_argument("--preset", default="superfast", help="superfast measured 1.5x faster and 12%% smaller than fast")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--pools", type=int, default=2, help="x265 threads per worker; keep workers*pools <= CPU cores")
    ap.add_argument("--cameras", default="all", help="comma-separated subset, e.g. zed_link,left_realsense_link")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    build_root(src, dst, a.size)
    streams = RGB_STREAMS if a.cameras == "all" else [s for s in RGB_STREAMS
                                                       if any(c in s for c in a.cameras.split(","))]
    jobs = []
    for stream in streams:
        for f in sorted((src / "videos" / stream).rglob("*.mp4")):
            out = dst / f.relative_to(src)
            if not out.exists():
                jobs.append((str(f), str(out), a.size, a.crf, a.gop, a.preset, a.pools))
    print(f"{len(jobs)} files to transcode with {a.workers} workers (already done files are skipped)", flush=True)
    ok = failed = 0
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (out, success, secs, err) in enumerate(pool.imap_unordered(transcode_one, jobs), 1):
            if success:
                ok += 1
            else:
                failed += 1
                print(f"FAILED {out}: {err}", flush=True)
            if i % 50 == 0 or i == len(jobs):
                rate = i / max(time.time() - t0, 1e-9) * 3600
                print(f"  {i}/{len(jobs)} files, {failed} failed, {rate:.0f} files/h", flush=True)
    print(f"done: {ok} written, {failed} failed. Re-run the same command to retry failures.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
