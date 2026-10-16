#!/usr/bin/env python3
"""Verify the 224 training root before training on it (cheap: reads container headers, decodes nothing).

    python3 scripts/verify_rgb224.py --src $WORK/data/2026-challenge-demos --dst $WORK/data/2026-challenge-rgb224

Checks every RGB video: it exists, has EXACTLY the source's frame count (a short file silently shifts frame indices),
is 224x224, and has no B-frames (has_b_frames=0 -- otherwise the loader's fast seek is wrong and training crawls).
Exit code 0 only if every file passes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RGB_STREAMS = ["observation.rgb.zed_link_camera_0",
               "observation.rgb.left_realsense_link_camera_0",
               "observation.rgb.right_realsense_link_camera_0"]


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_frames,width,height,has_b_frames,codec_name", "-of", "json", str(path)],
                         capture_output=True, text=True).stdout
    s = (json.loads(out or "{}").get("streams") or [{}])[0]
    return {"frames": int(s.get("nb_frames", 0) or 0), "w": s.get("width"), "h": s.get("height"),
            "bframes": int(s.get("has_b_frames", -1)), "codec": s.get("codec_name")}


def check(pair: tuple[Path, Path], size: int) -> str | None:
    src, dst = pair
    if not dst.exists():
        return f"missing: {dst}"
    a, b = probe(src), probe(dst)
    if a["frames"] != b["frames"]:
        return f"frame count {b['frames']} != source {a['frames']}: {dst}"
    if (b["w"], b["h"]) != (size, size):
        return f"size {b['w']}x{b['h']}: {dst}"
    if b["bframes"] != 0:
        return f"has B-frames ({b['bframes']}): {dst}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    pairs = [(f, dst / f.relative_to(src)) for s in RGB_STREAMS for f in sorted((src / "videos" / s).rglob("*.mp4"))]
    with ThreadPoolExecutor(a.threads) as ex:
        problems = [p for p in ex.map(lambda pr: check(pr, a.size), pairs) if p]
    for p in problems[:50]:
        print("FAIL", p)
    info = json.loads((dst / "meta/info.json").read_text())
    depth_left = [k for k in info["features"] if "depth" in k]
    print(f"{len(pairs)} RGB files checked, {len(problems)} problems; depth features left in info.json: {depth_left}")
    return 1 if problems or depth_left else 0


if __name__ == "__main__":
    sys.exit(main())
