#!/usr/bin/env python3
"""Compute the pi0.5 normalisation statistics (state + actions) for a B1K training root.

    cd <openpi checkout> && uv run python <this repo>/scripts/compute_norm_stats.py \
        --dataset-root $WORK/data/2026-challenge-rgb224 --repo-id behavior-1k/2026-challenge-rgb224 \
        --assets-base-dir ./assets --no-video

The fork's own scripts/compute_norm_stats.py only takes a config name (its pi05_b1k config hard-codes a dataset
path), so this wrapper loads that config, swaps in your dataset root / repo_id / assets dir, and reuses the fork's own
loader and statistics code. Training must then use the same repo_id and --assets_base_dir, so that it finds
<assets_base_dir>/pi05_b1k/<repo_id>/norm_stats.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

import numpy as np
import tqdm

import openpi.shared.normalize as normalize
import openpi.training.config as _config



# ---------------------------------------------------------------------------------------------------------------
# --no-video: compute the statistics WITHOUT decoding any video.
#
# Why: these statistics use only `state` and `actions` (84 floats = 336 bytes per sample), but a LeRobot sample
# drags 15.7 MB of decoded float32 video (3 RGB + 3 depth) with it. That waste ran our first attempts out of memory:
# per-worker decoder state grew ~17 GB/min without plateauing.
#
# How: lerobot's dataset_reader skips the whole decode path when `meta.video_keys` is empty
# (`if len(self._meta.video_keys) > 0:`). `video_keys` is a read-only property derived from `meta.features`, but
# `features` is INSTANCE data, so re-typing the video entries there both disables decoding and survives being
# pickled into `spawn`ed workers (a class-level monkey-patch would NOT: spawned workers re-import the module).
#
# openpi's B1KInputs still looks up the camera keys, so a tiny dummy array is substituted for each. The images are
# never read by the statistics -- only `state` and `actions` are accumulated -- so this cannot change the result.
# That claim is not taken on trust: --verify-against compares the output against stats computed WITH video.


class _NoVideoDataset:
    """Wraps a LeRobot dataset so items carry placeholder images instead of decoded frames."""

    def __init__(self, inner, image_keys, shape=(3, 8, 8)):
        self._inner, self._image_keys, self._shape = inner, list(image_keys), shape

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        item = self._inner[idx]
        for k in self._image_keys:
            if k not in item:
                item[k] = np.zeros(self._shape, dtype=np.float32)
        return item

    def __getattr__(self, name):          # delegate .meta, .features, ... to the real dataset
        # Only reached when normal lookup fails. Inside a SPAWNED DataLoader worker the instance is unpickled
        # before its __dict__ is restored, so `self._inner` itself misses -> here -> `self._inner` -> ... 997
        # frames later RecursionError, and the parent reports only "worker exited unexpectedly".
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)


def _disable_video(dataset):
    """Re-type video features so lerobot skips decoding. Returns the list of keys that were disabled."""
    meta = dataset.meta
    disabled = []
    for holder in (getattr(meta, "features", None), getattr(meta, "info", {}).get("features")):
        if not isinstance(holder, dict):
            continue
        for key, ft in holder.items():
            if isinstance(ft, dict) and ft.get("dtype") == "video":
                ft["dtype"] = "_skipped_video"
                disabled.append(key)
    if meta.video_keys:
        raise RuntimeError(f"failed to disable video decoding; video_keys still {meta.video_keys}")
    return sorted(set(disabled))


def _load_fork_script():
    """Import the fork's scripts/compute_norm_stats.py as a real module so DataLoader worker processes
    can unpickle its classes (RemoveStrings etc.)."""
    import openpi
    scripts_dir = str(pathlib.Path(openpi.__file__).resolve().parents[2] / "scripts")   # <openpi checkout>/scripts
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import compute_norm_stats  # noqa: E402  (now importable by name in workers too)
    return compute_norm_stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-name", default="pi05_b1k")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--assets-base-dir", required=True)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-video", action="store_true",
                   help="skip video decoding entirely (statistics only use state/actions); see the note above")
    p.add_argument("--verify-against", default=None,
                   help="path to a norm_stats.json computed WITH video; asserts this run reproduces it exactly")
    a = p.parse_args()

    cns = _load_fork_script()
    config = _config.get_config(a.config_name)
    base = dataclasses.replace(config.data.base_config, dataset_root=a.dataset_root)
    data = dataclasses.replace(config.data, repo_id=a.repo_id, base_config=base)
    config = dataclasses.replace(config, data=data, assets_base_dir=a.assets_base_dir)
    if a.batch_size:
        config = dataclasses.replace(config, batch_size=a.batch_size)
    if a.num_workers is not None:
        config = dataclasses.replace(config, num_workers=a.num_workers)

    data_config = config.data.create(config.assets_dirs, config.model)
    assert data_config.dataset_root == a.dataset_root, data_config.dataset_root
    if a.no_video:
        import openpi.training.data_loader as _dl
        _orig = _dl.create_b1k_dataset

        def _patched(data_config, action_horizon):
            ds = _orig(data_config, action_horizon)
            inner = ds                                # unwrap TransformedDataset layers (attr is _dataset)
            for _ in range(8):
                if hasattr(inner, "meta"):
                    break
                nxt = getattr(inner, "_dataset", None) or getattr(inner, "dataset", None)
                if nxt is None:
                    raise RuntimeError(f"could not reach the LeRobot dataset; stopped at {type(inner).__name__}")
                inner = nxt
            keys = _disable_video(inner)
            print(f"--no-video: decoding disabled for {len(keys)} video features: {keys}", flush=True)
            return _NoVideoDataset(ds, keys)

        _dl.create_b1k_dataset = _patched
        cns._data_loader.create_b1k_dataset = _patched

    data_loader, num_batches = cns.create_b1k_dataloader(
        data_config, config.model.action_horizon, config.batch_size, config.num_workers, a.max_frames
    )
    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))
    norm_stats = {key: s.get_statistics() for key, s in stats.items()}
    if a.verify_against:
        import json
        ref = json.loads(pathlib.Path(a.verify_against).read_text())["norm_stats"]
        worst, worst_desc = 0.0, ""
        for key in keys:
            got = norm_stats[key]
            for field in ("mean", "std", "q01", "q99"):
                gv = np.asarray(getattr(got, field))
                rv = np.asarray(ref[key][field])
                if gv.shape != rv.shape:
                    print(f"VERIFY FAILED: {key}.{field} shape {gv.shape} != reference {rv.shape}")
                    return 2
                scale = max(float(np.abs(rv).max()), 1e-6)
                d = float(np.abs(gv - rv).max()) / scale
                if d > worst:
                    worst, worst_desc = d, f"{key}.{field}"
        print(f"VERIFY: worst relative deviation {worst:.3e} ({worst_desc}) vs {a.verify_against}")
        if worst > 1e-6:
            print("VERIFY FAILED: --no-video changed the statistics; do NOT use it")
            return 2
        print("VERIFY PASSED: identical to the with-video statistics")

    out = config.assets_dirs / cns.resolve_asset_id(data_config)
    print(f"Writing stats to: {out}")
    normalize.save(out, norm_stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
