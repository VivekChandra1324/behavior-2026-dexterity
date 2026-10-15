# Modified for the 2026 BEHAVIOR Challenge (team Dexterity) -- see openpi/CHANGES.md in the training repository.
from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.training.lerobot_compat as _lerobot_compat
import openpi.transforms as _transforms


def _patch_lerobot_seek_mode() -> None:
    """B1K patch: choose torchcodec's seek mode from the DATA instead of hard-coding it.

    LeRobot hard-codes seek_mode="approximate", which maps a frame index to a position WITHOUT reading the packet
    index. That is correct only when frames are stored in display order, i.e. when the video has no B-frames:
      * the original 2026 challenge videos have none            -> "approximate" is correct AND fast
      * our first 224 transcode (libx265 superfast) has them    -> "approximate" returns the WRONG frame ~37% of
        the time (T39), so those files need "exact"
    "exact" is always correct but scans the packet index on every open: measured 0.821 s per open on the B-frame
    files and still 0.086 s on B-frame-free ones, against 0.003 s for "approximate" (T41). Choosing wrong costs
    either correctness or ~25x the decode time, so we do not guess: on the first decoder the patch probes one file
    with ffprobe and picks "approximate" when has_b_frames == 0, "exact" otherwise. B1K_VIDEO_SEEK_MODE overrides.
    Module level + _worker_init_fn: DataLoader workers are `spawn`ed and import this module, so they decide too.
    """
    try:
        import lerobot.datasets.video_utils as _vu
    except ImportError:
        return
    if getattr(_vu.VideoDecoderCache, "_b1k_patched", False):
        return
    import collections as _collections
    import threading as _threading

    import fsspec
    from torchcodec.decoders import VideoDecoder

    def _mode_for(video_path: str) -> str:
        forced = os.environ.get("B1K_VIDEO_SEEK_MODE")
        if forced:
            return forced
        try:
            import subprocess
            out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                  "stream=has_b_frames", "-of", "csv=p=0", str(video_path)],
                                 capture_output=True, text=True, timeout=60).stdout.strip()
            mode = "approximate" if out and int(out) == 0 else "exact"
        except Exception:
            mode = "exact"                      # the always-correct choice if the probe fails
        logging.info(f"b1k video seek mode: {mode} (probed {video_path})")
        return mode

    # Upstream's cache NEVER evicts: it keeps a decoder and an open file handle for every video the worker has
    # ever touched. With 3 tasks that is 9 files; on the full root each worker wanders over 5,286 files and the
    # process grows without bound until Slurm's OOM killer takes the job (host RAM, not GPU) -- a test job
    # step 69/70, killed at step 59 with 16 workers and step 33 with 32. Keeping the N most recently used
    # decoders bounds it. A miss is cheap now that the videos have no B-frames: 0.003 s to reopen (T41).
    _cap = int(os.environ.get("B1K_DECODER_CACHE", "64"))

    def get_decoder(self, video_path):
        video_path = str(video_path)
        with self._lock:
            if not isinstance(self._cache, _collections.OrderedDict):
                # LeRobot builds its module-level `_default_decoder_cache` at import time -- before this patch
                # replaces __init__ -- so that instance still carries a plain dict, which has no move_to_end.
                # Convert whatever we are handed, which covers instances made before and after the patch.
                self._cache = _collections.OrderedDict(self._cache)
            if video_path in self._cache:
                self._cache.move_to_end(video_path)          # LRU: mark as most recently used
            else:
                mode = getattr(type(self), "_b1k_seek_mode", None) or _mode_for(video_path)
                type(self)._b1k_seek_mode = mode
                file_handle = fsspec.open(video_path).__enter__()
                try:
                    decoder = VideoDecoder(file_handle, seek_mode=mode)
                except Exception:
                    file_handle.close()
                    raise
                self._cache[video_path] = (decoder, file_handle)
                while len(self._cache) > _cap:               # evict the least recently used, closing its handle
                    _, (_, old_handle) = self._cache.popitem(last=False)
                    try:
                        old_handle.close()
                    except Exception:
                        pass
            return self._cache[video_path][0]

    def _b1k_init(self):
        self._cache = _collections.OrderedDict()             # ordered so eviction can pick the oldest
        self._lock = _threading.Lock()

    _vu.VideoDecoderCache.get_decoder = get_decoder
    _vu.VideoDecoderCache.__init__ = _b1k_init
    _vu.VideoDecoderCache._b1k_patched = True
    logging.info(f"b1k decoder cache: at most {_cap} open videos per worker (upstream: unbounded)")


_patch_lerobot_seek_mode()

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        # B1K patch (T54): a single bad video read must not kill a multi-day multi-node job. One
        # FrameTimestampError in one worker on one node took down a 2-node run (the surviving process was aborted by
        # the JAX coordination service). Retry a few times with a different random index, logging each failure so
        # a systematic problem stays visible; give up only if every attempt fails. B1K_GETITEM_RETRIES=0 disables.
        retries = int(os.environ.get("B1K_GETITEM_RETRIES", "8"))
        for attempt in range(retries + 1):
            try:
                return self._transform(self._dataset[index])
            except Exception as e:  # noqa: BLE001 -- anything the reader raises; the job must live
                if attempt >= retries:
                    raise
                logging.warning("sample %s failed (%s: %s); resampling (attempt %d/%d)",
                                index, type(e).__name__, str(e)[:160], attempt + 1, retries)
                index = int(np.random.randint(len(self._dataset)))
        raise RuntimeError("unreachable")

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_b1k_dataset(data_config: _config.DataConfig, action_horizon: int) -> Dataset:
    """Create a behavior dataset for training."""
    if isinstance(data_config.repo_id, list):
        dataset_meta = _lerobot_compat.LeRobotDatasetMetadata(
            repo_id=data_config.repo_id[0], root=os.path.join(data_config.dataset_root, data_config.repo_id[0])
        )
        dataset_kwargs = {"repo_ids": data_config.repo_id, **data_config.dataset_kwargs}
    else:
        dataset_meta = _lerobot_compat.LeRobotDatasetMetadata(
            repo_id=data_config.repo_id, root=data_config.dataset_root
        )
        dataset_kwargs = {"repo_id": data_config.repo_id, **data_config.dataset_kwargs}
    dataset = data_config.data_cls(
        root=data_config.dataset_root,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        **dataset_kwargs,
    )

    # B1K patch (opt-in, B1K_SKIP_DEPTH=1): do not decode the depth video streams.
    # pi0.5's R1Pro config reads only observation.rgb.* -- depth is never used -- yet LeRobot decodes EVERY video
    # feature per sample. Depth is 2.08 TB of the 3.09 TB dataset (2.2x the RGB bytes), so on the full root it
    # dominates Lustre I/O: the training loader measured 10.1 samples/s against the 78 the GPUs need (a test job).
    # The reader skips any key absent from meta.video_keys, a property derived from meta.features; `features` is
    # INSTANCE data, so re-typing the depth entries survives pickling into `spawn`ed workers (a class-level patch
    # would not). RGB, state and actions take the unchanged code path -- verified by scripts/profiling/
    # verify_skip_depth_parity.py. See our project notes.
    if os.environ.get("B1K_SKIP_DEPTH", "0") == "1":
        _meta = getattr(dataset, "meta", None)
        _depth = set(getattr(_meta, "depth_keys", []) or [])
        _skipped = []
        for _holder in (getattr(_meta, "features", None), (getattr(_meta, "info", None) or {}).get("features")):
            if isinstance(_holder, dict):
                for _k, _ft in _holder.items():
                    if _k in _depth and isinstance(_ft, dict) and _ft.get("dtype") == "video":
                        _ft["dtype"] = "_skipped_depth_video"
                        _skipped.append(_k)
        _left = [k for k in _meta.video_keys if k in _depth]
        if _left:
            raise RuntimeError(f"B1K_SKIP_DEPTH: failed to disable depth decoding for {_left}")
        logging.info(f"B1K_SKIP_DEPTH=1: depth decoding disabled for {sorted(set(_skipped))}; "
                     f"still decoding {_meta.video_keys}")

    if data_config.prompt_from_task:
        dataset = TransformedDataset(
            dataset, [_transforms.PromptFromLeRobotTask(_lerobot_compat.tasks_from_metadata(dataset_meta))]
        )

    return dataset


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = _lerobot_compat.LeRobotDatasetMetadata(repo_id)
    dataset = _lerobot_compat.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        episodes=data_config.episodes_index,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(
            dataset, [_transforms.PromptFromLeRobotTask(_lerobot_compat.tasks_from_metadata(dataset_meta))]
        )

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )



class TaskTemperedSampler(torch.utils.data.Sampler):
    """B1K addition (B1K_TASK_ALPHA, our project notes): draw a task with probability
    proportional to (its frame count)^alpha, then a frame uniformly inside that task's contiguous index block.
    alpha=1 is frame-proportional (stock), 0 is task-uniform. Sampling is with replacement; each process (rank), each
    resumed link (resume_step) and each pass draws its own stream, reproducibly (T65)."""

    def __init__(self, ranges, *, alpha: float, num_samples: int, seed: int, rank: int = 0, resume_step: int = 0):
        self._starts = np.asarray([a for a, _ in ranges], np.int64)
        self._sizes = np.asarray([b - a for a, b in ranges], np.int64)
        w = self._sizes.astype(np.float64) ** alpha
        self._p = w / w.sum()
        self._n, self._key, self._pass = int(num_samples), (int(seed), int(rank), int(resume_step)), 0

    def shares(self) -> np.ndarray:
        return self._p

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([*self._key, self._pass]))
        self._pass += 1
        left = self._n
        while left > 0:
            k = min(left, 1_000_000)
            t = rng.choice(len(self._p), size=k, p=self._p)
            idx = self._starts[t] + (rng.random(k) * self._sizes[t]).astype(np.int64)
            yield from idx.tolist()
            left -= k


def _b1k_task_ranges(dataset_len: int) -> list[tuple[int, int]]:
    """Per-task [start, end) index blocks from the dataset root's episode metadata; refuses if they do not tile the
    dataset exactly (e.g. a subset root with re-based indices), because then the sampler would draw wrong frames."""
    import glob

    import pyarrow.parquet as pq

    root = os.environ["B1K_DATASET_ROOT"]
    rows = []
    for f in sorted(glob.glob(f"{root}/meta/episodes/chunk-*/*.parquet")):
        rows += pq.read_table(f, columns=["task_index", "dataset_from_index", "dataset_to_index"]).to_pylist()
    by = {}
    for r in rows:
        a, b = by.get(r["task_index"], (None, None))
        by[r["task_index"]] = (r["dataset_from_index"] if a is None else min(a, r["dataset_from_index"]),
                               r["dataset_to_index"] if b is None else max(b, r["dataset_to_index"]))
    ranges = sorted(by.values())
    total = sum(r["dataset_to_index"] - r["dataset_from_index"] for r in rows)
    if total != dataset_len or sum(b - a for a, b in ranges) != total or ranges[0][0] != 0 or ranges[-1][1] != dataset_len:
        raise ValueError(f"B1K_TASK_ALPHA: task blocks do not tile the dataset ({len(ranges)} tasks, {total} frames in "
                         f"metadata vs {dataset_len} in the dataset); refusing to sample")
    return ranges


def create_b1k_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_b1k_dataset(data_config=data_config, action_horizon=config.model.action_horizon)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # --- B1K addition (multi-host): THIS is the loader B1K training uses (train_b1k.py:456), not
    # create_torch_data_loader. Without a per-process sampler every node shuffles the same dataset with the same seed
    # and loads IDENTICAL samples -- found on 2-node a test job, where the global batch of 8 was the same 4 samples
    # twice: correct gradients, double the compute for the same data. DistributedSampler gives each process a disjoint
    # shard. Inert on one process (sampler stays None, behaviour unchanged).
    sampler = None
    # T65: a resumed link must NOT replay the stream from its start. Checkpoints do not store the loader position, so
    # every link used to draw the same permutation (same seed, epoch 0) -- ~2-day links would all re-read the same
    # first ~10% of the data. The stream is now keyed by the step the link resumes from (0 on a fresh run, which keeps
    # fresh runs bit-identical to before). train_b1k.py sets B1K_RESUME_STEP.
    _resume = int(os.environ.get("B1K_RESUME_STEP", "0"))
    _alpha = os.environ.get("B1K_TASK_ALPHA")
    if _alpha:
        sampler = TaskTemperedSampler(
            _b1k_task_ranges(len(dataset)), alpha=float(_alpha), num_samples=len(dataset) // jax.process_count(),
            seed=config.seed, rank=jax.process_index(), resume_step=_resume)
        _sh = np.sort(sampler.shares())
        logging.info(f"task-tempered sampler: alpha={_alpha} process {jax.process_index()}/{jax.process_count()} "
                     f"resume_step={_resume}; task shares min {100*_sh[0]:.2f}% median {100*np.median(_sh):.2f}% "
                     f"max {100*_sh[-1]:.2f}% over {len(_sh)} tasks")
    elif jax.process_count() > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=jax.process_count(), rank=jax.process_index(),
            shuffle=shuffle, seed=config.seed, drop_last=True,
        )
        sampler.set_epoch(_resume)
        _first = [i for _, i in zip(range(3), iter(sampler))]
        logging.info(f"multi-host sampler (b1k): process {jax.process_index()}/{jax.process_count()} gets "
                     f"{len(sampler)} of {len(dataset)} samples; resume_step={_resume}; first indices {_first}")

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        sampler=sampler,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "jax" and jax.process_count() > 1:
        # --- B1K addition (multi-host, opt-in via B1K_MULTIHOST=1): give each JAX process a DISJOINT
        # shard of the dataset, exactly as DistributedSampler does for torch DDP below. Without it every node would
        # shuffle with the same seed and draw identical batches -- two nodes' compute for one node's data.
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=jax.process_count(), rank=jax.process_index(),
            shuffle=shuffle, seed=seed, drop_last=True,
        )
        _first = next(iter(sampler), None)
        logging.info(f"multi-host sampler: process {jax.process_index()}/{jax.process_count()} gets "
                     f"{len(sampler)} of {len(dataset)} samples (first index {_first}); shards are disjoint by construction")
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        # Upstream refuses multi-process loading. The machinery below already supports it (per-process local batch,
        # jax.make_array_from_process_local_data), so the This fork lifts the guard ONLY when explicitly opted
        # in; the per-process sampler that makes it correct is added in create_torch_data_loader.
        if jax.process_count() > 1 and os.environ.get("B1K_MULTIHOST") != "1":
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            # B1K addition (T70, opt-in B1K_LOADER_IN_ORDER=0): hand over batches as they complete. In
            # order (torch default), ONE worker stuck on a slow filesystem read idles the whole node even when the
            # other workers have later batches ready. The sampler is random, so batch order carries no meaning.
            in_order=os.environ.get("B1K_LOADER_IN_ORDER", "1") != "0",
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    _patch_lerobot_seek_mode()


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
