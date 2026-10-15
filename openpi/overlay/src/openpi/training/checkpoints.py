# Modified for the 2026 BEHAVIOR Challenge (team Dexterity) -- see openpi/CHANGES.md in the training repository.
from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
import os
from typing import List, Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    # --- B1K addition (T69). On several hosts the parameters are REPLICATED across hosts (FSDP inside a
    # node, data-parallel across nodes), and orbax's default replica-parallel save splits every shard between the
    # replicas with an eager jax.lax.slice. After any other eager JAX work in the process (the held-out exam, the
    # LRSCHED log line), that slice is dispatched to the wrong GPU ("Buffer ... is on device cuda:0, but replica is
    # assigned to device cuda:2") and the save aborts the job -- on 27 Sep at the first mid-run save of the production
    # rehearsal. Without replica parallelism each shard is written whole by one replica: no slicing, no eager op. A
    # single host has one replica, so this changes nothing there. B1K_ORBAX_REPLICA_PARALLEL=1 restores the default.
    if jax.process_count() > 1 and os.environ.get("B1K_ORBAX_REPLICA_PARALLEL", "0") != "1":
        ocp.type_handlers.register_type_handler(
            jax.Array, ocp.type_handlers.ArrayHandler(use_replica_parallel=False), override=True)
        logging.info("orbax: replica-parallel save disabled on multi-host (T69)")
    if overwrite and jax.process_count() > 1:
        # --- B1K addition (multi-host). Upstream wipes the directory on EVERY process. On two nodes both
        # rmtree the same Lustre directory at once: one gets "OSError: [Errno 116] Stale file handle" and dies, the
        # other waits at the next barrier until JAX aborts (a test job, 21 Sep). Only process 0 deletes; everyone then
        # meets at a barrier. The barrier is UNCONDITIONAL: if each process decided via its own exists() check, one
        # could look in the instant between rmtree and mkdir, skip the barrier, and leave the other waiting forever.
        from jax.experimental import multihost_utils
        if jax.process_index() == 0 and checkpoint_dir.exists():
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir} (process 0 of {jax.process_count()})")
        multihost_utils.sync_global_devices("openpi_overwrite_checkpoint_dir")
    elif checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            asset_id = data_config.asset_id if isinstance(data_config.asset_id, str) else data_config.asset_id[0]
            _normalize.save(directory / asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str | List[str]) -> dict[str, _normalize.NormStats] | None:
    if isinstance(asset_id, list):
        # Only load the first asset_id assuming that the datasets are similar
        asset_id = asset_id[0]
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
