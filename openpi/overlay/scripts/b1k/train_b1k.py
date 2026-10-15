# Modified for the 2026 BEHAVIOR Challenge (team Dexterity) -- see openpi/CHANGES.md in the training repository.
import dataclasses
import os
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def _extract_training_norm_stats(train_data_loader: _data_loader.DataLoader | None) -> dict | None:
    """Extract normalization stats from the training data loader."""
    if train_data_loader is not None and hasattr(train_data_loader, "data_config"):
        training_data_config = train_data_loader.data_config()
        if hasattr(training_data_config, "norm_stats") and training_data_config.norm_stats is not None:
            return training_data_config.norm_stats
    return None


def _prepare_validation_config(
    config: _config.TrainConfig, training_norm_stats: dict | None
) -> tuple[_config.TrainConfig, str, bool]:
    """
    Prepare validation configuration with training norm_stats.

    Returns:
        tuple: (validation_config, repo_id, use_norm_stats, actual_val_data_config)
    """
    val_config = dataclasses.replace(
        config,
        batch_size=config.val_batch_size or config.batch_size,
    )

    use_norm_stats = False
    if config.val_repo_id or hasattr(val_config.data, "repo_id"):
        repo_id = config.val_repo_id or (getattr(val_config.data, "repo_id", None) + "-val")
        episodes_index = config.val_episodes_index

        # Create validation data config by copying the training data config but changing repo_id
        val_base_config = dataclasses.replace(val_config.data.base_config, episodes_index=episodes_index)
        val_data_config = dataclasses.replace(val_config.data, repo_id=repo_id, base_config=val_base_config)
        val_config = dataclasses.replace(val_config, data=val_data_config)

        # Create the actual DataConfig using the factory and add norm_stats
        actual_val_data_config = val_config.data.create(val_config.assets_dirs, val_config.model)

        # Explicitly use norm_stats from training data loader for validation
        if training_norm_stats is not None:
            actual_val_data_config = dataclasses.replace(actual_val_data_config, norm_stats=training_norm_stats)
            logging.info("Copied norm_stats from training data loader to validation config")
            logging.info("Training norm_stats keys: %s", list(training_norm_stats.keys()))
            use_norm_stats = True
        else:
            logging.warning("No norm_stats found in training data loader - skipping normalization for validation")

        return val_config, repo_id, use_norm_stats, actual_val_data_config

    raise ValueError("No validation repository ID could be determined")



def _create_validation_data_loader(
    actual_val_data_config,
    val_config: _config.TrainConfig,
    *,
    use_norm_stats: bool,
    replicated_sharding: jax.sharding.NamedSharding,
) -> _data_loader.DataLoader:
    """Create validation data loader with custom wrapper for training norm_stats."""

    # Custom wrapper class to bridge our modified DataConfig with the expected DataLoader interface
    class ValidationDataLoader(_data_loader.DataLoader):
        """
        Custom data loader wrapper for validation that preserves training normalization stats.

        This class is necessary because:
        1. Validation datasets typically don't have their own norm_stats.json files
        2. We need to use the training dataset's normalization stats for consistent validation
        3. The standard create_data_loader() function expects a TrainConfig, but we have a custom DataConfig
        """

        def __init__(self, data_config, torch_data_loader):
            self._data_config = data_config
            self._torch_data_loader = torch_data_loader

        def data_config(self):
            return self._data_config

        def __iter__(self):
            for batch in self._torch_data_loader:
                yield _model.Observation.from_dict(batch), batch["actions"]

    val_dataset = _data_loader.create_b1k_dataset(actual_val_data_config, val_config.model.action_horizon)
    logging.info(f"Validation dataset created for {actual_val_data_config.repo_id}")
    val_dataset = _data_loader.transform_dataset(
        val_dataset, actual_val_data_config, skip_norm_stats=not use_norm_stats
    )

    val_torch_data_loader = _data_loader.TorchDataLoader(
        val_dataset,
        local_batch_size=val_config.batch_size // jax.process_count(),
        sharding=replicated_sharding,
        shuffle=False,
        num_batches=val_config.val_num_batches,
        num_workers=val_config.num_workers,
        seed=val_config.seed,
    )

    return ValidationDataLoader(actual_val_data_config, val_torch_data_loader)


def _compute_validation_losses(
    val_loader: _data_loader.DataLoader,
    train_state: training_utils.TrainState,
    mesh: jax.sharding.Mesh,
    train_state_sharding: jax.sharding.NamedSharding,
    replicated_sharding: jax.sharding.NamedSharding,
    config: _config.TrainConfig,
) -> float | None:
    """Compute validation losses over multiple batches."""

    # Define validation loss function (similar to train_step structure)
    @at.typecheck
    def val_loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=False)
        return jnp.mean(chunked_loss)

    def validation_step(state, batch, rng):
        """Single validation step, aligned with train_step structure."""
        model = nnx.merge(state.model_def, state.params)
        model.eval()

        observation, actions = batch
        val_rng = jax.random.fold_in(rng, state.step)

        return val_loss_fn(model, val_rng, observation, actions)

    # JIT compile the validation step
    pvalidation_step = jax.jit(
        validation_step,
        in_shardings=(train_state_sharding, replicated_sharding, replicated_sharding),
        out_shardings=replicated_sharding,
    )

    val_iter = iter(val_loader)
    losses = []
    # Use a separate RNG for validation to avoid interference with training RNG,
    # the specific seed offset is arbitrary.
    val_rng = jax.random.key(config.seed + 1000)

    for batch_idx in range(config.val_num_batches):
        try:
            batch = next(val_iter)
        except StopIteration:
            break

        try:
            with sharding.set_mesh(mesh):
                loss = pvalidation_step(train_state, batch, val_rng)
            losses.append(jax.device_get(loss))
        except (RuntimeError, ValueError) as e:
            logging.warning("Error computing validation loss for batch %d: %s", batch_idx, e)
            continue

    if not losses:
        return None
    return float(jnp.mean(jnp.array(losses)))


def compute_validation_loss(
    config: _config.TrainConfig,
    train_state: training_utils.TrainState,
    mesh: jax.sharding.Mesh,
    train_state_sharding: jax.sharding.NamedSharding,
    replicated_sharding: jax.sharding.NamedSharding,
    train_data_loader: _data_loader.DataLoader | None = None,
) -> float | None:
    """Compute average validation loss over a few batches. Downloads validation dataset if missing locally."""
    # Extract training normalization stats
    training_norm_stats = _extract_training_norm_stats(train_data_loader)

    # Prepare validation configuration
    try:
        val_config, repo_id, use_norm_stats, actual_val_data_config = _prepare_validation_config(
            config, training_norm_stats
        )
    except ValueError:
        logging.warning("Could not determine validation repository ID, skipping validation loss.")
        return None

    # Check if validation dataset exists locally (same behavior as training dataset)
    is_local = True

    # Create validation data loader (this will use local data or trigger download if needed)
    try:
        val_loader = _create_validation_data_loader(
            actual_val_data_config, val_config, use_norm_stats=use_norm_stats, replicated_sharding=replicated_sharding
        )
        if not is_local:
            logging.info(f"Validation dataset downloaded successfully: {repo_id}")
    except Exception as e:
        logging.warning(f"Failed to create validation data loader for {repo_id}: {e}")
        logging.warning("Skipping validation loss computation.")
        return None

    # Compute and return validation losses
    return _compute_validation_losses(val_loader, train_state, mesh, train_state_sharding, replicated_sharding, config)


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            group="openpi"
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    # B1K addition: log the schedule the optimizer ACTUALLY uses (WSD override included), so a run's
    # learning rate is verifiable from its log -- also after a resume with changed settings.
    _sched = _optimizer.wsd_from_env(config.lr_schedule) or config.lr_schedule.create()
    _n = config.num_train_steps
    _pts = sorted({0, 100, 500, 1000, 5000, 10000, _n // 4, _n // 2, (3 * _n) // 4, _n - 1} & set(range(_n)))
    logging.info(f"LRSCHED source={'B1K_WSD=' + os.environ['B1K_WSD'] if os.environ.get('B1K_WSD') else type(config.lr_schedule).__name__} "
                 f"peak={config.lr_schedule.peak_lr:g} warmup={config.lr_schedule.warmup_steps} steps={_n} "
                 + " ".join(f"lr@{p}={float(_sched(p)):.3e}" for p in _pts))

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    # --- B1K addition: multi-node training, opt-in via B1K_MULTIHOST=1. Upstream openpi never calls
    # jax.distributed.initialize(), so every process would see only its own node's GPUs and train alone. With no
    # arguments JAX auto-detects the Slurm topology (SLURM_JOB_ID, SLURM_NTASKS, SLURM_PROCID, SLURM_STEP_NODELIST);
    # launch with `srun --ntasks-per-node=1` so each node runs one process owning its 4 GPUs. Must run before any
    # JAX computation (it is placed after init_logging(), which does not touch JAX -- placed before, the
    # confirmation log line was silently dropped because logging was not yet configured). Single-node runs are
    # unaffected.
    if os.environ.get("B1K_MULTIHOST") == "1":
        coord = os.environ.get("B1K_COORDINATOR")
        if coord:
            # Explicit coordinator on a verified PRIVATE IP (set by the job's rendezvous stage). Auto-detection uses
            # the first node's hostname, which on our cluster resolves to the PUBLIC interface -- the suspected cause of
            # the "worker never joins the master" failure. Bounded, so a failed join errors instead of hanging.
            # local_device_ids MUST be passed explicitly. If ANY of the four arguments is None, JAX falls back to
            # its Slurm cluster detector for that argument, and the detector assumes the standard "one process per
            # GPU" layout: it returns local_device_ids = [SLURM_LOCALID] and pins the process to that single GPU.
            # We run ONE process per node owning all 4 GPUs, so every process was pinned to device 0 and JAX saw
            # 2 devices instead of 8 -- a test job died at `Number of devices 2 must be divisible by the number
            # of FSDP devices 4` (T40). The 2 x 1-GPU tests could not expose this. SLURM_GPUS_ON_NODE is what Slurm
            # reports for this step; B1K_LOCAL_DEVICES overrides it.
            _ndev = int(os.environ.get("B1K_LOCAL_DEVICES") or os.environ.get("SLURM_GPUS_ON_NODE") or 0)
            if _ndev <= 0:
                raise RuntimeError("multi-host: cannot determine the GPU count on this node (SLURM_GPUS_ON_NODE "
                                   "unset). Set B1K_LOCAL_DEVICES to the GPUs per process.")
            jax.distributed.initialize(coordinator_address=coord, num_processes=int(os.environ["SLURM_NTASKS"]),
                                       process_id=int(os.environ["SLURM_PROCID"]), initialization_timeout=180,
                                       local_device_ids=list(range(_ndev)))
        else:
            jax.distributed.initialize()
        logging.info(f"multi-host: process {jax.process_index()}/{jax.process_count()}, "
                     f"{jax.local_device_count()} local / {jax.device_count()} global devices")
    # GUARD against the silent failure of smoke a test job (21 Sep): srun launched one task per node, but
    # B1K_MULTIHOST never reached python, so each node trained ALONE -- same data, same seed, identical logs, and it
    # looked like success. If Slurm started several tasks in this step but JAX sees one process, stop.
    if os.environ.get("B1K_MULTIHOST") == "1":
        _want = int(os.environ.get("B1K_LOCAL_DEVICES") or os.environ.get("SLURM_GPUS_ON_NODE") or 0)
        if _want and jax.local_device_count() != _want:
            raise RuntimeError(f"multi-host: this process owns {jax.local_device_count()} GPU(s) but Slurm gave the "
                               f"node {_want}. Every GPU would sit idle or the mesh would fail (T40).")
    _step_tasks = int(os.environ.get("SLURM_STEP_NUM_TASKS", "1"))
    if _step_tasks > 1 and jax.process_count() == 1:
        raise RuntimeError(f"srun started {_step_tasks} tasks but JAX sees 1 process: every task would train "
                           f"independently on identical data. Set B1K_MULTIHOST=1 (exported, not a prefix on `cd`).")

    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    # B1K patch: upstream hardcodes ~/.cache/jax. $HOME here is quota- and inode-constrained and
    # the project rule is that nothing is cached there. Because this is an explicit jax.config.update(),
    # the JAX_COMPILATION_CACHE_DIR environment variable CANNOT override it -- it has to be patched.
    _jax_cache = os.environ.get("JAX_COMPILATION_CACHE_DIR") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", str(epath.Path("~/.cache").expanduser())), "jax")
    jax.config.update("jax_compilation_cache_dir", _jax_cache)
    logging.info(f"JAX compilation cache: {_jax_cache}")

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    # multi-host: only process 0 owns the W&B run (the others would each start a duplicate). No-op on one node.
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled and jax.process_index() == 0)

    # T65: key the data stream to the step this link resumes from, so a resumed link does not replay the previous
    # link's samples (checkpoints do not store the loader position).
    os.environ["B1K_RESUME_STEP"] = str((checkpoint_manager.latest_step() or 0) if resuming else 0)
    logging.info(f"data stream key: resume_step={os.environ['B1K_RESUME_STEP']}")
    data_loader = _data_loader.create_b1k_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        skip_norm_stats=False
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    # B1K patch: only on a fresh run. On a resume the trainer is past step 0, and W&B
    # silently DROPS a log at a step lower than the current one -- and offline resume reuses the run id.
    if not resuming:
        wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    # B1K addition: record this link's runtime settings in the W&B config. The production switches
    # (normalization, WSD schedule, sampler, staging...) live in B1K_* environment variables, not in the dataclass
    # config, and the dataclass alone shows the stock LR schedule. One entry per link: its job id and start step.
    # The full dataclass config is re-sent too: an offline run resumed with resume="must" holds only what this link
    # sends, and its sync must not leave the run's config reduced to the link entry.
    if wandb.run is not None:
        wandb.config.update({**dataclasses.asdict(config), f"links/{os.environ.get('SLURM_JOB_ID', 'local')}": {
            "start_step": start_step, **{k: v for k, v in sorted(os.environ.items()) if k.startswith("B1K_")}}},
            allow_val_change=True)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    # B1K addition (opt-in: B1K_LOG_DATAWAIT=1): how long the loop blocks waiting for the next batch,
    # per log interval. Near 0 = the loader is ahead of the GPUs; close to the step time = the GPUs are starved.
    import time as _time
    _dw, _dw_t0, _dw_n = 0.0, _time.perf_counter(), 0
    # The LR the optimizer applies at a step, for W&B. Evaluated on the host CPU backend, never as an eager GPU op.
    _lr_sched = _optimizer.wsd_from_env(config.lr_schedule) or config.lr_schedule.create()

    def _lr_at(s):
        try:
            with jax.default_device(jax.devices("cpu")[0]):
                return float(_lr_sched(s))
        except Exception:  # noqa: BLE001 -- a logging value must never stop training
            return float("nan")
    for step in pbar:
        # B1K addition (opt-in: B1K_LOG_MEM=1): an exact per-GPU memory breakdown, once, before the
        # first step. The training-state split is measured from the actual on-device shards (weights, optimizer
        # state, EMA copy); the compiled step's memory analysis gives the batch inputs (arguments beyond the state)
        # and the temporary working memory (activations, gradients in flight, communication buffers).
        if step == start_step and os.environ.get("B1K_LOG_MEM") == "1":
            try:
                def _dev0_bytes(tree):
                    tot = 0
                    for leaf in jax.tree_util.tree_leaves(tree):
                        shards = getattr(leaf, "addressable_shards", None)
                        if shards:
                            tot += shards[0].data.nbytes
                    return tot / 2**30
                _w, _o = _dev0_bytes(train_state.params), _dev0_bytes(train_state.opt_state)
                _e = _dev0_bytes(train_state.ema_params) if train_state.ema_params is not None else 0.0
                _b = _dev0_bytes(batch)
                logging.info(f"MEMBREAK state_per_gpu_GiB weights={_w:.3f} optimizer={_o:.3f} ema={_e:.3f} "
                             f"batch_inputs={_b:.3f}")
                # compile() rebuilds TrainState from abstract ArgInfo leaves, which the jaxtyping check rejects.
                # B1K_MEMBREAK_COMPILE=0 skips this extra compile (its autotuning scratch may fragment the pool).
                _ma = None
                if os.environ.get("B1K_MEMBREAK_COMPILE", "1") == "1":
                    with sharding.set_mesh(mesh), at.disable_typechecking():
                        _ma = ptrain_step.lower(train_rng, train_state, batch).compile().memory_analysis()
                if _ma is not None:
                    logging.info(
                        "MEMBREAK compiled_step_per_gpu_GiB "
                        f"arguments={_ma.argument_size_in_bytes / 2**30:.3f} outputs={_ma.output_size_in_bytes / 2**30:.3f} "
                        f"aliased={_ma.alias_size_in_bytes / 2**30:.3f} temporaries={_ma.temp_size_in_bytes / 2**30:.3f} "
                        f"code={_ma.generated_code_size_in_bytes / 2**30:.3f}")
            except Exception as _e_mem:  # noqa: BLE001 -- a diagnostic must never stop training
                logging.warning(f"MEMBREAK failed: {type(_e_mem).__name__}: {_e_mem}")
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            # B1K addition (opt-in: B1K_LOG_MEM=1): the REAL device memory in use, from JAX's allocator.
            # With XLA_PYTHON_CLIENT_MEM_FRACTION the pool is reserved up front, so nvidia-smi always shows ~76 GiB
            # whatever the model needs; peak_bytes_in_use is what the batch actually costs.
            if os.environ.get("B1K_LOG_MEM") == "1":
                _ms = [d.memory_stats() or {} for d in jax.local_devices()]
                _pk = max(m.get("peak_bytes_in_use", 0) for m in _ms) / 2**30
                _cur = max(m.get("bytes_in_use", 0) for m in _ms) / 2**30
                _lim = max(m.get("bytes_limit", 0) for m in _ms) / 2**30
                logging.info(f"MEM step={step} peak_in_use_GiB={_pk:.2f} in_use_GiB={_cur:.2f} limit_GiB={_lim:.2f}")
            # B1K addition: throughput, LR, real memory and loader wait go to W&B as well as the text log.
            _wall = _time.perf_counter() - _dw_t0
            _extra = {"lr": _lr_at(step)}
            if _dw_n and _wall > 0:
                _extra.update(sec_per_step=_wall / _dw_n, samples_per_sec=config.batch_size * _dw_n / _wall,
                              data_wait_frac=_dw / _wall)
                # Model FLOPs Utilization (opt-in: B1K_TFLOP_PER_SAMPLE, the model's training FLOPs per sample,
                # our MFU estimate) against the H100 SXM dense BF16 peak of 989 TFLOP/s per GPU.
                if os.environ.get("B1K_TFLOP_PER_SAMPLE"):
                    _tf = _extra["samples_per_sec"] * float(os.environ["B1K_TFLOP_PER_SAMPLE"]) / jax.device_count()
                    _extra.update(tflops_per_gpu=_tf, mfu=_tf / 989.0)
                if os.environ.get("B1K_LOG_DATAWAIT") == "1":
                    logging.info(f"DATAWAIT step={step} steps={_dw_n} wall_s_per_step={_wall/_dw_n:.3f} "
                                 f"wait_s_per_step={_dw/_dw_n:.3f} wait_frac={_dw/_wall:.3f}")
            if os.environ.get("B1K_LOG_MEM") == "1":
                _extra["mem_peak_gib"] = _pk
            _dw, _dw_t0, _dw_n = 0.0, _time.perf_counter(), 0
            wandb.log({**reduced_info, **_extra}, step=step)
            infos = []
        if config.val_log_interval and step % config.val_log_interval == 0:
            val_loss = compute_validation_loss(
                config, train_state, mesh, train_state_sharding, replicated_sharding, data_loader
            )
            if val_loss is not None:
                wandb.log({"val_loss": val_loss}, step=step)
                logging.info("Validation loss at step %d: %.4f", step, val_loss)
        _t_wait = _time.perf_counter()
        batch = next(data_iter)
        _dw += _time.perf_counter() - _t_wait
        _dw_n += 1

        # B1K addition (opt-in: B1K_EVAL_ACTION_ERROR=<n_batches>): mean absolute action error in
        # RAW joint units. Training loss cannot compare normalisation conventions -- z-score and quantile scale the
        # targets differently, so their losses sit on different axes (measured: 0.2515 vs 0.9386 from identical
        # weights). Un-normalising prediction and target puts every run on the same physical axis.
        # It runs HERE, just before the save, because the save below always fires on the final step regardless of
        # save_interval, and a 12 GB async save OOMs a 64 GB host -- which would kill the process before any
        # evaluation placed after the loop. Inert unless the variable is set.
        # B1K_EVAL_AT_STEPS="600,1200,...": also grade at these intermediate steps (learning curves on the fixed set).
        _eval_at = {int(x) for x in os.environ.get("B1K_EVAL_AT_STEPS", "").split(",") if x.strip()}
        if (step == config.num_train_steps - 1 or step in _eval_at) and int(os.environ.get("B1K_EVAL_ACTION_ERROR", "0")):
            import numpy as _np

            _n_eval = int(os.environ["B1K_EVAL_ACTION_ERROR"])
            _stats = config.data.create(config.assets_dirs, config.model).norm_stats["actions"]
            _use_q = os.environ.get("B1K_QUANTILE_NORM", "0") == "1"
            _mean, _std = _np.asarray(_stats.mean), _np.asarray(_stats.std)

            # The loader pads actions from the robot's 23 real dimensions to the model's 32; the padding carries
            # no statistics and no meaning, so the comparison uses the real dimensions only.
            _real = len(_mean)

            import openpi.transforms as _tf

            def _unnorm(x):  # the serving path's own inverse, so every mode (z, q, B1K_NORM_ACTIONS=zstep) is covered
                return _tf.Unnormalize({"actions": _stats}, use_quantiles=_use_q)({"actions": x})["actions"][..., :_real]

            # B1K_EVAL_USE_EMA=1: grade the EMA weights -- what a saved checkpoint serves (checkpoints.py stores
            # ema_params as the params). Default: the raw weights, as every E5 score so far.
            _use_ema = os.environ.get("B1K_EVAL_USE_EMA") == "1" and train_state.ema_params is not None
            _eval_model = nnx.merge(train_state.model_def, train_state.ema_params if _use_ema else train_state.params)
            # On one host the sampler runs eagerly (as every E5 score so far). Across hosts eager ops would mix
            # process-local noise with global arrays, so it is jit-compiled there (as the serving policy does).
            _sample = (nnx_utils.module_jit(_eval_model.sample_actions) if jax.process_count() > 1
                       else _eval_model.sample_actions)
            _key = jax.random.key(0)
            _errs = []
            # B1K_EVAL_FIXED=1: grade every run on the SAME examples. Without it the eval batches are just the next
            # batches of this run's own training stream, so two arms are scored on different data and their batch-
            # to-batch spread is mixed into the comparison. With it, a fixed list of dataset indices (seeded by
            # B1K_EVAL_SEED, identical for every arm on the same root) is read in the same order, so arm-to-arm
            # differences are paired. At ~0.1% of an epoch per test run, the chance a fixed example was also a
            # training example is ~0.1%, equal for every arm.
            if os.environ.get("B1K_EVAL_FIXED") == "1":
                import torch as _torch
                _tdl = data_loader._data_loader
                _ds = _tdl._data_loader.dataset
                _rs = _np.random.default_rng(int(os.environ.get("B1K_EVAL_SEED", "20260925")))
                _idx = _rs.choice(len(_ds), size=_n_eval * config.batch_size, replace=False).tolist()
                # Several hosts: each process reads its own contiguous slice of every global batch, so the global
                # batches hold exactly the same examples as on one host (predictions are gathered below).
                _P, _p = jax.process_count(), jax.process_index()
                _bl = config.batch_size // _P
                _idx_local = [i for b in range(_n_eval)
                              for i in _idx[b * config.batch_size + _p * _bl: b * config.batch_size + (_p + 1) * _bl]]

                class _FixedOrder(_torch.utils.data.Sampler):
                    def __iter__(self):
                        return iter(_idx_local)

                    def __len__(self):
                        return len(_idx_local)

                _eval_iter = iter(_data_loader.DataLoaderImpl(data_loader.data_config(), _data_loader.TorchDataLoader(
                    _ds, local_batch_size=_bl, sharding=_tdl._sharding, sampler=_FixedOrder(),
                    shuffle=False, num_batches=_n_eval, num_workers=min(8, config.num_workers), seed=0)))
                logging.info(f"ACTION_ERROR fixed eval set: {len(_idx)} indices, first {_idx[:4]}")
            else:
                _eval_iter = data_iter
            _by_step = []
            for _ in range(_n_eval):
                _obs, _act = next(_eval_iter)
                _key, _sub = jax.random.split(_key)
                _pred = _sample(_sub, _obs, num_steps=10)
                if jax.process_count() > 1:  # global arrays span hosts: gather the whole batch on every process
                    from jax.experimental import multihost_utils
                    _pred = multihost_utils.process_allgather(_pred, tiled=True)
                    _act = multihost_utils.process_allgather(_act, tiled=True)
                _pred, _tgt = _np.asarray(_pred), _np.asarray(_act)
                _ae = _np.abs(_unnorm(_pred) - _unnorm(_tgt))            # (B, H, real)
                _errs.append(_ae.mean())
                _by_step.append(_ae.mean(axis=(0, 2)))                   # error per horizon step
            logging.info(
                f"ACTION_ERROR step={step} ema={int(_use_ema)} quantile={int(_use_q)} norm_state={os.environ.get('B1K_NORM_STATE', '-')} norm_actions={os.environ.get('B1K_NORM_ACTIONS', '-')} batches={_n_eval} "
                f"MAE_raw_units={float(_np.mean(_errs)):.6f} std={float(_np.std(_errs)):.6f} "
                f"per_batch={','.join(f'{e:.6f}' for e in _errs)}"
            )
            _bs = _np.mean(_by_step, axis=0)
            logging.info(f"ACTION_ERROR_BY_STEP step={step} first8={_bs[:8].mean():.6f} steps8_15={_bs[8:16].mean():.6f} "
                         f"last16={_bs[16:].mean():.6f} by_step={','.join(f'{e:.6f}' for e in _bs)}")


        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
            # B1K addition (opt-in: B1K_HANDOFF_MARKER=<path>). The queued successor job, if it is
            # given nodes while this job is still running, writes that file and waits for this job to exit. Stop
            # here -- right after a save, never mid-interval -- so the successor resumes from exactly this step and
            # nothing is recomputed. On several hosts every process must reach the same decision, or the ones that
            # carry on hang in the next collective, so process 0's view of the file is what counts. The save is
            # asynchronous and must be made durable before exiting. our project notes section 8.
            _marker = os.environ.get("B1K_HANDOFF_MARKER")
            if _marker and step < config.num_train_steps - 1:
                _requested = os.path.exists(_marker)
                if jax.process_count() > 1:
                    from jax.experimental import multihost_utils
                    _requested = bool(multihost_utils.broadcast_one_to_all(np.array(_requested)))
                if _requested:
                    checkpoint_manager.wait_until_finished()
                    try:
                        _by = open(_marker).read().strip()
                    except OSError:
                        _by = "?"
                    logging.info("handoff requested by job %s: checkpoint %d is complete, exiting for the successor", _by, step)
                    break

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())