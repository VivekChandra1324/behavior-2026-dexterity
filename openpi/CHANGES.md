# Changes to openpi

`overlay/` holds complete copies of the files we modified in the BEHAVIOR fork of openpi
([`wensi-ai/openpi`](https://github.com/wensi-ai/openpi), branch `behavior`), at commit
`0cc8e355f7bac0976db1cc3139b1ff0379feea60` (`BASE_COMMIT`). `scripts/setup_openpi.sh` checks out that commit and
copies the overlay on top. Every modified or added file carries a one-line notice, and every change is explained in a
comment at the change. Training changes are opt-in through a `B1K_*` environment variable or keep the upstream
behaviour by default; serving options are off unless given. Two serving changes are always on because the current
evaluator needs them: the `robot_r1` observation keys and the disabled server keepalive ping.
Tags such as `(T41)` or `(T65)` in those comments are entry numbers in our internal engineering log; the comment
next to each tag states the finding itself.

openpi is licensed under Apache-2.0 (see `LICENSE` at the repository root); these files remain under that license.

| file | what changed | switch |
|---|---|---|
| `scripts/b1k/train_b1k.py` | Multi-host start-up (`jax.distributed.initialize` with explicit local GPU ids; guards against a silent single-process or single-GPU run); JAX compilation cache location; the LR schedule actually in use is logged; optional memory, data-wait and MFU logging; clean stop at a checkpoint on request | `B1K_MULTIHOST`, `B1K_LOCAL_DEVICES`, `B1K_COORDINATOR`, `B1K_LOG_MEM`, `B1K_LOG_DATAWAIT`, `B1K_TFLOP_PER_SAMPLE`, `B1K_HANDOFF_MARKER` |
| `src/openpi/training/data_loader.py` | Video seek mode chosen by probing the data (the cheap seek is only correct without B-frames); an LRU-bounded decoder cache (bounded RAM per worker); out-of-order batch hand-over; task-balanced sampling (task drawn with probability ∝ frames^α); retries for transient read errors; optional depth skipping; per-process sampler for multi-host runs | `B1K_VIDEO_SEEK_MODE`, `B1K_DECODER_CACHE`, `B1K_LOADER_IN_ORDER`, `B1K_TASK_ALPHA`, `B1K_GETITEM_RETRIES`, `B1K_SKIP_DEPTH` |
| `src/openpi/training/config.py` | `pi05_b1k` reads the dataset root and repo id from the environment (the fork hard-codes a path and suppresses the CLI flags); timestamp tolerance for re-encoded video | `B1K_DATASET_ROOT`, `B1K_REPO_ID`, `B1K_TOLERANCE_S` |
| `src/openpi/training/optimizer.py` | Warmup-stable-decay (WSD) learning-rate schedule | `B1K_WSD="<stable_end>,<total>"` |
| `src/openpi/training/checkpoints.py` | `--overwrite` safe when several processes start at once (one owner + a barrier); multi-host saves without orbax's replica-parallel slicing, which aborted a 2-node save with a device-placement error | `B1K_ORBAX_REPLICA_PARALLEL` |
| `src/openpi/training/weight_loaders.py` | Loading the base weights into a model with an extra (stage) head | `B1K_STAGE_AUX` |
| `src/openpi/models/pi0.py` | Multi-sample flow matching (N noise/time samples per example, the 2025 winner's trick); optional correlated action noise; auxiliary temporal-stage head; optional frozen vision tower; selectable rematerialisation policy | `B1K_NUM_FLOW_SAMPLES`, `B1K_NOISE_COV`, `B1K_CORRELATED_NOISE`, `B1K_STAGE_AUX`, `B1K_STAGE_LOSS_FIX`, `B1K_FREEZE_VISION`, `B1K_REMAT_POLICY` |
| `src/openpi/models/model.py`, `src/openpi/models/gemma.py` | Plumbing for the above (stage label in the observation; remat policy) | — |
| `src/openpi/policies/b1k_policy.py` | Temporal stage label for the auxiliary head, computed from the frame index and a per-task stage count | `B1K_STAGE_AUX`, `B1K_STAGE_TABLE` |
| `src/openpi/transforms.py` | Per-key choice of normalisation (quantile or z-score) for state and actions | `B1K_NORM_STATE`, `B1K_NORM_ACTIONS`, `B1K_QUANTILE_NORM`, `B1K_ACTION_PERSTEP` |
| `scripts/b1k/serve_b1k.py` | Serving entry point: one server for all 100 tasks (`--task b1k/auto`, the prompt follows each request's `task_id`); action compression and the gripper re-open rule as options | `--task b1k/auto`, `--compress_from`, `--compress_to`, `--gripper_table` |
| `src/openpi/serving/websocket_b1k_server.py` | Maps the evaluator's `task_id` to the task name; the evaluator's optional action-chunk replay protocol (`--replay-action-chunk-size K`); no server keepalive pings (the default ones dropped heavy-scene rollouts; the official client runs its own); optional timing log | `B1K_LOG_INFER` |
| `src/openpi/shared/eval_b1k_wrapper.py` | Receding-horizon execution with chunk replay (`act_chunk`, replans when a chunk would cross a replan step); action compression (cubic spline of the first 26 predicted steps onto 20, base velocity scaled, off through a grasp); hook for the gripper rule; per-request task switching | — |
| `src/openpi/shared/b1k_correction_rules.py` (new) | The gripper re-open rule (2025 winning solution), reading the per-task, per-stage table `serving/gripper_table.json` | — |
| `src/openpi/shared/eval_b1k_wrapper_test.py` (new) | Tests (pytest): chunk replay gives the same actions as per-step requests, with and without compression; a chunk that would cross a replan step replans; task switching resets the plan; evaluator `task_id` mapping | — |
| `src/openpi/configs/robots/b1k.py` | R1Pro observation keys renamed `robot` → `robot_r1`, the robot name used by the BEHAVIOR-1K v3.9.3 evaluator (serving only; training reads `dataset_key`) | — |
| `src/openpi/configs/tasks/b1k_task_ids.json` (new) | Evaluator `task_id` → task name for all 100 tasks | — |
| `src/openpi/configs/tasks/b1k.py` | Prompts for all 100 tasks. **The prompt is the task's snake_case name** (e.g. `turning_on_radio`), because `prompt_from_task=True` reads the task names stored in the dataset; any prompt you use at inference must match | — |

The overlay contains every file the published Docker image changes: the image is built from the same base commit
with the same changes. `scripts/serve.sh` runs the image's entrypoint command on a source checkout.
