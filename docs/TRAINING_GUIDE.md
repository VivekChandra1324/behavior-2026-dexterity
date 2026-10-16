# Training π0.5 on BEHAVIOR-1K (2026 challenge) — step-by-step guide

This guide reproduces the training of our submitted checkpoint, and shows how to adapt it to **your** hardware.
It assumes a Linux machine with NVIDIA GPUs, `git`, `ffmpeg`/`ffprobe` and [`uv`](https://docs.astral.sh/uv/).

| step | what | time (our hardware) |
|---|---|---|
| 1 | Set up the training code | ~15 min |
| 2 | Download the dataset | hours (≈1 TB) |
| 3 | Build the 224×224 training dataset | 8–12 h on ~48–96 CPU cores |
| 4 | Normalisation statistics + stage table | ~1 h 40 min + seconds |
| 5 | Size the run for your GPUs | — |
| 6 | Train | ~11 days on 8× H100 for our full run |

---

## 1. Set up the training code

We train with Physical Intelligence's [openpi](https://github.com/Physical-Intelligence/openpi) (JAX) through the
BEHAVIOR fork [`wensi-ai/openpi`](https://github.com/wensi-ai/openpi), plus our modifications (`openpi/overlay`,
listed in [`openpi/CHANGES.md`](../openpi/CHANGES.md)).

```bash
git clone https://github.com/VivekChandra1324/behavior-2026-dexterity.git
cd behavior-2026-dexterity
bash scripts/setup_openpi.sh ../openpi-b1k          # fork at the pinned commit + our overlay
cd ../openpi-b1k && GIT_LFS_SKIP_SMUDGE=1 uv sync    # Python 3.11, JAX 0.5.3 (CUDA 12), the lerobot fork, ...
uv run python -c "import jax; print(jax.devices())" # must list every GPU
```

The π0.5 base weights (`gs://openpi-assets/checkpoints/pi05_base`) are downloaded automatically on the first run
into `$OPENPI_DATA_HOME` (default `~/.cache/openpi`, ~12 GB); point that variable at a large disk if needed.

## 2. Download the dataset

The official data is [`behavior-1k/2026-challenge-demos`](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos)
(LeRobot v3): **20,000 teleoperated episodes, 100 tasks × 200, 30 fps, 211 M frames**. Each frame has three RGB
cameras (head 720×720, two wrists 480×480), three depth streams, a 61-dim proprioceptive state and a 23-dim
action (absolute joint targets for base, torso, both arms and grippers).

```bash
pip install "huggingface_hub[cli,hf_xet]"
bash scripts/download_dataset.sh /data/2026-challenge-demos          # RGB + low-dim, ~1 TB
```

**Depth is skipped on purpose:** it is 2.1 of the 3.3 TB and π0.5 does not read it (neither does the other
official baseline). Add `INCLUDE_DEPTH=1` only if you want to experiment with it.

## 3. Build the 224×224 training dataset — do not skip this

```bash
python scripts/transcode_rgb224.py --src /data/2026-challenge-demos --dst /data/2026-challenge-rgb224 --workers 48
python scripts/verify_rgb224.py   --src /data/2026-challenge-demos --dst /data/2026-challenge-rgb224
```

This re-encodes the 3 × 1,762 RGB videos to 224×224 (HEVC, CRF 28, keyframe every 8 frames, **no B-frames**) and
symlinks the unchanged low-dimensional data. It is resumable — re-run the same command after any interruption —
and `verify_rgb224.py` must print `0 problems` (identical frame counts, 224×224, no B-frames) before you train.

Why it matters — this was the largest speed-up of the whole project:
* π0.5 sees 224×224 images anyway (openpi resizes every frame before augmentation), so storing 224 is lossless
  for the model's input pipeline except for the re-encode itself, and it cuts the bytes read per sample ~10×.
* **B-frames make random access ~270× more expensive.** With B-frames the decoder's cheap seek returns wrong
  frames, so the loader must use the exact seek, which scans the file's packet index on every open: 0.82 s per
  open vs 0.003 s without. With three cameras per sample our loader fell to ~37 samples/s while the GPUs needed
  ~88. Re-encoding without B-frames fixed it (the loader went from 25 to 82 samples/s on the same machine).

## 4. Normalisation statistics and stage table

```bash
cd ../openpi-b1k
uv run python ../behavior-2026-dexterity/scripts/compute_norm_stats.py \
    --dataset-root /data/2026-challenge-rgb224 --repo-id behavior-1k/2026-challenge-rgb224 \
    --assets-base-dir $PWD/assets --no-video
uv run python ../behavior-2026-dexterity/scripts/compute_stage_table.py --root /data/2026-challenge-rgb224 \
    --out $PWD/assets/pi05_b1k/behavior-1k/2026-challenge-rgb224/stage_table.json
```

* `--no-video` computes the statistics without decoding any frame (they only use state and actions). Without it
  the loader decodes three videos per sample for nothing, and per-worker memory grew without bound. On the full
  root it took ~1 h 40 min in one process at ~14 GB of RAM. `--max-frames N` gives a quicker approximation.
* The stage table holds, per task, how many temporal stages to predict (`clip(round(mean length / 900), 5, 15)`)
  and every episode's length. It needs only the dataset metadata and takes seconds.
* Both files must be in `<assets>/pi05_b1k/<repo_id>/`; `train.sh` checks for them.

## 5. Size the run for your GPUs

Edit the "your hardware" and "schedule" blocks of [`configs/train.env`](../configs/train.env). Our run:

| | our final run |
|---|---|
| GPUs | 8× H100 80 GB (2 nodes × 4) |
| global batch | 832 (104 per GPU) |
| FSDP | 4 (parameters and optimizer state sharded inside each node, data-parallel across nodes) |
| peak LR / warmup / schedule | 6.4e-5 / 1,000 steps / warmup-stable-decay, decay over the last 15 % |
| steps | 135,200 → 112.5 M samples ≈ 0.53 epochs |
| speed | ~6.75 s/step ≈ 123 samples/s (≈ 15 samples/s per GPU) |
| wall time | ~11 days ≈ 2,400 H100-hours including start-ups and checkpoints |

**GPU memory (measured).** The model has 3.35 B parameters. Its training state, measured on one GPU: weights
12.5 GiB + Adam optimizer 25.0 GiB + EMA copy 12.5 GiB = **50 GiB**, plus ~25 GiB of temporaries for a step (gradients
and buffers, almost independent of the batch), plus activations that grow with the batch. FSDP divides the
*state* across `FSDP_DEVICES` GPUs (~12.5 GiB per GPU at FSDP 4). So:
* **80 GB GPUs, 4 or more per machine (H100/A100):** FSDP 4; we ran 104 samples per GPU. On one machine with
  4 GPUs keep that per-GPU batch (global 416) or less.
* **2× 80 GB GPUs (tested end to end):** `FSDP_DEVICES=2`, EMA on, our full configuration — the minimum we
  verified: training, checkpointing and resuming all work. At 8 samples per GPU the state is 25 GiB per GPU and a
  step peaks at 44.5 GiB. Measured on 2× H100 80 GB, EMA on:

  | per-GPU batch (global) | peak GPU memory | |
  |---|---|---|
  | 8 (16) | 44.5 GiB | fits — trained, saved, resumed |
  | **32 (64)** | **51.9 GiB** | **fits — recommended for 2 GPUs** |
  | 64 (128) | > 80 GiB | out of memory |

  With 2 GPUs use `BATCH_SIZE=64` and `PEAK_LR≈1.8e-5` (√-scaled). Two GPUs give at most a quarter of our 8-GPU
  throughput, so our full 112.5 M-sample budget would take well over a month: plan a smaller sample budget
  (`TOTAL_STEPS`) or more GPUs.
* **40–48 GB GPUs:** shard over all GPUs of the machine (`FSDP_DEVICES` = GPUs per node, at least 4) and lower the
  per-GPU batch until it fits; start around a quarter of ours (~24 per GPU) and raise it.
* **One 80 GB GPU:** it does **not** fit as configured (state + temporaries ≈ 75 GiB before any activations; we
  measured an out-of-memory error even at batch 2). It fits with the EMA copy turned off — add `--ema-decay None`
  to the `train.sh` command — at a small batch: batch 2 peaked at 62 GiB. Without EMA you train the raw weights
  (we used EMA 0.99); expect it to be slow — this is a multi-GPU model.

**Host memory (RAM).** Saving a checkpoint copies the training state into host memory, and every data-loader
worker holds open video decoders. A single-GPU test was killed by the out-of-memory killer at its first checkpoint
save with 64 GB of RAM (54 GB in use); the 2-GPU test peaked at 147 GB with 8 loader workers; our training
machines had 512 GB. Plan for **at least ~192 GB per machine** (more with more loader workers), and lower
`NUM_WORKERS` / `B1K_DECODER_CACHE` if RAM is tight.

If a configuration does not fit, the run fails at the first step with an out-of-memory error; lower the per-GPU
batch. `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` (set by `train.sh`) lets JAX use almost all GPU memory.

**Batch, learning rate and steps.** Keep the *number of samples*, not the number of steps:
`TOTAL_STEPS ≈ 112.5 M / BATCH_SIZE` reproduces our data budget (fewer if your budget is smaller — our checkpoint
saw only half an epoch, and more data would very likely help). When you change the batch, scale the peak LR with
the square root of the batch: `PEAK_LR ≈ 6.4e-5 × sqrt(BATCH_SIZE / 832)`. We picked 6.4e-5 with a short LR
sweep at batch 832 (the best arm, then one √2 step lower for safety on the long run).

**Data loader.** The GPUs consume ~15 samples/s each; the loader must keep up or the GPUs idle. 16 workers per node
with `B1K_DECODER_CACHE=16` delivered 93 % of what 4 H100s consume on our 96-core nodes. More workers are not
always better: 24 workers × 64 cached decoders used 236 GiB of RAM and throughput *collapsed* to 35 samples/s.
Watch host RAM and the data-wait fraction (`B1K_LOG_DATAWAIT=1` logs it) when you tune `NUM_WORKERS`.

## 6. Train

```bash
cd behavior-2026-dexterity
bash scripts/train.sh ../openpi-b1k                     # or: TRAIN_ENV=my.env bash scripts/train.sh ../openpi-b1k
```

* **Start-up takes a few minutes** before step 0: building the sampling index over 211 M frames (~3 min),
  loading the 12.5 GB base weights and compiling the training step (~1–2 min with a cold compile cache).
* Checkpoints go to `<CKPT_DIR>/pi05_b1k/<EXP_NAME>/<step>/` every `SAVE_INTERVAL` steps.
* **Resuming:** run the same command again. `train.sh` resumes from the newest checkpoint when one exists and
  starts fresh only when none does — so a crash, reboot or job time limit costs at most `SAVE_INTERVAL` steps.
  If a process was killed *while* saving, delete that half-written step directory (it lacks a complete `params/`)
  before resuming. Never change `B1K_STAGE_AUX`, `B1K_FREEZE_VISION` or the normalisation mid-run: the checkpoint
  would no longer match the model.
* **Logging:** W&B runs offline by default (`WANDB_MODE=online` to stream). Useful extra switches:
  `B1K_LOG_MEM=1` (real per-GPU memory), `B1K_LOG_DATAWAIT=1` (fraction of time waiting for data).
* **What healthy looks like:** the loss falls fast in the first few thousand steps and then slowly; GPU utilisation
  stays high and data-wait low. Step time much slower than expected almost always means the data loader is
  starved (check §3 and the worker count) — not the GPUs.
* **Checkpoint for inference:** use the final step's directory (it holds `params/` and the `assets/` with the normalisation statistics).

### More than one machine

`train_b1k.py` supports multi-host training (`B1K_MULTIHOST=1`). Start **one process per machine**, each seeing
all of its GPUs, with:

```bash
export B1K_MULTIHOST=1 B1K_COORDINATOR=<first-machine-ip>:29600 B1K_LOCAL_DEVICES=<GPUs per machine>
# plus SLURM_NTASKS=<machines> and SLURM_PROCID=<this machine's index, 0..N-1> (set automatically under Slurm)
```

Keep `FSDP_DEVICES` = GPUs per machine (shard inside a machine, replicate across machines): sharding across the
network saturated our InfiniBand and bought nothing (0.98×). Two machines paid off only at a large per-GPU batch:
the speed-up was 0.82× at 16 samples/GPU, 1.13× at 32 and **1.44× at 64**, because the cross-machine traffic is
fixed by the parameter count (~18 GB per step) while compute grows with the batch. Make sure the coordinator
address is on an interface the machines can reach each other through.

## 7. What mattered (measured, not assumed)

| decision | result |
|---|---|
| 224×224 RGB, re-encoded **without B-frames** | loader 25 → 82 samples/s; the difference between starving and feeding the GPUs |
| depth not used | 2.1 TB less to store and read; π0.5 is RGB-pretrained |
| 15 flow-matching samples per example | −8.3 % held-out action error (2 seeds) |
| quantile normalisation for the state | state values saturating the range: 28.6 % (z-score) → 5.0 % |
| z-score normalisation for the actions | a per-step variant was 4.7 % worse on the first 8 steps |
| task sampling ∝ frames^0.5 | long tasks are not over-sampled relative to short ones |
| vision tower trained, not frozen | freezing was 17 % faster but 4 % worse on held-out error |
| stage head (auxiliary) | no measurable difference either way; kept on |
| batch 832, LR 6.4e-5, WSD | from a short LR sweep at the final batch |

**The prompt is the task name.** The data's language instruction is the task's snake_case name
(`turning_on_radio`, …) because training reads it from the dataset. Use exactly that string at inference.

## 8. Using the trained model

Our trained checkpoint (step 135,199) is on Hugging Face (link in the README; Gemma Terms of Use apply), and a
ready-to-run policy server with it is published as a Docker image (README). To serve a checkpoint you trained
yourself, use the BEHAVIOR fork's `scripts/b1k/serve_b1k.py` with your checkpoint directory and the same
normalisation settings you trained with (`B1K_NORM_STATE`, `B1K_NORM_ACTIONS`, `B1K_STAGE_AUX`).

## Troubleshooting

| symptom | likely cause | fix |
|---|---|---|
| samples/s far below ~15 per GPU, GPUs idle | loader starved: B-frames, slow disk, too few/many workers | §3 verify, §5 data loader |
| host RAM keeps climbing, workers killed | too many workers × cached decoders | lower `NUM_WORKERS` / `B1K_DECODER_CACHE` |
| out of memory at step 1 | per-GPU batch too large for your GPU | lower `BATCH_SIZE`, raise `FSDP_DEVICES`, or (one GPU) `--ema-decay None` |
| process killed while saving a checkpoint | host RAM exhausted | more RAM, fewer `NUM_WORKERS`, smaller `B1K_DECODER_CACHE` |
| `jax.devices()` shows fewer GPUs than you have | CUDA/driver or `CUDA_VISIBLE_DEVICES` | fix before training: FSDP would silently use fewer |
| norm stats run out of memory | video decoded for nothing | use `--no-video` |
| resume starts again at step 0 | no complete checkpoint in the run directory | check `<CKPT_DIR>/pi05_b1k/<EXP_NAME>/<step>/params` |
