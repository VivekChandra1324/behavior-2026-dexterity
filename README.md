# Team Dexterity — π0.5 for the 2026 BEHAVIOR Challenge

[![Model on HF](https://img.shields.io/badge/🤗%20model-pi05--behavior--2026--dexterity-yellow)](https://huggingface.co/VivekChandra/pi05-behavior-2026-dexterity)
[![Results & videos](https://img.shields.io/badge/🤗%20results%20%26%20videos-b1k2026--rollout--videos-yellow)](https://huggingface.co/datasets/VivekChandra/b1k2026-rollout-videos)
[![License: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue)](LICENSE)
[![Weights: Gemma terms](https://img.shields.io/badge/weights-Gemma%20Terms%20of%20Use-orange)](LICENSE_GEMMA.txt)

Code and a step-by-step guide to fine-tune **π0.5** ([openpi](https://github.com/Physical-Intelligence/openpi), JAX)
on the official demonstrations of the **[2026 BEHAVIOR Challenge](https://behavior.stanford.edu/challenge/index.html)**:
100 long-horizon household tasks performed by a Galaxea R1 Pro robot in the OmniGibson simulator. It is written to
run on *your* hardware — paths, number of GPUs and batch size are settings — and was tested end to end from a fresh
clone on 2× H100.

**Result — self-evaluated overall score 0.339** on the public test set with the official evaluator: 853 rollouts,
154 full successes, 79 of 100 tasks scoring. Per-task table: [docs/RESULTS.md](docs/RESULTS.md).

## Contents

* [Release](#release) · [Requirements](#requirements) · [Quick start](#quick-start) · [What we did](#what-we-did)
* [Repository layout](#repository-layout) · [Using our trained model](#using-our-trained-model) · [FAQ](#faq)
* [License](#license) · [Acknowledgements](#acknowledgements) · [Citation](#citation)
* Full guide: **[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** — every step, timings, sizing for your GPUs,
  multi-machine training, what mattered, troubleshooting.

## Release

| what | where |
|---|---|
| Training code + guide | this repository |
| Trained model, step 135,199 (12.4 GB, orbax) | 🤗 [VivekChandra/pi05-behavior-2026-dexterity](https://huggingface.co/VivekChandra/pi05-behavior-2026-dexterity) |
| Ready-to-run policy server (all 100 tasks, one 24 GB GPU) | [`ghcr.io/vivekchandra1324/b1k-pi05-policy:135199`](https://github.com/users/VivekChandra1324/packages/container/package/b1k-pi05-policy) |
| Self-evaluation: 853 result JSONs + evaluation README | 🤗 [submission.zip](https://huggingface.co/datasets/VivekChandra/b1k2026-rollout-videos/resolve/main/submission.zip) · [README](https://huggingface.co/datasets/VivekChandra/b1k2026-rollout-videos/blob/main/README.md) |
| All 853 rollout videos | 🤗 [VivekChandra/b1k2026-rollout-videos/videos](https://huggingface.co/datasets/VivekChandra/b1k2026-rollout-videos/tree/main/videos) |

Related: [challenge website](https://behavior.stanford.edu/challenge/index.html) ·
[leaderboard](https://behavior-1k-2026-challenge-leaderboard.hf.space) ·
[official dataset](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos) ·
[BEHAVIOR-1K / OmniGibson](https://github.com/StanfordVL/BEHAVIOR-1K) · [openpi](https://github.com/Physical-Intelligence/openpi)

## Requirements

| | |
|---|---|
| GPUs | **at least 2× 80 GB** (tested on 2× H100 with our full configuration); we trained on 8× H100. One 80 GB GPU works only with the EMA copy turned off. Details: [guide §5](docs/TRAINING_GUIDE.md#5-size-the-run-for-your-gpus) |
| RAM | ~192 GB per machine (the 2-GPU test peaked at 147 GB) |
| Disk | ~2 TB: dataset 1.2 TB + 224 copy 0.57 TB + ~190 GB of caches in `$WORK/cache` + ~25 GB per checkpoint (the 1.1 TB of original videos can be deleted after step 3). Nothing large is written to `$HOME` |
| Software | Linux, NVIDIA driver for CUDA 12, git, ffmpeg with libx265, Python ≥ 3.9, [uv](https://docs.astral.sh/uv/), the `hf` CLI |
| Accounts | none — the dataset, base weights and this repository are public |

## Quick start

```bash
# 0. prerequisites (details: guide §0)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install "huggingface_hub[cli,hf_xet]"                  # plus git and ffmpeg (with libx265) from your OS
export WORK=/path/to/a/large/disk/b1k && mkdir -p $WORK/data    # everything lives under $WORK
export UV_CACHE_DIR=$WORK/cache/uv                              # keeps uv's ~13 GB package cache off $HOME

# 1. code: openpi (BEHAVIOR fork) at the pinned commit + our changes
cd $WORK && git clone https://github.com/VivekChandra1324/behavior-2026-dexterity.git && cd behavior-2026-dexterity
bash scripts/setup_openpi.sh $WORK/openpi-b1k
(cd $WORK/openpi-b1k && GIT_LFS_SKIP_SMUDGE=1 uv sync && uv run python -c "import jax; print(jax.devices())")

# 2. data (1.2 TB) and 3. the 224x224 training copy without B-frames (8-12 h on ~48 cores)
bash scripts/download_dataset.sh $WORK/data/2026-challenge-demos
python3 scripts/transcode_rgb224.py --src $WORK/data/2026-challenge-demos --dst $WORK/data/2026-challenge-rgb224 --workers 48
python3 scripts/verify_rgb224.py   --src $WORK/data/2026-challenge-demos --dst $WORK/data/2026-challenge-rgb224   # must report 0 problems

# 4. normalisation statistics + stage table (~1 h 40 min)
bash scripts/prepare_assets.sh $WORK/openpi-b1k

# 5. set your GPU count / batch in configs/train.env (guide §5), then 6. train -- re-run the same command to resume
bash scripts/train.sh $WORK/openpi-b1k
```

For 2 GPUs, set in `configs/train.env`: `BATCH_SIZE=64`, `FSDP_DEVICES=2`, `PEAK_LR=1.8e-5`, `NUM_WORKERS=8` and a
`TOTAL_STEPS` that fits your time budget.

## What we did

* **Model:** π0.5 (PaliGemma vision-language backbone + flow-matching action expert, 3.35 B parameters), fine-tuned
  from `pi05_base` on all 100 tasks at once. Inputs: three RGB cameras at 224×224 + proprioception; no depth.
  The language prompt is the task's snake_case name (e.g. `turning_on_radio`).
* **Data:** the official 20,000 demonstrations, re-encoded to 224×224 RGB **without B-frames** — the change that
  made the data loader fast enough to feed the GPUs (25 → 82 samples/s).
* **Recipe** (each choice measured against an alternative — see the guide §7): 15 flow-matching samples per example,
  quantile-normalised state and z-scored actions, task sampling ∝ frames^0.5, an auxiliary temporal-stage head,
  vision tower trained, warmup-stable-decay schedule with peak LR 6.4e-5.
* **Scale:** global batch 832 on 8× H100 for 135,200 steps (112.5 M samples ≈ 0.53 epochs, ~2,400 H100-hours).
* **Inference** (in the Docker image): replanning every 20 steps from a compressed 26-step action chunk, and a
  per-task gripper re-open rule derived from the demonstrations.

## Repository layout

```
configs/train.env               exact configuration of our final run, every switch explained
scripts/setup_openpi.sh         openpi (BEHAVIOR fork) at the pinned commit + our overlay
scripts/download_dataset.sh     official dataset from Hugging Face (RGB + low-dim; depth optional)
scripts/transcode_rgb224.py     build the 224x224, B-frame-free training dataset
scripts/verify_rgb224.py        check it before training
scripts/prepare_assets.sh       normalisation statistics + stage table, where train.sh expects them
scripts/compute_norm_stats.py   (used by prepare_assets.sh) statistics without decoding video
scripts/compute_stage_table.py  (used by prepare_assets.sh) per-task stage counts for the stage head
scripts/train.sh                train, or resume from the newest checkpoint
openpi/overlay/                 our modified openpi files    openpi/CHANGES.md   what changed and why
docs/TRAINING_GUIDE.md          the step-by-step guide       docs/RESULTS.md     per-task results
```

## Using our trained model

* **Policy server (simplest):** `docker run --gpus all -p 8000:8000 ghcr.io/vivekchandra1324/b1k-pi05-policy:135199`
  serves all 100 tasks over the BEHAVIOR WebSocket policy protocol on port 8000, offline, on one 24 GB GPU. Evaluate
  it with the official evaluator from [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) (see the
  [evaluation README](https://huggingface.co/datasets/VivekChandra/b1k2026-rollout-videos/blob/main/README.md) for
  the exact command we used).
* **Weights:** [VivekChandra/pi05-behavior-2026-dexterity](https://huggingface.co/VivekChandra/pi05-behavior-2026-dexterity)
  — orbax checkpoint (`checkpoint/params`, `checkpoint/assets`) for openpi's `pi05_b1k` config; the model card
  lists the settings that must match training. Also a good starting point for further fine-tuning.

## FAQ

**Can I train on fewer or smaller GPUs?** Yes, with a smaller batch and more steps — see the measured memory table
in [guide §5](docs/TRAINING_GUIDE.md#5-size-the-run-for-your-gpus). 2× 80 GB is the smallest setup we tested with
the full configuration; one 80 GB GPU needs `--ema-decay None`.

**How long does it take?** Our full run: ~11 days on 8× H100. Throughput scales roughly with GPU count, so plan
your sample budget (`TOTAL_STEPS × BATCH_SIZE`) to your time; we used 112.5 M samples (half an epoch).

**Do I need the depth videos?** No. π0.5 does not read them; skipping them saves 2.1 TB.

**Can I skip the 224×224 transcode?** You can, but training will be several times slower: the loader cannot keep
up with the GPUs on the original videos (B-frames, 10× the bytes). See [guide §3](docs/TRAINING_GUIDE.md#3-build-the-224224-training-dataset--do-not-skip-this).

**Training crashed / my job hit its time limit.** Run the same `train.sh` command again; it resumes from the newest
complete checkpoint.

**Something else is wrong.** Check [Troubleshooting](docs/TRAINING_GUIDE.md#troubleshooting), then open an
[issue](https://github.com/VivekChandra1324/behavior-2026-dexterity/issues) with your command, your
`configs/train.env` changes and the last lines of the log.

## License

* **Code:** [Apache-2.0](LICENSE). The files in `openpi/overlay` are modified from openpi (Apache-2.0) and carry a
  notice; changes are listed in [openpi/CHANGES.md](openpi/CHANGES.md). See [NOTICE](NOTICE).
* **Model weights:** π0.5's backbone is PaliGemma, a Gemma model, so our weights are a Gemma Model Derivative:
  Gemma is provided under and subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) (copy in
  [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt)), including the [prohibited use policy](https://ai.google.dev/gemma/prohibited_use_policy).
* **Data:** the BEHAVIOR Challenge dataset is distributed by its authors under its own license (MIT); it is not
  redistributed here.

## Acknowledgements

Built on [openpi](https://github.com/Physical-Intelligence/openpi) by Physical Intelligence, the BEHAVIOR fork
[`wensi-ai/openpi`](https://github.com/wensi-ai/openpi), [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K)
and the 2026 BEHAVIOR Challenge dataset by the Stanford Vision and Learning Lab. Several recipe choices follow the
published solutions of the 2025 BEHAVIOR Challenge.

## Citation

```bibtex
@misc{dexterity2026behavior,
  title  = {Team Dexterity: pi0.5 for the 2026 BEHAVIOR Challenge},
  author = {Vivek Chandra},
  year   = {2026},
  url    = {https://github.com/VivekChandra1324/behavior-2026-dexterity}
}
```
