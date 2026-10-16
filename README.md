# Team Dexterity — π0.5 for the 2026 BEHAVIOR Challenge

Training code and a step-by-step guide for fine-tuning **π0.5** ([openpi](https://github.com/Physical-Intelligence/openpi), JAX)
on the **2026 BEHAVIOR Challenge** demonstrations: 100 long-horizon household tasks with the Galaxea R1 Pro robot in
OmniGibson. Everything here is written to be run on *your* hardware — paths, GPU count and batch size are settings.

**Self-evaluation (public test set, official evaluator): overall score 0.339** — 853 rollouts, 154 full successes;
79 of 100 tasks scored on at least one instance (task success with partial credit, mean over 100 tasks × 10 instances,
instances not run counted as 0).

| release | link |
|---|---|
| Training code + guide | this repository |
| Trained model (step 135,199) | 🤗 [`VivekChandra/pi05-behavior-2026-dexterity`](https://huggingface.co/VivekChandra/pi05-behavior-2026-dexterity) |
| Ready-to-run policy server | `ghcr.io/vivekchandra1324/b1k-pi05-policy:135199` (Docker, one 24 GB GPU) |
| Self-evaluation results + all rollout videos | 🤗 [`VivekChandra/b1k2026-rollout-videos`](https://huggingface.co/datasets/VivekChandra/b1k2026-rollout-videos) |

## Requirements

* **GPUs:** at least **2× 80 GB** — tested on 2× H100 with our full configuration (EMA on): training,
  checkpointing and resuming at 8 samples per GPU, and 32 per GPU fits (51.9 GiB peak). One 80 GB GPU works only
  with EMA turned off. We trained on 8× H100.
* **RAM:** ~192 GB per machine (the 2-GPU test peaked at 147 GB). **Disk:** ~1 TB for the dataset + ~0.6 TB for the
  224 copy + ~25 GB per checkpoint. Linux, CUDA 12 driver, `ffmpeg`, `uv`.

## Quick start

```bash
git clone https://github.com/VivekChandra1324/behavior-2026-dexterity.git && cd behavior-2026-dexterity
bash scripts/setup_openpi.sh ../openpi-b1k && (cd ../openpi-b1k && GIT_LFS_SKIP_SMUDGE=1 uv sync)   # code
bash scripts/download_dataset.sh /data/2026-challenge-demos                                         # ~1 TB
python scripts/transcode_rgb224.py --src /data/2026-challenge-demos --dst /data/2026-challenge-rgb224
python scripts/verify_rgb224.py   --src /data/2026-challenge-demos --dst /data/2026-challenge-rgb224
# normalisation statistics + stage table: see docs/TRAINING_GUIDE.md §4
# edit configs/train.env (paths, GPUs, batch), then:
bash scripts/train.sh ../openpi-b1k
```

**Read [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** — every step with timings, how to size the run for your
GPUs (memory, batch, learning rate, data loader), multi-machine training, what mattered and troubleshooting.

## What we did

* **Model:** π0.5 (PaliGemma VLM + flow-matching action expert, 3.35 B parameters) fine-tuned from `pi05_base` on
  all 100 tasks at once. Inputs: three RGB cameras at 224×224 + proprioception; no depth.
* **Data:** the official 20,000 demonstrations, re-encoded to 224×224 RGB **without B-frames** — the change that
  made the data loader fast enough to feed the GPUs (25 → 82 samples/s).
* **Recipe** (each measured against an alternative): 15 flow-matching samples per example, quantile-normalised
  state and z-scored actions, task sampling ∝ frames^0.5, an auxiliary temporal-stage head, vision tower trained,
  warmup-stable-decay schedule at peak LR 6.4e-5.
* **Scale:** global batch 832 on 8× H100 for 135,200 steps (112.5 M samples, ≈0.53 epochs, ~2,400 H100-hours).
* **Inference** (in the Docker image, not in this repo): replanning every 20 steps from a compressed 26-step action
  chunk, and a gripper re-open rule derived from the demonstrations.

## Repository layout

```
configs/train.env              the exact training configuration of our final run, annotated
scripts/setup_openpi.sh        openpi fork at the pinned commit + our overlay
scripts/download_dataset.sh    official dataset (RGB + low-dim; depth optional)
scripts/transcode_rgb224.py    build the 224x224, B-frame-free training dataset
scripts/verify_rgb224.py       check it before training
scripts/compute_norm_stats.py  normalisation statistics (without decoding video)
scripts/compute_stage_table.py per-task stage counts for the auxiliary stage head
scripts/train.sh               train / resume
openpi/overlay/                our modified openpi files;  openpi/CHANGES.md: what and why
docs/TRAINING_GUIDE.md         the guide
```

## License

Code: **Apache-2.0** ([LICENSE](LICENSE)); the files in `openpi/overlay` are modified from openpi (Apache-2.0) and
marked as such. **Model weights:** π0.5's backbone is PaliGemma, a Gemma model, so the released weights are a Gemma
Model Derivative, provided under and subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
(copy in [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt)), including its prohibited-use policy. See [NOTICE](NOTICE).

## Acknowledgements

Built on [openpi](https://github.com/Physical-Intelligence/openpi) by Physical Intelligence, the BEHAVIOR fork
[`wensi-ai/openpi`](https://github.com/wensi-ai/openpi), [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) and
the 2026 BEHAVIOR Challenge dataset by the Stanford Vision and Learning Lab. Several recipe choices follow the
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
