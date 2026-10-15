# Modified for the 2026 BEHAVIOR Challenge (team Dexterity) -- see openpi/CHANGES.md in the training repository.
import dataclasses
import os
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at


@runtime_checkable
class LRScheduleConfig(Protocol):
    def create(self) -> optax.Schedule: ...


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6

    def create(self) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )


@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Inverse square root decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000

    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                lambda step: self.peak_lr / jnp.sqrt((self.timescale + step) / self.timescale),
            ],
            [self.warmup_steps],
        )


@runtime_checkable
class OptimizerConfig(Protocol):
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation: ...


@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    # Changing this to 0 can cause out-of-memory errors for some reason, so we set it to a negligible value.
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        tx = optax.adamw(
            lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
        )

        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)


@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        assert weight_decay_mask is None, "Weight decay is not supported for SGD"
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)


def wsd_from_env(lr_schedule: LRScheduleConfig) -> optax.Schedule | None:
    """B1K addition (opt-in: B1K_WSD="<stable_end>,<total>"): warmup-stable-decay. The same linear
    warm-up as the configured schedule (its `warmup_steps`, `peak_lr`), then the peak held constant through step
    `stable_end`, then a (1 - sqrt) cool-down to 0 at `total` (Hagele et al. 2024, arXiv 2405.18392). A run can be
    compared with another only after its cool-down; the checkpoint at `stable_end` is un-annealed and can be continued.
    """
    spec = os.environ.get("B1K_WSD")
    if not spec:
        return None
    stable_end, total = (int(x) for x in spec.split(","))
    warmup, peak = lr_schedule.warmup_steps, lr_schedule.peak_lr
    if not warmup <= stable_end < total:
        raise ValueError(f"B1K_WSD needs warmup ({warmup}) <= stable_end < total, got {spec}")

    def schedule(step):
        warm = peak / (warmup + 1) + (peak - peak / (warmup + 1)) * jnp.minimum(step, warmup) / warmup
        frac = jnp.clip((step - stable_end) / (total - stable_end), 0.0, 1.0)
        return jnp.where(step < warmup, warm, peak * (1.0 - jnp.sqrt(frac)))

    return schedule


def create_optimizer(
    optimizer: OptimizerConfig, lr_schedule: LRScheduleConfig, weight_decay_mask: at.PyTree | None = None
) -> optax.GradientTransformation:
    lr = wsd_from_env(lr_schedule) or lr_schedule.create()
    return optimizer.create(lr, weight_decay_mask=weight_decay_mask)
