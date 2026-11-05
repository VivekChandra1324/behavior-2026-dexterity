# Added for the 2026 BEHAVIOR Challenge (team Dexterity) -- see openpi/CHANGES.md in the training repository.
"""Gripper-failure correction for BEHAVIOR-1K, ported from the 2025 winning solution.

PROVENANCE
  Ported from IliaLarchenko/behavior-1k-solution `src/b1k/shared/correction_rules.py` (Apache-2.0), the only
  component with an isolated ablation in either 2025 paper:
      "Gripper opening correction rule showed 2.2x increase in q-score for a subset of 13 tasks with 3 episodes
       each (39 episodes total)"
      "This correction rule alone approximately doubled our success rate in selected tasks where grabbing objects
       was a common failure mode."
  It exists because the dominant failure mode in their hand-labelled taxonomy is DEXTERITY (~1/3 of all failures):
  the gripper closes on nothing and the policy, never having seen that state in the idealized demos, freezes.

WHAT IS DIFFERENT HERE
  They hand-wrote the gating tables for 50 tasks (ALWAYS_OPEN_LEFT_GRIPPER_TASKS, MIN_STAGE_FOR_CLOSURE, ...).
  We DERIVE them from the demonstrations for all 100 tasks -- see scripts/derive_gripper_table.py (training repository). The
  question the table answers is exactly theirs: "in this task, at this stage, do the demos ever close this gripper?"

CONVENTIONS (verified against our dataset, not assumed)
  Model state is the 23-dim B1K vector; index 14 = left gripper width, 22 = right.
  Each finger qpos lies in [0, 0.05] and the width is their sum, so raw width in [0, 0.1], normalized
  width*20 - 1 in [-1, +1]:  -1 fully closed, +1 fully open.
  CLOSED_THRESHOLD = -0.98 is deliberately strict -- their comment: "very strict threshold to avoid false
  positives". A false positive force-opens a real grasp, which is worse than a missed correction, so every
  ambiguity in this module is resolved toward NOT correcting.

BEHAVIOUR WHEN IT FIRES
  The entire predicted action chunk is replaced by hold-position (current state tiled over the horizon) with the
  offending gripper commanded fully open (+1.0). That is their design: stop whatever the policy was doing, let go,
  and let the next chunk re-plan from a recoverable state.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

LEFT_GRIPPER_IDX = 14
RIGHT_GRIPPER_IDX = 22
CLOSED_THRESHOLD = -0.98
OPEN_COMMAND = 1.0

# A (task, stage) counts as "the demos close this gripper here" only if closure appears in at least this FRACTION
# of its frames. CALIBRATED, not guessed: swept against the winner's hand-built 50-task table
# (an internal calibration script, calibrate_gripper_threshold.py). 0.002 maximises agreement at 94/100 while never being STRICTER
# than their human-validated table -- i.e. it never introduces a false-positive correction they would have avoided.
# Raising it to 0.005+ starts over-correcting (2 tasks); lowering it to 0 drops agreement to 84/100 because a
# handful of initialisation frames licenses closure for an entire task.
MIN_CLOSED_FRAC = 0.002


@dataclasses.dataclass
class CorrectionStats:
    checked: int = 0
    fired: int = 0
    left: int = 0
    right: int = 0


class GripperCorrector:
    """Table-driven port of the winner's gripper rule, covering all 100 tasks."""

    def __init__(self, table_path: str | Path, closed_threshold: float = CLOSED_THRESHOLD,
                 min_closed_frac: float = MIN_CLOSED_FRAC, enabled: bool = True):
        self.enabled = enabled
        self.closed_threshold = closed_threshold
        self.min_closed_frac = min_closed_frac
        self.stats = CorrectionStats()
        t = json.loads(Path(table_path).read_text())
        names = t["task_names"]                      # {"0": "turning_on_radio", ...}
        self._first_closure: dict[str, dict[str, int | None]] = {}

        def _first(per, side):
            # A stage counts as "closure expected" only if closure is PREVALENT there, not merely present:
            # initialisation frames (gripper starts closed) otherwise mark whole stages on ~0.003% of frames and
            # silently disable the correction for that task. See MIN_CLOSED_FRAC.
            hits = sorted(int(s) for s, v in per.items()
                          if v[f"{side}_closed"] >= self.min_closed_frac * max(v["frames"], 1))
            return hits[0] if hits else None

        for idx, name in names.items():
            per = t["per_task"].get(str(idx), {}).get("stages", {})
            left = [_first(per, "left")] if _first(per, "left") is not None else []
            right = [_first(per, "right")] if _first(per, "right") is not None else []
            # None => the demos NEVER close this gripper in this task => any closure is a failed grasp
            self._first_closure[name] = {"left": left[0] if left else None,
                                         "right": right[0] if right else None}
        logger.info("GripperCorrector: %d tasks; never-close left=%d right=%d",
                    len(self._first_closure),
                    sum(1 for v in self._first_closure.values() if v["left"] is None),
                    sum(1 for v in self._first_closure.values() if v["right"] is None))

    def _closure_expected(self, task: str, side: str, stage: int | None) -> bool:
        """Do the demonstrations justify a closed gripper here? Unknown tasks/stages => assume yes (do not correct)."""
        rec = self._first_closure.get(task)
        if rec is None:
            return True                       # unknown task: never intervene
        first = rec[side]
        if first is None:
            return False                      # never closed anywhere in this task
        if stage is None:
            return True                       # no stage signal: fall back to task level only
        return stage >= first

    def correct(self, task: str, stage: int | None, state: np.ndarray, actions: np.ndarray):
        """Return (actions, fired, info). `actions` is (horizon, action_dim); `state` is the 23-dim model state."""
        self.stats.checked += 1
        if not self.enabled:
            return actions, False, {}
        left_closed = float(state[LEFT_GRIPPER_IDX]) < self.closed_threshold
        right_closed = float(state[RIGHT_GRIPPER_IDX]) < self.closed_threshold
        fix_left = left_closed and not self._closure_expected(task, "left", stage)
        fix_right = right_closed and not self._closure_expected(task, "right", stage)
        if not (fix_left or fix_right):
            return actions, False, {}

        corrected = np.tile(np.asarray(state, dtype=actions.dtype)[: actions.shape[1]], (actions.shape[0], 1))
        if fix_left:
            corrected[:, LEFT_GRIPPER_IDX] = OPEN_COMMAND
            self.stats.left += 1
        if fix_right:
            corrected[:, RIGHT_GRIPPER_IDX] = OPEN_COMMAND
            self.stats.right += 1
        self.stats.fired += 1
        info = {"task": task, "stage": stage, "left": fix_left, "right": fix_right,
                "left_val": float(state[LEFT_GRIPPER_IDX]), "right_val": float(state[RIGHT_GRIPPER_IDX])}
        logger.info("gripper correction: task=%s stage=%s left=%s right=%s", task, stage, fix_left, fix_right)
        return corrected, True, info


def normalize_gripper_state(state23: np.ndarray) -> np.ndarray:
    """Convert the RAW gripper widths produced by `extract_state_from_proprio` into the normalized [-1, +1] scale
    the rule thresholds against.

    THIS CONVERSION IS LOAD-BEARING. `extract_state_from_proprio` sums the two finger qpos and returns the raw
    width in [0, 0.1]. The winner's thresholds (-0.98 closed, +0.90 open) are on the NORMALIZED scale, because in
    their pipeline normalisation happens upstream. Raw values are never < -0.98, so comparing raw against those
    thresholds silently means the rule NEVER fires -- a no-op that looks like a working integration.
    Caught by an internal integration test (test_wrapper_integration.py).
    """
    out = np.array(state23, dtype=np.float64, copy=True)
    for i in (LEFT_GRIPPER_IDX, RIGHT_GRIPPER_IDX):
        out[i] = out[i] * 20.0 - 1.0          # [0, 0.1] -> [-1, +1]
    return out


def stage_of(frame: int, episode_length: int, num_stages: int) -> int:
    """The winner's purely-temporal stage index (validated: our re-derivation reproduces their 50-task table exactly)."""
    if episode_length <= 0 or num_stages <= 0:
        return 0
    return int(np.clip(int(frame / (episode_length / num_stages)), 0, num_stages - 1))


# ---------------------------------------------------------------------------------------------------------------
# Action compression, ported from the 2025 winning solution (their `eval_b1k_wrapper.py`).
#
# WHAT IT DOES. The policy predicts a 32-step chunk. Instead of executing the first 20 steps as predicted, take the
# first 26 and resample them onto 20 steps with a cubic spline. The robot then traces the SAME path 26/20 = 1.3x
# faster.
#
# WHY IT HELPS HERE SPECIFICALLY. Evaluation cuts every episode off at 1.5x the mean human demo length, and the
# metric is the fraction of goal predicates satisfied AT THE END. Partial credit is roughly half the winner's score
# (binary success was only 12.4%), so a run that gets 1.3x further down a long task before the timeout scores
# strictly better even when it never finishes. This buys progress-per-wallclock, which is the quantity the metric
# actually rewards.
#
# WHY ONLY DIMS 0:3 ARE SCALED. Our action layout is base[0:3] torso[3:7] left_arm[7:14] left_gripper[14]
# right_arm[15:22] right_gripper[22]. Everything except the base is an ABSOLUTE joint target -- replaying the same
# targets in fewer steps just reaches them sooner, no rescaling needed. The base is a VELOCITY command, so covering
# the same ground in 20 steps instead of 26 requires multiplying it by the compression factor. Scaling the joint
# targets too would send the arms to the wrong place; leaving the base unscaled would make it undershoot.
BASE_VELOCITY_SLICE = slice(0, 3)
GRIPPER_VARIATION_THRESHOLD = 0.2


def check_gripper_variation(actions: np.ndarray, n_check: int) -> tuple[bool, float, float]:
    """True when a gripper opens/closes appreciably inside the window about to be compressed.

    Interpolating through a grasp is the one case where compression is actively harmful: a cubic spline across a
    step change in gripper command smears the open->close transition over several steps, so the fingers are still
    closing while the arm has already moved on, and the object is knocked away instead of picked up. The winner
    gates on this rather than compressing unconditionally.
    """
    w = actions[:n_check]
    left = float(np.max(w[:, LEFT_GRIPPER_IDX]) - np.min(w[:, LEFT_GRIPPER_IDX]))
    right = float(np.max(w[:, RIGHT_GRIPPER_IDX]) - np.min(w[:, RIGHT_GRIPPER_IDX]))
    return (left > GRIPPER_VARIATION_THRESHOLD or right > GRIPPER_VARIATION_THRESHOLD), left, right


def compress_action_chunk(actions: np.ndarray, n_from: int, n_to: int) -> tuple[np.ndarray, bool]:
    """Resample `actions[:n_from]` onto `n_to` steps. Returns (chunk, compressed?).

    Falls back to the uncompressed first `n_to` actions -- never to something in between -- when compression is
    unsafe or impossible, so the caller always receives exactly `n_to` executable steps.
    """
    from scipy.interpolate import interp1d

    actions = np.asarray(actions)

    def _uncompressed() -> np.ndarray:
        """Exactly `n_to` steps of the raw plan, padded by HOLDING if the chunk is short.

        The caller writes into a fixed (batch, n_to, dim) buffer, so returning a short array would corrupt it.
        Padding repeats the last absolute joint target (= hold position) but ZEROES the base velocity: repeating
        a velocity command would keep the base driving indefinitely, which is the opposite of a safe fallback.
        """
        out = actions[:n_to].copy()
        if out.shape[0] < n_to:
            pad = np.repeat(out[-1:], n_to - out.shape[0], axis=0)
            pad[:, BASE_VELOCITY_SLICE] = 0.0
            out = np.concatenate([out, pad], axis=0)
        return out

    if n_to >= n_from or actions.shape[0] < n_from or n_from < 4:      # cubic needs >= 4 samples
        return _uncompressed(), False
    high_var, _, _ = check_gripper_variation(actions, n_from)
    if high_var:
        return _uncompressed(), False

    src = actions[:n_from]
    xs = np.linspace(0, n_from - 1, n_from)
    xt = np.linspace(0, n_from - 1, n_to)
    out = np.empty((n_to, src.shape[1]), dtype=np.float64)
    for d in range(src.shape[1]):
        out[:, d] = interp1d(xs, src[:, d], kind="cubic")(xt)
    # The spline maps executed step 0..n_to-1 onto predicted step 0..n_from-1, so the trajectory is time-warped by
    # (n_from-1)/(n_to-1) -- for 26->20 that is 25/19 = 1.3158, NOT 26/20 = 1.3000.
    # The winner scaled the base by n_from/n_to, which leaves the base under-driving the arm by ~1.2%. We use the
    # ratio the spline actually applies so the base and the joint targets describe the same motion. The difference
    # is small, but it is a provable inconsistency rather than a tuned value, so there is nothing to preserve.
    out[:, BASE_VELOCITY_SLICE] *= (n_from - 1) / (n_to - 1)
    return out.astype(actions.dtype), True
