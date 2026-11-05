# Modified for the 2026 BEHAVIOR Challenge (team Dexterity) -- see openpi/CHANGES.md in the training repository.
import numpy as np
import torch
from openpi_client.base_policy import BasePolicy
from openpi_client.image_tools import resize_with_pad
from openpi.configs.robots import ROBOT_REGISTRY


class B1KPolicyWrapper:
    def __init__(
        self,
        policy: BasePolicy,
        robot: str,
        text_prompt: str,
        control_mode: str,
        action_horizon: int,
        max_len: int,
        obs_size: tuple[int, int] = (224, 224),
        gripper_table: str | None = None,
        task_name: str | None = None,
        stage_fn=None,
        compress_from: int | None = None,
        compress_to: int | None = None,
    ) -> None:
        # Load robot config from registry
        self.robot = ROBOT_REGISTRY[robot]
        self.policy = policy
        self.text_prompt = text_prompt
        self.control_mode = control_mode
        self.action_horizon = action_horizon
        self.obs_size = obs_size
        self.max_len = max_len
        self.temporal_ensemble_max = 5  # max number of sequences to ensemble

        # Extract gripper indices from robot config (end-effectors to preserve)
        self.gripper_indices = []
        for action_config in self.robot.action:
            if action_config.is_eef and action_config.indices is not None:
                self.gripper_indices.extend(action_config.indices)
        # Vectorized action buffers (initialized on first call)
        # --- B1K addition: gripper-failure correction, ported from the 2025 winning solution.
        # OPT-IN: disabled unless `gripper_table` is given, so default behaviour is unchanged.
        # Their ablation: "2.2x increase in q-score for a subset of 13 tasks". See
        # openpi/shared/b1k_correction_rules.py.
        # `stage_fn` is optional: with no stage signal the rule degrades gracefully to the TASK-level test
        # ("this task never closes this gripper at all"), which still covers 90/100 tasks for the left gripper
        # and 78/100 for the right. A learned stage head can be supplied later without touching this code.
        self.task_name = task_name or text_prompt
        self.stage_fn = stage_fn
        self.gripper_corrector = None
        if gripper_table:
            from openpi.shared.b1k_correction_rules import GripperCorrector
            self.gripper_corrector = GripperCorrector(gripper_table)

        # --- B1K addition: action compression, ported from the 2025 winning solution.
        # OPT-IN: off unless BOTH compress_from and compress_to are given, so the stock path is untouched.
        # Their defaults were 26 -> 20 (factor 1.3). See b1k_correction_rules.compress_action_chunk for the
        # reasoning, especially why only the base velocity dims are rescaled.
        self.compress_from = compress_from
        self.compress_to = compress_to
        self.compression_enabled = bool(compress_from and compress_to and compress_to < compress_from)
        if self.compression_enabled and compress_from > action_horizon:
            raise ValueError(f"compress_from={compress_from} exceeds action_horizon={action_horizon}")
        # Replanning cadence: with compression we consume `compress_to` steps per chunk, not `action_horizon`.
        # Getting this wrong is silent -- the robot would simply re-run stale actions -- so it is derived here once.
        self.execute_steps = self.compress_to if self.compression_enabled else action_horizon
        self.n_compressed = 0
        self.n_chunks = 0

        self.batch_size = None
        self.action_buffer = None  # Shape: (batch, max_sequences, max_horizon, action_dim)
        self.sequence_indices = None  # Shape: (batch, max_sequences) - current position in each sequence
        self.sequence_lengths = None  # Shape: (batch, max_sequences) - total length of each sequence
        self.num_active_sequences = None  # Shape: (batch,) - number of active sequences per batch element
        self.step_counter = None  # Shape: (batch,)

    def reset(self):
        self.batch_size = None
        self.action_buffer = None
        self.sequence_indices = None
        self.sequence_lengths = None
        self.num_active_sequences = None
        self.step_counter = None
        self.policy.reset()

    def _ensure_batch_initialized(self, batch_size: int, action_dim: int = None):
        """Ensure buffers are initialized for the given batch size."""
        if self.batch_size != batch_size or self.action_buffer is None:
            self.batch_size = batch_size

            # For receding_horizon: simple buffer
            # We'll set action_dim when we first see actions
            if action_dim is not None:
                self.action_buffer = np.zeros(
                    (batch_size, self.temporal_ensemble_max, self.max_len, action_dim), dtype=np.float32
                )
                self.sequence_indices = np.zeros((batch_size, self.temporal_ensemble_max), dtype=np.int32)
                self.sequence_lengths = np.zeros((batch_size, self.temporal_ensemble_max), dtype=np.int32)
                self.num_active_sequences = np.zeros(batch_size, dtype=np.int32)

            self.step_counter = np.zeros(batch_size, dtype=np.int32)

    def process_input(self, obs: dict) -> dict:
        """
        Process the input dictionary to match the expected input format for the model.
        """
        prop_state = obs[f"{self.robot.name}::proprio"]
        if prop_state.ndim == 1:
            prop_state = prop_state[None, :]  # Add batch dimension
        batch_size = prop_state.shape[0]
        # Process camera images from robot config
        observations = []
        for camera_key in sorted(self.robot.observations.keys()):
            camera_obs = obs[self.robot.observations[camera_key].obs_key][..., :3]
            if camera_obs.ndim == 3:
                camera_obs = camera_obs[None, ...]  # Add batch dimension
            observations.append(resize_with_pad(camera_obs, *self.obs_size))

        # Pad with zeros if we have fewer than 3 cameras (expected by model)
        while len(observations) < 3:
            observations.append(np.zeros((batch_size, *self.obs_size, 3), dtype=np.uint8))
        img_obs = np.stack(observations, axis=1)  # Shape: (B, 3, H, W, C)
        processed_input = [
            {
                "observation/image_0": img_obs[i, 0],
                "observation/image_1": img_obs[i, 1],
                "observation/image_2": img_obs[i, 2],
                "observation/state": prop_state[i],
                "prompt": self.text_prompt,
            }
            for i in range(batch_size)
        ]
        return processed_input

    def _get_current_actions(
        self,
        *,
        batch_range: np.ndarray,
        seq_range: np.ndarray,
        active_mask: np.ndarray,
        current_indices: np.ndarray,
        sequence_lengths: np.ndarray,
    ) -> np.ndarray:
        """Safely gather current actions for active sequences only."""
        safe_lengths = np.maximum(sequence_lengths, 1)
        safe_indices = np.minimum(current_indices, safe_lengths - 1)
        safe_indices = np.where(active_mask, safe_indices, 0)
        batch_idx = batch_range[:, None, None]
        seq_idx = seq_range[None, :, None]
        pos_idx = safe_indices[:, :, None]
        return self.action_buffer[batch_idx, seq_idx, pos_idx].squeeze(-2)

    def _apply_gripper_correction(self, input_obs, indices, target_action):
        """Overwrite a predicted chunk with hold-position + open gripper when the demos say this gripper is never
        closed here. No-op unless a gripper table was supplied."""
        if self.gripper_corrector is None:
            return target_action
        from openpi.policies.b1k_policy import extract_state_from_proprio
        from openpi.shared.b1k_correction_rules import normalize_gripper_state

        proprio = np.asarray(input_obs[f"{self.robot.name}::proprio"])
        if proprio.ndim == 1:
            proprio = proprio[None]
        stage = self.stage_fn() if self.stage_fn is not None else None
        for k, i in enumerate(indices):
            state23 = np.asarray(extract_state_from_proprio(proprio[i], self.robot)).reshape(-1)
            # raw finger-sum widths -> the normalized scale the thresholds are defined on (see that function)
            state23 = normalize_gripper_state(state23)
            fixed, fired, info = self.gripper_corrector.correct(
                self.task_name, stage, state23, target_action[k]
            )
            if fired:
                target_action[k] = fixed[:, : target_action.shape[2]]
        return target_action

    def _compress_actions(self, target_action: np.ndarray) -> np.ndarray:
        """Resample each predicted chunk to `compress_to` steps. No-op unless compression is enabled.

        Applied AFTER the gripper correction on purpose: the correction may replace a chunk with a hold-position
        command, and that replacement should be compressed like any other chunk rather than bypassing the timing.
        """
        if not self.compression_enabled:
            return target_action
        from openpi.shared.b1k_correction_rules import compress_action_chunk

        out = np.empty((target_action.shape[0], self.compress_to, target_action.shape[2]), dtype=target_action.dtype)
        for k in range(target_action.shape[0]):
            chunk, did = compress_action_chunk(target_action[k], self.compress_from, self.compress_to)
            out[k] = chunk
            self.n_compressed += int(did)
            self.n_chunks += 1
        return out

    def act_receding_horizon(self, input_obs):
        """
        receding horizon: execute actions from current plan for up to action_horizon steps, then replan.
        Uses vectorized buffers for efficient batch processing.
        """
        batched = input_obs[f"{self.robot.name}::proprio"].ndim != 1
        input_batch = self.process_input(input_obs)
        batch_size = len(input_batch)
        if self.sequence_indices is None:
            needs_inference = np.ones(batch_size, dtype=bool)
        else:
            needs_inference = (self.sequence_indices[:, 0] >= self.sequence_lengths[:, 0]) | (
                (self.step_counter % self.execute_steps) == 0
            )

        if needs_inference.any():
            indices_needing_inference = np.where(needs_inference)[0]
            # Create sub-batch for elements that need inference
            action = []
            for i in indices_needing_inference:
                action.append(self.policy.infer(input_batch[i]))
            target_action = np.array([a["actions"].copy() for a in action])  # (sub_batch_size, T, action_dim)
            target_action = self._apply_gripper_correction(input_obs, indices_needing_inference, target_action)
            target_action = self._compress_actions(target_action)

            # Initialize buffers on first inference
            if self.action_buffer is None:
                action_dim = target_action.shape[2]
                self._ensure_batch_initialized(batch_size, action_dim)

            # Store actions in buffer
            seq_len = min(target_action.shape[1], self.max_len)
            self.action_buffer[indices_needing_inference, 0, :seq_len] = target_action[:, :seq_len]
            self.sequence_lengths[indices_needing_inference, 0] = seq_len
            self.sequence_indices[indices_needing_inference, 0] = 0

        # Extract next action for each batch element (fully vectorized!)
        batch_range = np.arange(batch_size)
        current_indices = self.sequence_indices[batch_range, 0]
        final_actions = self.action_buffer[batch_range, 0, current_indices]

        # Increment indices (vectorized)
        self.sequence_indices[:, 0] += 1
        self.step_counter += 1
        if not batched:
            final_actions = final_actions[0]
        return torch.from_numpy(final_actions)

    def act_receding_temporal(self, input_obs):
        """
        receding temporal: infer every K steps and smooth actions across recent sequences.
        Uses vectorized buffers for efficient batch processing.
        """
        batched = input_obs[f"{self.robot.name}::proprio"].ndim != 1
        input_batch = self.process_input(input_obs)
        batch_size = len(input_batch)

        # Initialize buffers on first call
        if self.action_buffer is None:
            # Need to infer once to get action_dim
            action = []
            for i in range(batch_size):
                action.append(self.policy.infer(input_batch[i]))
            target_action = np.array([a["actions"].copy() for a in action])  # (B, T, action_dim)
            action_dim = target_action.shape[2]
            self._ensure_batch_initialized(batch_size, action_dim)

            # Store first inference for all batch elements
            seq_len = min(target_action.shape[1], self.max_len)
            self.action_buffer[:, 0, :seq_len] = target_action[:, :seq_len]
            self.sequence_lengths[:, 0] = seq_len
            self.sequence_indices[:, 0] = 0
            self.num_active_sequences[:] = 1

        # Step 1: Run policy for elements that need replanning
        needs_replan = (self.step_counter % self.action_horizon) == 0
        if needs_replan.any():
            indices_needing_replan = np.where(needs_replan)[0]

            # Run inference only on sub-batch
            action = []
            for i in indices_needing_replan:
                action.append(self.policy.infer(input_batch[i]))
            target_action = np.array([a["actions"].copy() for a in action])  # (B, T, action_dim)

            # Add new sequences (vectorized where possible)
            seq_len = min(target_action.shape[1], self.max_len)

            # Handle elements that need shifting
            needs_shift = self.num_active_sequences[indices_needing_replan] >= self.temporal_ensemble_max
            if needs_shift.any():
                shift_indices = indices_needing_replan[needs_shift]
                # Vectorized shift for all elements that need it
                self.action_buffer[shift_indices, :-1] = self.action_buffer[shift_indices, 1:]
                self.sequence_indices[shift_indices, :-1] = self.sequence_indices[shift_indices, 1:]
                self.sequence_lengths[shift_indices, :-1] = self.sequence_lengths[shift_indices, 1:]

            # Calculate insert indices vectorized
            insert_indices = np.where(
                needs_shift, self.temporal_ensemble_max - 1, self.num_active_sequences[indices_needing_replan]
            )

            # Increment active sequences for elements that don't need shifting
            self.num_active_sequences[indices_needing_replan[~needs_shift]] += 1

            # Store new sequences (fully vectorized using advanced indexing)
            self.action_buffer[indices_needing_replan, insert_indices, :seq_len] = target_action[:, :seq_len]
            self.sequence_indices[indices_needing_replan, insert_indices] = 0
            self.sequence_lengths[indices_needing_replan, insert_indices] = seq_len

        # Step 2: Extract and ensemble current actions
        action_dim = self.action_buffer.shape[3]
        max_seq = self.temporal_ensemble_max
        seq_range = np.arange(max_seq)
        batch_range = np.arange(batch_size)
        # Create mask for active sequences (B, max_seq)
        active_mask = seq_range[None, :] < self.num_active_sequences[:, None]
        current_indices = self.sequence_indices[:, :max_seq]  # (B, max_seq)
        actions_current = self._get_current_actions(
            batch_range=batch_range,
            seq_range=seq_range,
            active_mask=active_mask,
            current_indices=current_indices,
            sequence_lengths=self.sequence_lengths[:, :max_seq],
        )  # (B, max_seq, action_dim)
        k = 0.005
        exp_weights = np.exp(k * seq_range)[None, :]  # (1, max_seq)
        masked_weights = exp_weights * active_mask  # (B, max_seq)
        normalized_weights = masked_weights / masked_weights.sum(axis=1, keepdims=True)  # (B, max_seq)
        final_actions = (actions_current * normalized_weights[:, :, None]).sum(axis=1)  # (B, action_dim)

        # Preserve grippers from most recent rollout
        # Get the last active sequence index for each batch element
        last_seq_idx = self.num_active_sequences - 1
        for gripper_idx in self.gripper_indices:
            final_actions[:, gripper_idx] = actions_current[batch_range, last_seq_idx, gripper_idx]

        # Increment sequence indices
        self.sequence_indices[:, :max_seq] += 1

        # Check which sequences are still active (B, max_seq)
        still_active = self.sequence_indices[:, :max_seq] < self.sequence_lengths[:, :max_seq]
        still_active = still_active & active_mask  # Only consider currently active sequences
        # Drop exhausted sequences (compaction per batch element)
        for b in range(batch_size):
            active_in_slot = still_active[b, :]
            if not active_in_slot[: self.num_active_sequences[b]].all():
                # Some sequences exhausted, compact
                active_count = active_in_slot.sum()
                active_indices = np.where(active_in_slot)[0]
                self.action_buffer[b, :active_count] = self.action_buffer[b, active_indices]
                self.sequence_indices[b, :active_count] = self.sequence_indices[b, active_indices]
                self.sequence_lengths[b, :active_count] = self.sequence_lengths[b, active_indices]
                self.num_active_sequences[b] = active_count

        self.step_counter += 1
        if not batched:
            final_actions = final_actions[0]
        return torch.from_numpy(final_actions)

    def act_temporal_ensemble(self, input_obs):
        """
        Temporal ensemble: infer at every step and smooth via exponentially weighted average of all recent action sequences.
        """
        batched = input_obs[f"{self.robot.name}::proprio"].ndim != 1
        input_batch = self.process_input(input_obs)
        batch_size = len(input_batch)
        action = []
        for i in range(batch_size):
            action.append(self.policy.infer(input_batch[i]))
        target_action = np.array([a["actions"].copy() for a in action])  # (B, T, action_dim)
        action_dim = target_action.shape[2]

        # Initialize buffers on first call
        if self.action_buffer is None:
            self._ensure_batch_initialized(batch_size, action_dim)
            self.num_active_sequences[:] = 0

        # Vectorized shift for all elements that need it, increment active counts otherwise
        needs_shift = self.num_active_sequences >= self.temporal_ensemble_max
        if needs_shift.any():
            shift_indices = np.where(needs_shift)[0]
            self.action_buffer[shift_indices, :-1] = self.action_buffer[shift_indices, 1:]
            self.sequence_indices[shift_indices, :-1] = self.sequence_indices[shift_indices, 1:]
            self.sequence_lengths[shift_indices, :-1] = self.sequence_lengths[shift_indices, 1:]
        self.num_active_sequences[~needs_shift] += 1

        # Insert new sequences to action buffer
        insert_indices = np.where(needs_shift, self.temporal_ensemble_max - 1, self.num_active_sequences - 1)
        seq_len = min(target_action.shape[1], self.max_len)
        batch_range = np.arange(batch_size)
        self.action_buffer[batch_range, insert_indices, :seq_len] = target_action[:, :seq_len]
        self.sequence_indices[batch_range, insert_indices] = 0
        self.sequence_lengths[batch_range, insert_indices] = seq_len

        # Extract current actions
        seq_range = np.arange(self.temporal_ensemble_max)
        active_mask = seq_range[None, :] < self.num_active_sequences[:, None]
        current_indices = self.sequence_indices[:, : self.temporal_ensemble_max]  # (B, max_seq)
        actions_current = self._get_current_actions(
            batch_range=batch_range,
            seq_range=seq_range,
            active_mask=active_mask,
            current_indices=current_indices,
            sequence_lengths=self.sequence_lengths[:, : self.temporal_ensemble_max],
        )  # (B, max_seq, action_dim)

        k = 0.005
        exp_weights = np.exp(k * seq_range)[None, :]  # (1, max_seq)
        masked_weights = exp_weights * active_mask  # (B, max_seq)
        normalized_weights = masked_weights / masked_weights.sum(axis=1, keepdims=True)  # (B, max_seq)
        final_actions = (actions_current * normalized_weights[:, :, None]).sum(axis=1)  # (B, action_dim)

        # Preserve grippers from most recent rollout
        for gripper_idx in self.gripper_indices:
            final_actions[:, gripper_idx] = target_action[:, 0, gripper_idx]

        # Increment all sequence indices
        self.sequence_indices[:, : self.temporal_ensemble_max] += 1

        # Drop exhausted sequences (compaction per batch element)
        still_active = (
            self.sequence_indices[:, : self.temporal_ensemble_max]
            < self.sequence_lengths[:, : self.temporal_ensemble_max]
        )
        still_active = still_active & active_mask
        for b in range(batch_size):
            active_in_slot = still_active[b, :]
            if not active_in_slot[: self.num_active_sequences[b]].all():
                active_count = active_in_slot.sum()
                active_indices = np.where(active_in_slot)[0]
                self.action_buffer[b, :active_count] = self.action_buffer[b, active_indices]
                self.sequence_indices[b, :active_count] = self.sequence_indices[b, active_indices]
                self.sequence_lengths[b, :active_count] = self.sequence_lengths[b, active_indices]
                self.num_active_sequences[b] = active_count

        if not batched:
            final_actions = final_actions[0]
        return torch.from_numpy(final_actions)

    def set_task(self, task_name: str):
        """B1K addition (multi-task serving, `--task b1k/auto`): switch prompt and task when the evaluator's
        task_id names another task. A new task always starts from a fresh plan; the same task is a no-op."""
        if task_name == self.task_name:
            return
        from openpi.configs.tasks import TASK_REGISTRY
        self.text_prompt = TASK_REGISTRY["b1k"][task_name]
        self.task_name = task_name
        self.reset()

    def act_chunk(self, input_obs, k: int):
        """B1K addition: the evaluator's optional action-chunk replay protocol
        (BEHAVIOR-1K docs/challenge/evaluation.md, `--replay-action-chunk-size K`).
        Returns the current action plus the next k-1 actions already planned from this observation, shape
        (..., k, action_dim), and advances the buffers by k steps. Identical to k per-step act() calls whenever the
        current plan still holds k actions before its replan point. Otherwise (e.g. k=16 while compression consumes
        20 steps per plan) it replans from THIS observation rather than crossing the replan point, so a chunk is
        always open-loop from one observation and the evaluator's chunk size can never crash the server."""
        if self.control_mode != "receding_horizon" or not 1 <= k <= self.execute_steps:
            raise ValueError(f"chunk replay needs receding_horizon and k <= {self.execute_steps}; got "
                             f"{self.control_mode}, k={k}")
        if self.step_counter is not None:
            remaining = self.execute_steps - self.step_counter % self.execute_steps  # actions left in this plan
            if (remaining < k).any():
                self.step_counter[:] = 0                  # due for a replan: the next call runs the model
        first = self.act_receding_horizon(input_obs).numpy()
        idx = self.sequence_indices[:, 0]  # already advanced past `first`
        if (idx + k - 1 > self.sequence_lengths[:, 0]).any():
            raise ValueError("chunk is longer than the planned sequence")
        rest = np.stack([self.action_buffer[b, 0, idx[b] : idx[b] + k - 1] for b in range(len(idx))])
        self.sequence_indices[:, 0] += k - 1
        self.step_counter += k - 1
        chunk = np.concatenate([first.reshape(len(idx), 1, -1), rest], axis=1)
        if first.ndim == 1:
            chunk = chunk[0]
        return torch.from_numpy(chunk)

    def act(self, input_obs):
        """Dispatch to the appropriate action method based on control mode."""
        if self.control_mode == "receding_temporal":
            return self.act_receding_temporal(input_obs)
        elif self.control_mode == "receding_horizon":
            return self.act_receding_horizon(input_obs)
        elif self.control_mode == "temporal_ensemble":
            return self.act_temporal_ensemble(input_obs)
        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")
