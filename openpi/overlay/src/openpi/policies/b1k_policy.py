# Modified for the 2026 BEHAVIOR Challenge (team Dexterity) -- see openpi/CHANGES.md in the training repository.
import os
import json
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.configs.robots.base_config import RobotConfig


def make_b1k_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/image_0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_2": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(23),
        "prompt": "do something",
    }


def extract_state_from_proprio(proprio_data, robot_config: RobotConfig) -> np.ndarray:
    """Extract state from proprioception data based on robot configuration.

    We assume perfect correlation for the two gripper fingers.

    Args:
        proprio_data: Raw proprioception data
        robot_config: RobotConfig instance containing robot configuration

    Returns:
        Extracted state array
    """
    state = []
    for proprio in robot_config.proprio:
        if proprio.is_eef:
            # Sum the gripper finger positions to get a single width value
            state.append(proprio_data[..., proprio.indices].sum(axis=-1, keepdims=True))
        else:
            state.append(proprio_data[..., proprio.indices])
    return np.concatenate(state, axis=-1)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class B1KInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    # Robot configuration object
    robot_config: RobotConfig = dataclasses.field(default=None)

    def __call__(self, data: dict) -> dict:
        proprio_data = data["observation/state"]
        # extract joint position
        state = extract_state_from_proprio(proprio_data, self.robot_config)
        if "actions" in data:
            action = data["actions"]

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        images, image_masks = [], []
        for camera_id in self.robot_config.observations:
            images.append(_parse_image(data[f"observation/{camera_id}"]))
            image_masks.append(np.True_)
        while len(images) < 3:
            images.append(np.zeros_like(images[0]))
            image_masks.append(np.False_)
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = tuple(images)
                image_masks = tuple(image_masks)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = tuple(images)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = action

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # B1K addition (opt-in: B1K_STAGE_AUX=1): the temporal stage label for the auxiliary head.
        # stage = floor(frame / (episode_length / N)), N per task from scripts/compute_stage_table.py (training repo).
        # It is a TARGET only -- nothing downstream feeds it to the model -- and unlike a gripper-state label it
        # cannot be read off the observation, which is the point: the same scene looks identical early and late in
        # a task. See our project notes.
        if _stage_table() is not None and "frame_index" in data and "episode_index" in data:
            inputs["stage"] = _stage_label(int(data["frame_index"]), int(data["episode_index"]),
                                           int(data.get("task_index", 0)))

        return inputs


@dataclasses.dataclass(frozen=True)
class B1KOutputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    def __call__(self, data: dict) -> dict:
        # Only return the first 23 dims.
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}


# --- temporal stage labels (opt-in, B1K_STAGE_AUX=1) ---------------------------------------------------------
MAX_STAGES = 15  # the largest N the table can assign; the head has this many classes

_STAGE_TABLE = None
_STAGE_TABLE_LOADED = False


def _stage_table():
    """Load the stage table once per process (DataLoader workers are spawned, so each loads its own copy)."""
    global _STAGE_TABLE, _STAGE_TABLE_LOADED
    if not _STAGE_TABLE_LOADED:
        _STAGE_TABLE_LOADED = True
        path = os.environ.get("B1K_STAGE_TABLE")
        if os.environ.get("B1K_STAGE_AUX") == "1" and path and os.path.exists(path):
            with open(path) as f:
                raw = json.load(f)
            _STAGE_TABLE = ({int(k): v for k, v in raw["n_stages"].items()},
                            {int(k): v for k, v in raw["length"].items()})
    return _STAGE_TABLE


def _stage_label(frame_index: int, episode_index: int, task_index: int) -> int:
    n_stages_by_task, length_by_episode = _stage_table()
    n = n_stages_by_task.get(task_index, 5)
    length = length_by_episode.get(episode_index, 0)
    if length <= 0:
        return 0
    return min(int(frame_index // max(length / n, 1)), n - 1)
