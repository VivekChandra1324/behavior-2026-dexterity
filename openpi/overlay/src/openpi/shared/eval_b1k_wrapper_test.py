# Added for the 2026 BEHAVIOR Challenge (team Dexterity) -- see openpi/CHANGES.md in the training repository.
import numpy as np
import pytest

from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper

T, A = 32, 23  # model chunk length, action dim


class _FakePolicy:
    """Each infer() returns a distinct (T, A) chunk, so replayed actions identify the call that produced them."""

    def __init__(self, steady_grippers=False):
        self.calls = 0
        self.steady_grippers = steady_grippers  # constant gripper dims, so compression is not vetoed by its grasp guard

    def infer(self, _):
        self.calls += 1
        a = np.arange(T * A, dtype=np.float32).reshape(T, A) + 1000 * self.calls
        if self.steady_grippers:
            a[:, [14, 22]] = 1.0
        return {"actions": a}

    def reset(self):
        pass


def _obs(batch=None):
    """Observation keys exactly as BEHAVIOR-1K v3.9.3's evaluator sends them (robot `name: robot_r1` in r1pro.yaml,
    cameras from its eval.camera_sensor_names, flattened as `<robot>::<sensor>::<modality>`)."""
    shape = () if batch is None else (batch,)
    obs = {"robot_r1::proprio": np.zeros((*shape, 61), dtype=np.float32)}
    for cam in ("zed_link", "left_realsense_link", "right_realsense_link"):
        obs[f"robot_r1::robot_r1:{cam}:Camera:0::rgb"] = np.zeros((*shape, 64, 64, 4), dtype=np.uint8)
    return obs


def _wrapper(policy):
    return B1KPolicyWrapper(policy=policy, robot="b1k/R1Pro", text_prompt="x", control_mode="receding_horizon",
                            action_horizon=16, max_len=T)


@pytest.mark.parametrize("batch", [None, 2])
def test_act_chunk_matches_per_step_act(batch):
    step_policy, chunk_policy = _FakePolicy(), _FakePolicy()
    per_step = np.stack(_run_steps(_wrapper(step_policy), batch, 48))  # (48, ..., A)
    w = _wrapper(chunk_policy)
    chunks = [w.act_chunk(_obs(batch), 16) for _ in range(3)]
    replayed = np.concatenate([c.numpy() for c in chunks], axis=-2)  # (..., 48, A)
    np.testing.assert_array_equal(np.moveaxis(replayed, -2, 0), per_step)
    assert chunk_policy.calls == step_policy.calls  # same model calls, at the same steps
    for c in chunks:  # protocol: the first action of the chunk is the current action
        assert c.shape == ((16, A) if batch is None else (batch, 16, A))


def _run_steps(w, batch, n):
    return [w.act(_obs(batch)).numpy() for _ in range(n)]


def test_act_chunk_replans_instead_of_crossing_a_replan_boundary():
    policy = _FakePolicy()
    w = _wrapper(policy)
    first = w.act(_obs()).numpy()                 # one per-step action: only 15 planned actions remain
    chunk = w.act_chunk(_obs(), 16).numpy()       # cannot be served from this plan -> replan from this observation
    assert policy.calls == 2
    np.testing.assert_array_equal(chunk, _FakePolicy().infer(None)["actions"][:16] + 1000)  # 2nd plan, steps 0..15
    assert not np.array_equal(first, chunk[0])


def test_act_chunk_longer_than_a_plan_is_rejected():
    with pytest.raises(ValueError):
        _wrapper(_FakePolicy()).act_chunk(_obs(), 17)


def _compressed(policy):
    return B1KPolicyWrapper(policy=policy, robot="b1k/R1Pro", text_prompt="x", control_mode="receding_horizon",
                            action_horizon=26, max_len=T, compress_from=26, compress_to=20)


def test_compressed_chunk_matching_the_plan_equals_per_step():
    step_policy, chunk_policy = _FakePolicy(True), _FakePolicy(True)
    per_step = np.stack(_run_steps(_compressed(step_policy), None, 60))
    w = _compressed(chunk_policy)
    replayed = np.concatenate([w.act_chunk(_obs(), 20).numpy() for _ in range(3)])
    np.testing.assert_allclose(replayed, per_step, rtol=0, atol=1e-6)
    assert chunk_policy.calls == step_policy.calls == 3
    assert w.n_compressed == 3                    # the spline really ran on every plan


def test_compressed_server_with_evaluator_chunk_16_does_not_crash():
    """Compression consumes 20 steps per plan; an evaluator asking for 16 must get a replan, never an exception."""
    policy = _FakePolicy(True)
    w = _compressed(policy)
    chunks = [w.act_chunk(_obs(), 16).numpy() for _ in range(4)]
    assert policy.calls == 4                      # every request starts a fresh plan (replan every 16 steps)
    assert w.n_compressed == 4
    for c in chunks:
        assert c.shape == (16, A)
    ref = _compressed(_FakePolicy(True)).act_chunk(_obs(), 20).numpy()
    np.testing.assert_allclose(chunks[0], ref[:16], rtol=0, atol=1e-6)   # first 16 steps of the compressed plan


class _PromptPolicy(_FakePolicy):
    def __init__(self):
        super().__init__(); self.prompts = []

    def infer(self, x):
        self.prompts.append(x["prompt"]); return super().infer(x)


def test_set_task_switches_prompt_and_resets_the_plan():
    policy = _PromptPolicy()
    w = _wrapper(policy)
    w.act(_obs()); w.act(_obs())                      # 1 inference, mid-plan
    w.set_task("cook_broccolini")
    w.act(_obs())                                     # new task -> fresh plan, new prompt
    assert policy.prompts == ["x", "cook_broccolini"] and w.task_name == "cook_broccolini"
    w.set_task("cook_broccolini"); w.act(_obs())      # same task again: no reset, still replaying the plan
    assert len(policy.prompts) == 2


def test_task_from_request_maps_evaluator_ids():
    from openpi.serving.websocket_b1k_server import task_from_request
    assert task_from_request({"task_id": np.array([[96]])}) == "cook_broccolini"
    assert task_from_request({"task_id": 0}) == "turning_on_radio"
    with pytest.raises(KeyError):
        task_from_request({})
