# Modified for the 2026 BEHAVIOR Challenge (team Dexterity) -- see openpi/CHANGES.md in the training repository.
import dataclasses
import logging
import socket
import tyro

from openpi.configs.tasks import TASK_REGISTRY
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_b1k_server
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
from openpi.training import config as _config


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    robot: str
    task: str
    # Specifies how to load the policy.
    policy: Checkpoint

    # LeRobot repo id whose checkpoint assets contain the norm stats.
    repo_id: str | None = None
    control_mode: str = "receding_horizon"
    # Number of actions to execute before replanning.
    action_horizon: int = 16
    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False
    # B1K additions (2025-winner inference measures; off unless given):
    # gripper re-open rule table (serving/gripper_table.json in the training repository), and action compression, e.g.
    # --action_horizon 26 --compress_from 26 --compress_to 20 (execute 26 planned steps in 20, ~1.3x faster).
    gripper_table: str | None = None
    compress_from: int | None = None
    compress_to: int | None = None


def main(args: Args) -> None:
    # Load task from registry
    task_bucket, task_name = args.task.split("/")
    # B1K addition: `--task b1k/auto` serves every task; the prompt follows each request's task_id
    auto_task = task_name == "auto"
    task_prompt = None if auto_task else TASK_REGISTRY[task_bucket][task_name]
    # log the prompt used
    logging.info(f"Using robot: {args.robot}, prompt: {task_prompt}")
    
    # Load training config and override request-specific fields.
    config = _config.get_config(args.policy.config)
    norm_stats_repo_id = args.repo_id or args.task
    logging.info("Using norm stats for repo: %s", norm_stats_repo_id)
    config = dataclasses.replace(
        config, data=dataclasses.replace(config.data, repo_id=norm_stats_repo_id, robot_config_name=args.robot)
    )

    policy = _policy_config.create_trained_policy(
        config, args.policy.dir, default_prompt=task_prompt
    )
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    policy = B1KPolicyWrapper(
        policy=policy,
        robot=args.robot,
        text_prompt=task_prompt or "",
        control_mode=args.control_mode,
        action_horizon=args.action_horizon,
        max_len=config.model.action_horizon,
        gripper_table=args.gripper_table,
        task_name=None if auto_task else task_name,
        compress_from=args.compress_from,
        compress_to=args.compress_to,
    )
    logging.info("Inference measures: gripper_table=%s compress=%s->%s, action_horizon=%d, replan every %d steps",
                 args.gripper_table, args.compress_from, args.compress_to, args.action_horizon, policy.execute_steps)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_b1k_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
        auto_task=auto_task,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
