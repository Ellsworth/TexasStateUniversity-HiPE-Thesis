import dataclasses
import numpy as np
import d3rlpy
import argparse
import os

import torch
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from d3rlpy.logging import TensorboardAdapterFactory

from firebot_agent.utils import convert_continuous_to_discrete
from firebot_agent.gym_env import FireBotEnv
from firebot_agent.heatmap_wrapper import PositionHeatmapWrapper
from firebot_agent.reward_tracker_wrapper import RewardTrackerWrapper
from firebot_agent.log_master import FireBotLogger
from firebot_agent.training_utils import (
    print_gpu_info,
    load_dataset,
    get_evaluators,
    prestack_frames,
    GridObservationWrapper,
)


def main():
    parser = argparse.ArgumentParser(description="DiscreteCQL Online Fine-tuning")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained .pt model file")
    parser.add_argument("--dataset", type=str, default="offline_dataset.npz", help="Path to offline dataset for replay blending (npz file or directory)")
    parser.add_argument("--finetune-steps", type=int, default=100000, help="Number of online fine-tuning steps")
    parser.add_argument("--n-frames", type=int, default=4, help="Number of frames to stack")
    parser.add_argument("--buffer-size", type=int, default=200000, help="Replay buffer size for online training")
    parser.add_argument("--mix-ratio", type=float, default=0.5, help="Fraction of each batch sampled from offline data (0.0=all online, 1.0=all offline)")
    parser.add_argument("--mock", action="store_true", help="Use mock environment (no ZMQ)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="ZMQ server host")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ server port")

    args = parser.parse_args()

    # Create a single timestamped log directory for this run
    logger = FireBotLogger(base_dir="logs", experiment_name="CQL_Finetune")
    log_dir = logger.get_log_dir()

    # Output model path
    model_path = os.path.join(log_dir, "cql_finetuned.d3")

    print_gpu_info()

    # ── Create environment ──────────────────────────────────────────────────
    online_dataset_path = os.path.join(log_dir, "online_dataset.npz")
    env = FireBotEnv(
        ip=args.host,
        port=args.port,
        discrete_actions=True,
        mock=args.mock,
        agent_name="CQL_Agent",
        record_data=True,
        output_file=online_dataset_path,
        max_episode_steps=10_000
    )
    env = PositionHeatmapWrapper(env, save_every=10_000, save_dir=log_dir)
    env = RewardTrackerWrapper(env, log_dir=log_dir)
    env = GridObservationWrapper(env)

    if args.n_frames > 1:
        print(f"Stacking {args.n_frames} frames...")
        env = FrameStackObservation(env, stack_size=args.n_frames)

    # ── Load pretrained model ───────────────────────────────────────────────
    print("=" * 60)
    print("ONLINE FINE-TUNING WITH DISCRETECQL")
    print("=" * 60)

    cql = d3rlpy.load_learnable(args.checkpoint, device=torch.cuda.is_available())
    print(f"Loaded checkpoint from {args.checkpoint}")

    learning_rate = 3e-5
    cql._config = dataclasses.replace(
        cql._config,
        learning_rate=learning_rate
    )
    print(f"Learning rate set to {learning_rate}")

    evaluators = get_evaluators()

    # ── Frame-stacking strategy for fine-tuning ─────────────────────────────
    # The online env is wrapped with FrameStackObservation, so each step
    # already returns (n_frames, H, W) observations.  Storing those in the
    # FIFO buffer and then applying FrameStackTransitionPicker on top would
    # double-stack.  Instead we:
    #   1. Use the default BasicTransitionPicker for the online buffer.
    #   2. Pre-stack the offline observations to (n_frames, H, W) so they
    #      match the online format.

    # Create online FIFO buffer (BasicTransitionPicker by default)
    online_buffer = d3rlpy.dataset.create_fifo_replay_buffer(
        limit=args.buffer_size,
        env=env,
    )

    # Load offline data to blend into training batches
    print("Loading offline dataset for replay buffer blending...")
    off_obs, off_act_raw, off_rew, off_term = load_dataset(args.dataset)
    off_act = convert_continuous_to_discrete(off_act_raw)

    if args.n_frames > 1:
        print(f"Pre-stacking offline observations to ({args.n_frames}, H, W)...")
        off_obs = prestack_frames(off_obs, off_term, args.n_frames)
        print(f"Pre-stacked shape: {off_obs.shape}")

    offline_mdp = d3rlpy.dataset.MDPDataset(
        observations=off_obs,
        actions=off_act,
        rewards=off_rew,
        terminals=off_term,
    )

    buffer = d3rlpy.dataset.MixedReplayBuffer(
        primary_replay_buffer=online_buffer,
        secondary_replay_buffer=offline_mdp,
        secondary_mix_ratio=args.mix_ratio,
    )

    # Seed the online buffer with a few offline episodes so it has transitions
    seed_episodes = list(offline_mdp.episodes)[:20]
    for ep in seed_episodes:
        online_buffer.append_episode(ep)
    print(f"Pre-seeded online buffer with {len(seed_episodes)} episodes "
          f"({online_buffer.transition_count} transitions).")

    print(f"MixedReplayBuffer: {args.mix_ratio*100:.0f}% offline / "
          f"{(1-args.mix_ratio)*100:.0f}% online per batch")

    # ── Explorer ────────────────────────────────────────────────────────────
    try:
        explorer = d3rlpy.algos.LinearDecayEpsilonGreedy(
            start_epsilon=0.10,
            end_epsilon=0.02,
            duration=args.finetune_steps
        )
    except AttributeError:
        explorer = None

    fit_args = {
        "env": env,
        "buffer": buffer,
        "n_steps": args.finetune_steps,
        "n_steps_per_epoch": 1000,
        "random_steps": 1000,
        "experiment_name": "CQL_Online_Finetune",
        "logger_adapter": TensorboardAdapterFactory(root_dir=log_dir),
    }
    if explorer:
        fit_args["explorer"] = explorer

    print(f"Fine-tuning for {args.finetune_steps} steps...")
    cql.fit_online(**fit_args)

    env.close()

    # Save the fine-tuned model
    cql.save(model_path)
    print(f"\nFine-tuned model saved to {model_path}")


if __name__ == "__main__":
    main()
