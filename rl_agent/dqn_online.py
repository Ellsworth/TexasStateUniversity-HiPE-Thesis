import dataclasses
import numpy as np
import d3rlpy
import argparse
import os

import torch
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from d3rlpy.logging import TensorboardAdapterFactory

from firebot_agent.gym_env import FireBotEnv
from firebot_agent.heatmap_wrapper import PositionHeatmapWrapper
from firebot_agent.reward_tracker_wrapper import RewardTrackerWrapper
from firebot_agent.log_master import FireBotLogger
from firebot_agent.training_utils import (
    print_gpu_info,
    GridObservationWrapper,
)

def create_dqn(device=None):
    """Create a DQN algorithm with standard hyperparameters.

    Args:
        device: Device to use. If None, auto-detects CUDA.

    Returns:
        tuple: A DQN instance and a dictionary of hyperparameters.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    hyperparams = {
        "learning_rate": 1e-4,
        "batch_size": 256,
        "target_update_interval": 500, # Keep this high for stability
        "gamma": 0.99,
    }

    dqn = d3rlpy.algos.DQNConfig(
        learning_rate=hyperparams["learning_rate"],
        batch_size=hyperparams["batch_size"],
        target_update_interval=hyperparams["target_update_interval"],
        gamma=hyperparams["gamma"],
        
        # Preprocessing
        observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(),
        reward_scaler=d3rlpy.preprocessing.MinMaxRewardScaler(
            minimum=-20.0,
            maximum=15.0
        ),
    ).create(device=device)
    
    return dqn, hyperparams


def main():
    parser = argparse.ArgumentParser(description="DQN Online Training")
    parser.add_argument("--train-steps", type=int, default=1000000, help="Number of online training steps")
    parser.add_argument("--n-frames", type=int, default=4, help="Number of frames to stack")
    parser.add_argument("--buffer-size", type=int, default=200000, help="Replay buffer size for online training")
    parser.add_argument("--mock", action="store_true", help="Use mock environment (no ZMQ)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="ZMQ server host")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ server port")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional: Path to pretrained .pt model file to resume from")

    args = parser.parse_args()

    # Create a single timestamped log directory for this run
    logger = FireBotLogger(base_dir="logs", experiment_name="DQN_Online")
    log_dir = logger.get_log_dir()

    # Output model path
    model_path = os.path.join(log_dir, "dqn_online.d3")

    print_gpu_info()

    # ── Create environment ──────────────────────────────────────────────────
    online_dataset_path = os.path.join(log_dir, "online_dataset.npz")
    env = FireBotEnv(
        ip=args.host,
        port=args.port,
        discrete_actions=True,
        mock=args.mock,
        agent_name="DQN_Agent",
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

    # ── Initialization ───────────────────────────────────────────────
    print("=" * 60)
    print("ONLINE TRAINING WITH DQN")
    print("=" * 60)

    if args.checkpoint:
        dqn = d3rlpy.load_learnable(args.checkpoint, device=torch.cuda.is_available())
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        dqn, hyperparams = create_dqn()
        print("Initialized new DQN agent")

    # Create online FIFO buffer
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(
        limit=args.buffer_size,
        env=env,
    )

    # ── Explorer ────────────────────────────────────────────────────────────
    try:
        explorer = d3rlpy.algos.LinearDecayEpsilonGreedy(
            start_epsilon=1.0,
            end_epsilon=0.1,
            duration=int(args.train_steps * 0.8)
        )
    except AttributeError:
        # Fallback if explorer is structured differently in this d3rlpy version
        explorer = None

    fit_args = {
        "env": env,
        "buffer": buffer,
        "n_steps": args.train_steps,
        "n_steps_per_epoch": 1000,
        "random_steps": 10000,
        "experiment_name": "DQN_Online_Training",
        "logger_adapter": TensorboardAdapterFactory(root_dir=log_dir),
    }
    if explorer:
        fit_args["explorer"] = explorer

    print(f"Training for {args.train_steps} steps...")
    dqn.fit_online(**fit_args)

    env.close()

    # Save the trained model
    dqn.save(model_path)
    print(f"\nTrained model saved to {model_path}")


if __name__ == "__main__":
    main()
