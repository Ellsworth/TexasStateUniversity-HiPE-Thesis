import numpy as np
import d3rlpy
import argparse
import os
import time

import torch
from firebot_agent.gym_env import FireBotEnv
from firebot_agent.heatmap_wrapper import PositionHeatmapWrapper
from firebot_agent.log_master import FireBotLogger
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation


class GridObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shape = env.observation_space["local_grid"].shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(shape[1], shape[2]),  # (1, 65, 65) -> (65, 65)
            dtype=np.uint8
        )

    def observation(self, obs):
        return obs["local_grid"].squeeze(0)


def main():
    parser = argparse.ArgumentParser(description="DiscreteCQL Inference (no training)")
    parser.add_argument("--model", type=str, default="d3rlpy_logs/cql_model.pt",
                        help="Path to the trained model (.pt file)")
    parser.add_argument("--n-episodes", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--n-frames", type=int, default=4,
                        help="Number of frames to stack (must match training)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="ZMQ server host")
    parser.add_argument("--port", type=int, default=5555,
                        help="ZMQ server port")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock environment (no ZMQ)")
    parser.add_argument("--record-data", action="store_true",
                        help="Record transitions to an offline dataset")
    parser.add_argument("--no-save-heatmap", dest="save_heatmap", action="store_false",
                        help="Disable saving the position heatmap (enabled by default)")
    parser.set_defaults(save_heatmap=True)

    args = parser.parse_args()

    # ── GPU info ──────────────────────────────────────────────────────────────
    print("=" * 60)
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"GPUs available: {n_gpus}")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available. Using CPU.")
    print("=" * 60)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found at: {args.model}")

    # ── Create a timestamped log directory for this inference run ─────────────
    logger = FireBotLogger(base_dir="logs", experiment_name="CQL_Infer")
    log_dir = logger.get_log_dir()

    output_file = os.path.join(log_dir, "inference_recording.npz")
    if args.record_data:
        print(f"Recording will be saved to: {output_file}")

    # ── Environment (must be built before loading weights) ────────────────────
    env = FireBotEnv(
        ip=args.host,
        port=args.port,
        discrete_actions=True,
        mock=args.mock,
        agent_name="CQL_Infer",
        record_data=args.record_data,
        output_file=output_file,
    )

    if args.save_heatmap:
        env = PositionHeatmapWrapper(env, save_every=1000, save_dir=log_dir)

    env = GridObservationWrapper(env)

    if args.n_frames > 1:
        print(f"Stacking {args.n_frames} frames...")
        env = FrameStackObservation(env, stack_size=args.n_frames)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Loading model from {args.model} ...")
    cql = d3rlpy.load_learnable(args.model, device=torch.cuda.is_available())
    print("Model loaded successfully.")
    print("=" * 60)

    # ── Inference loop ────────────────────────────────────────────────────────
    print(f"\nRunning {args.n_episodes} episode(s) in inference mode...\n")

    total_rewards = []

    for episode in range(1, args.n_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0.0
        step = 0
        done = False

        print(f"── Episode {episode} ──────────────────────────────────────")

        while not done:
            # predict() returns a numpy array of actions; take the first element
            action = cql.predict(np.expand_dims(obs, axis=0))[0]

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            done = terminated or truncated

            if step % 100 == 0:
                print(f"  Step {step:5d} | reward so far: {episode_reward:.3f} | {info}")

        print(f"  Episode {episode} finished after {step} steps | total reward: {episode_reward:.3f}\n")
        total_rewards.append(episode_reward)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Ran {args.n_episodes} episode(s)")
    print(f"  Mean reward : {np.mean(total_rewards):.3f}")
    print(f"  Std  reward : {np.std(total_rewards):.3f}")
    print(f"  Min  reward : {np.min(total_rewards):.3f}")
    print(f"  Max  reward : {np.max(total_rewards):.3f}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
