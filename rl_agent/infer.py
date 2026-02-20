import numpy as np
import d3rlpy
import argparse
import os
import time
from datetime import datetime

import torch
from firebot_agent.gym_env import FireBotEnv
from firebot_agent.heatmap_wrapper import PositionHeatmapWrapper
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
    parser.add_argument("--model", type=str, default="d3rlpy_logs/cql_model.d3",
                        help="Path to the trained model (.d3 file)")
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

    # ── Recording path (mirrors teleop naming convention) ─────────────────────
    recording_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
    os.makedirs(recording_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(recording_dir, f"inference_{timestamp}.npz")
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
        env = PositionHeatmapWrapper(env, save_every=1000)

    env = GridObservationWrapper(env)

    if args.n_frames > 1:
        print(f"Stacking {args.n_frames} frames...")
        env = FrameStackObservation(env, stack_size=args.n_frames)

    # ── Model (built against env so architecture matches saved weights) ───────
    print(f"Loading model from {args.model} ...")
    cql = d3rlpy.algos.DiscreteCQLConfig(
        learning_rate=3e-4,
        batch_size=64,
        target_update_interval=100,
        alpha=1.0,
    ).create(device=torch.cuda.is_available())
    cql.build_with_env(env)
    cql.load_model(args.model)
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
