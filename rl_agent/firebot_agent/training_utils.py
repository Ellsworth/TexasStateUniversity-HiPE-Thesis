import numpy as np
import os

import torch
import gymnasium as gym
import d3rlpy
from d3rlpy.logging import TensorboardAdapterFactory
from firebot_agent.utils import convert_continuous_to_discrete


def print_gpu_info():
    """Print available GPU information."""
    print("=" * 60)
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"GPUs available: {n_gpus}")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available. Using CPU.")
    print("=" * 60)


def load_dataset(path):
    """Load and concatenate one or more .npz dataset files.

    Returns:
        Tuple of (observations, actions, rewards, terminals) as numpy arrays.
    """
    if os.path.isdir(path):
        files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.npz')])
        if not files:
            raise ValueError(f"No .npz files found in {path}")
        print(f"Found {len(files)} .npz files in {path}")
    else:
        if not os.path.exists(path):
            raise ValueError(f"Dataset not found at {path}")
        files = [path]

    all_observations = []
    all_actions = []
    all_rewards = []
    all_terminals = []

    obs_shape = None

    for f in files:
        print(f"Loading {f}...")
        data = np.load(f)

        if 'obs_local_grid' in data:
            obs = data['obs_local_grid']
        elif 'observations' in data:
            obs = data['observations']
        else:
            raise ValueError(f"Could not find observations in {f}")

        # Validation
        if obs_shape is None:
            obs_shape = obs.shape[1:]
        elif obs.shape[1:] != obs_shape:
            raise ValueError(f"Shape mismatch in {f}: expected {obs_shape}, got {obs.shape[1:]}")

        all_observations.append(obs)
        all_actions.append(data['actions'])
        all_rewards.append(data['rewards'])
        all_terminals.append(data['terminals'])

    # Concatenate
    observations = np.concatenate(all_observations, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)
    terminals = np.concatenate(all_terminals, axis=0)

    return observations, actions, rewards, terminals

def get_evaluators():
    """Return the standard evaluator dict for CQL training."""
    return {
        "value_scale": d3rlpy.metrics.AverageValueEstimationEvaluator(),
        "td_error": d3rlpy.metrics.TDErrorEvaluator()
    }


def prestack_frames(obs, terminals, n_frames):
    """Convert (N, 1, H, W) raw frames to (N, n_frames, H, W).

    Episode boundaries (terminals) prevent frames from bleeding across
    episodes; the first n_frames-1 steps repeat the episode's first frame.
    """
    N, _, H, W = obs.shape
    out = np.zeros((N, n_frames, H, W), dtype=obs.dtype)
    ep_start = 0
    for i in range(N):
        for f in range(n_frames):
            src = max(ep_start, i - (n_frames - 1 - f))
            out[i, f] = obs[src, 0]  # squeeze channel dim
        if terminals[i]:
            ep_start = i + 1
    return out


class GridObservationWrapper(gym.ObservationWrapper):
    """Extracts local_grid from Dict obs and squeezes the channel dimension.

    Converts (1, H, W) -> (H, W) so FrameStackObservation can produce
    (stack_size, H, W).
    """
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

