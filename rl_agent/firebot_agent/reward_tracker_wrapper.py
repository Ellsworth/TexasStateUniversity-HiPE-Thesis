import os

import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter


class RewardTrackerWrapper(gym.Wrapper):
    """Gymnasium wrapper that logs episode reward statistics to TensorBoard.

    Tracked metrics (written at each episode boundary):
      - episode/mean_reward   - mean per-step reward for the episode
      - episode/total_reward  - cumulative reward (return) for the episode
      - episode/length        - number of steps in the episode

    The x-axis for all scalars is the global environment step count.
    """

    def __init__(self, env: gym.Env, log_dir: str):
        super().__init__(env)
        tb_dir = os.path.join(log_dir, "reward_tracker")
        self._writer = SummaryWriter(log_dir=tb_dir)

        self._episode_reward = 0.0
        self._episode_length = 0
        self._global_step = 0
        self._episode_count = 0

    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._episode_reward = 0.0
        self._episode_length = 0
        return obs, info

    # ------------------------------------------------------------------
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._episode_reward += float(reward)
        self._episode_length += 1
        self._global_step += 1

        if terminated or truncated:
            mean_reward = (
                self._episode_reward / self._episode_length
                if self._episode_length > 0
                else 0.0
            )
            self._episode_count += 1

            self._writer.add_scalar(
                "episode/mean_reward", mean_reward, self._global_step
            )
            self._writer.add_scalar(
                "episode/total_reward", self._episode_reward, self._global_step
            )
            self._writer.add_scalar(
                "episode/length", self._episode_length, self._global_step
            )
            self._writer.flush()

            print(
                f"[RewardTracker] Episode {self._episode_count} | "
                f"steps={self._episode_length} | "
                f"total_reward={self._episode_reward:.2f} | "
                f"mean_reward={mean_reward:.4f}"
            )

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def close(self):
        self._writer.close()
        super().close()
