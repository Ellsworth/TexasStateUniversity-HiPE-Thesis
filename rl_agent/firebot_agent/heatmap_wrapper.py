import os
import glob

import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

try:
    from .extract_tiles import extract_tiles
except ImportError:
    try:
        from extract_tiles import extract_tiles
    except ImportError:
        extract_tiles = None


class PositionHeatmapWrapper(gym.Wrapper):
    """Wrapper that collects agent (x, y) positions and saves heatmap PNGs."""
    def __init__(self, env, log_root="logs/runs", experiment_name="CQL_Online_Finetune",
                 save_every=5000, save_dir=None):
        super().__init__(env)
        self.log_root = log_root
        self.experiment_name = experiment_name
        self.save_every = save_every
        self.positions = []
        self.step_count = 0
        self._run_dir = None
        # If an explicit directory is given, use it directly
        self._explicit_save_dir = save_dir
        
        self.tiles = []
        if extract_tiles is not None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            sdf_path = os.path.join(project_root, "ros2_ws", "src", "firebot_rl", "assets", "world-test.sdf")
            if os.path.exists(sdf_path):
                self.tiles = extract_tiles(sdf_path)


    def _get_run_dir(self):
        """Find the latest run directory matching our experiment name."""
        if self._run_dir is None:
            pattern = os.path.join(self.log_root, f"{self.experiment_name}_*")
            matches = sorted(glob.glob(pattern))
            if matches:
                self._run_dir = matches[-1]
        return self._run_dir

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.positions.append((info.get("agent_x", 0.0), info.get("agent_y", 0.0)))
        self.step_count += 1
        if self.step_count % self.save_every == 0:
            self._save_heatmap()
        return obs, reward, term, trunc, info

    def _save_heatmap(self):
        if self._explicit_save_dir is not None:
            run_dir = self._explicit_save_dir
        else:
            run_dir = self._get_run_dir()
            if run_dir is None:
                run_dir = "heatmaps"
        os.makedirs(run_dir, exist_ok=True)

        pos = np.array(self.positions)
        heatmap, xedges, yedges = np.histogram2d(
            pos[:, 0], pos[:, 1], bins=50
        )
        heatmap = gaussian_filter(heatmap, sigma=2)

        plt.figure()
        plt.imshow(heatmap.T, origin='lower', cmap='hot',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        
        if self.tiles:
            tile_xs = [t['x'] for t in self.tiles]
            tile_ys = [t['y'] for t in self.tiles]
            plt.scatter(tile_xs, tile_ys, color='blue', marker='x', s=30, label='Tiles', alpha=0.7)
            
        plt.colorbar(label='Visit Density')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(f'Position Heatmap (step {self.step_count})')
        if self.tiles:
            plt.legend(loc='upper right', prop={'size': 8})
        plt.savefig(os.path.join(run_dir, f'heatmap_{self.step_count}.png'), dpi=150)
        plt.close()
        self.positions.clear()
