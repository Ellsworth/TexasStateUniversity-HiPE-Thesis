import gymnasium as gym
from gymnasium import spaces
import zmq
import numpy as np
import msgpack
import msgpack_numpy as m
import csv
import os
import math
from scipy.ndimage import rotate as ndimage_rotate
from .offline_dataset_recorder import OfflineDataCollector
from .rl_zmq_client import RLZmqClient

# Patch msgpack to automatically handle numpy arrays
m.patch()

# Discrete Action Map: [linear_x, angular_z]
DISCRETE_ACTION_MAP = {
    0: [0.0, 0.0],   # Stop
    1: [1.0, 0.0],   # Forward
    2: [-1.0, 0.0],  # Backward
    3: [0.0, 1.0],   # Spin Left
    4: [0.0, -1.0],  # Spin Right
    5: [0.5, 0.5],   # Curve Left
    6: [0.5, -0.5]   # Curve Right
}

# Visited-map constants — must match local occupancy grid pixel scale
VISITED_MAP_RESOLUTION = 0.25  # metres per pixel
VISITED_MAP_SIZE       = 65    # pixels (width = height)

class FireBotEnv(gym.Env):
    """
    Custom Gymnasium Environment for the FireBot robot via ZMQ.
    """
    metadata = {"render_modes": [], "render_fps": 10}

    def __init__(self, ip="127.0.0.1", port=5555, max_episode_steps=1000, discrete_actions=False, mock=False, agent_name="unknown", record_data=False, output_file="offline_dataset.npz"):
        super().__init__()
        
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.discrete_actions = discrete_actions
        self.mock = mock
        self.agent_name = agent_name
        self.record_data = record_data
        self.output_file = output_file
        
        if self.record_data:
            self.collector = OfflineDataCollector()
            
        # Hysteresis for wall contact (noisy sensor)
        self.collision_active = False
        self.steps_since_contact = 0
        
        # Two-tier stuck detection
        self.soft_stuck_window = 50     # Steps to look back for soft-stuck check
        self.hard_stuck_limit = 150    # Consecutive soft-stuck steps before hard-stuck termination
        self.stuck_threshold = 0.3     # Meters; displacement below this = stuck
        self.position_history = []     # List of (x, y) tuples
        self.steps_stuck = 0           # Consecutive steps the agent has been soft-stuck

        # Spatial exploration grid (2m x 2m cells)
        self.cell_size = 2.0           # Meters per grid cell
        self.visited_cells = set()     # Set of (cell_x, cell_y) tuples

        # Breadcrumb reward system
        self.breadcrumbs = self._load_breadcrumbs()
        self.claimed_breadcrumbs = set()
        
        # 1. Initialize ZMQ
        if not self.mock:
            self.client = RLZmqClient(ip=ip, port=port)
        else:
            print("Running in MOCK mode. No ZMQ connection.")
        
        # 2. Define Action Space
        if self.discrete_actions:
            # Discrete Actions:
            # 0: Stop
            # 1: Forward
            # 2: Backward
            # 3: Left
            # 4: Right
            # 5: Forward Left
            # 6: Forward Right
            self.action_space = spaces.Discrete(7)
            
            # Action Map: [linear_x, angular_z]
            # Action Map: [linear_x, angular_z]
            self.action_map = DISCRETE_ACTION_MAP
        else:
            # Continuous Actions
            # cmd_vel: [linear_x, angular_z]
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32), 
                high=np.array([1.0, 1.0], dtype=np.float32), 
                dtype=np.float32
            )

        # 3. Define Observation Space
        # We have:
        # - local_grid: 65x65 int16
        # - wall_contact: float (0.0 or 1.0) — hitting a wall/obstacle
        # ground_contact (string) is passed via the info dict, not the obs space
        self.observation_space = spaces.Dict({
            "local_grid": spaces.Box(
                low=0, high=255, 
                shape=(1, 65, 65), dtype=np.uint8
            ),
            "wall_contact": spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32
            ),
            "visited_map": spaces.Box(
                low=0, high=255,
                shape=(1, VISITED_MAP_SIZE, VISITED_MAP_SIZE), dtype=np.uint8
            ),
        })

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.current_step = 0
        # Initialize last_action for smoothness calculation
        if self.discrete_actions:
             self.last_action = np.zeros(2, dtype=np.float32)
        else:
             self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
             
        # Hysteresis for wall contact (noisy sensor)
        self.collision_active = False
        self.steps_since_contact = 0
        
        # Clear stuck detection history
        self.position_history = []
        self.steps_stuck = 0

        # Reset exploration grid
        self.visited_cells = set()

        # Reset breadcrumb claims so they can be collected again
        self.claimed_breadcrumbs = set()
        
        # visited_rooms is cleared AFTER we get the reset response below,
        # so we can pre-populate it with the spawn room.  Clearing it here
        # causes a race: if the first response still reports the old room,
        # the very first step fires a false +500 new-room bonus.
        
        if self.mock:
            self.visited_rooms = set()
            return self._get_mock_observation(), {"image": []}

        # Send reset command to ZMQ bridge
        try:
            data = self.client.step(cmd_vel=[0.0, 0.0], steps=100, reset=True)
            
            # Seed visited_rooms with wherever the agent actually spawned so
            # the first step never triggers a false new-room bonus.
            # ground_contact may be a comma-separated list (agent straddling tiles),
            # so split and store individual tile names.
            spawn_ground = data.get("ground_contact", "")
            self.visited_rooms = {r.strip() for r in spawn_ground.split(",") if r.strip()}
            
            info = {
                "ground_contact": spawn_ground,
                "image": data.get("image", [])
            }
            
            return self._process_observation(data), info
            
        except Exception as e:
            print(f"Error during reset: {e}")
            # Return empty/safe observation in case of failure? 
            # Or raise? Raising is probably better for debugging.
            raise e

    def step(self, action):
        # Prepare payload
        
        if self.discrete_actions:
            # Map discrete action to velocity command
            cmd_vel = self.action_map.get(int(action), [0.0, 0.0])
        else:
            cmd_vel = action
            
        if self.mock:
            self.current_step += 1
            terminated = False
            truncated = self.current_step >= self.max_episode_steps
            observation = self._get_mock_observation()
            reward = 0.0
            info = {"image": []}
        else:
            data = self.client.step(cmd_vel=cmd_vel, steps=100, reset=False)
            
            observation = self._process_observation(data)
            
            # --- Two-tier stuck detection ---
            agent_x = float(data.get("agent_x", 0.0))
            agent_y = float(data.get("agent_y", 0.0))
            self.position_history.append((agent_x, agent_y))
            soft_stuck = False
            hard_stuck = False
            if len(self.position_history) >= self.soft_stuck_window:
                self.position_history = self.position_history[-self.soft_stuck_window:]
                oldest_x, oldest_y = self.position_history[0]
                displacement = np.sqrt((agent_x - oldest_x)**2 + (agent_y - oldest_y)**2)
                if displacement < self.stuck_threshold:
                    self.steps_stuck += 1
                    soft_stuck = True
                    if self.steps_stuck >= self.hard_stuck_limit:
                        hard_stuck = True
                        print(f"[FireBotEnv] HARD STUCK (displacement={displacement:.3f}m, "
                              f"stuck for {self.steps_stuck} steps). Terminating episode.")
                    elif self.steps_stuck % 25 == 0:
                        print(f"[FireBotEnv] Soft stuck (displacement={displacement:.3f}m, "
                              f"stuck for {self.steps_stuck} steps, "
                              f"penalty={-0.5 * self.steps_stuck:.1f})")
                else:
                    self.steps_stuck = 0
            
            reward = self._calculate_reward(data, cmd_vel, soft_stuck=soft_stuck, hard_stuck=hard_stuck)
            
            # Update state needed for next step
            if not isinstance(cmd_vel, np.ndarray):
                cmd_vel = np.array(cmd_vel, dtype=np.float32)

            # Update state for next step
            self.last_action = cmd_vel.copy()
            self.current_step += 1
            
            terminated = hard_stuck  # Only terminate on hard stuck (truly wedged)
            truncated = self.current_step >= self.max_episode_steps
            info = {
                "ground_contact": data.get("ground_contact", ""),
                "wall_contact": bool(data.get("wall_contact", False)),
                "agent_x": float(data.get("agent_x", 0.0)),
                "agent_y": float(data.get("agent_y", 0.0)),
                "agent_z": float(data.get("agent_z", 0.0)),
                "agent_yaw": float(data.get("agent_yaw", 0.0)),
                "image": data.get("image", [])
            }

        # Record step if enabled
        if self.record_data:
             self.collector.add_step(observation, action, reward, terminated or truncated)

        return observation, reward, terminated, truncated, info

    def _get_mock_observation(self):
        """Generate random observation for testing."""
        return {
            "local_grid": np.random.randint(0, 255, (1, 65, 65), dtype=np.uint8),
            "wall_contact": np.array([0.0], dtype=np.float32),
            "visited_map": np.zeros((1, VISITED_MAP_SIZE, VISITED_MAP_SIZE), dtype=np.uint8),
        }

    def _process_observation(self, data):
        """Extract and format observation from ZMQ response."""
        # data["observation"] is the grid
        local_grid = data.get("observation", np.zeros((65,65), dtype=np.int16))
        if not isinstance(local_grid, np.ndarray):
             local_grid = np.array(local_grid, dtype=np.int16)
        
        # Ensure it's the right shape/type just in case
        if local_grid.shape != (65, 65):
             local_grid = np.zeros((65,65), dtype=np.int16)
        
        # Prepare uint8 grid for CNN
        # Map [-1, 100] -> [0, 255]
        # -1 (Unknown) -> 127
        # 0 (Free) -> 0
        # 100 (Occupied) -> 255
        
        # Create output array initialized to 127 (unknown)
        processed_grid = np.full(local_grid.shape, 127, dtype=np.uint8)
        
        # Mask for free space (0)
        mask_free = (local_grid == 0)
        processed_grid[mask_free] = 0
        
        # Mask for occupied (>0)
        # We also handle the unlikely case of negative values other than -1 by treating them as unknown (already 127)
        mask_occupied = (local_grid > 0)
        # Scale 1-100 to 2-255 roughly, or just clamp.
        # Simple linear scaling: val * 2.55
        processed_grid[mask_occupied] = np.clip(local_grid[mask_occupied] * 2.55, 0, 255).astype(np.uint8)

        wall_contact = np.array([1.0 if data.get("wall_contact", False) else 0.0], dtype=np.float32)

        agent_x = float(data.get("agent_x", 0.0))
        agent_y = float(data.get("agent_y", 0.0))
        agent_yaw = float(data.get("agent_yaw", 0.0))  # radians

        # NOTE: local_grid_window.py already rotates the grid by (-yaw - π/2)
        # so forward is up in the published grid. No further rotation needed here.

        # Add channel dimension: (65, 65) -> (1, 65, 65)
        local_grid = np.expand_dims(processed_grid, axis=0)

        visited_map = self._build_visited_map(agent_x, agent_y, agent_yaw)

        return {
            "local_grid": local_grid,
            "wall_contact": wall_contact,
            "visited_map": visited_map,
        }

    def _build_visited_map(self, agent_x: float, agent_y: float, agent_yaw: float = 0.0) -> np.ndarray:
        """Return a (1, 65, 65) uint8 array at 0.25 m/px centred on the agent.
        Each 2 m visited cell is rendered as an 8x8 pixel filled block.
        The map is rotated so the agent's heading points up (same frame as
        the rotated occupancy grid).
        Visited = 255, unvisited = 0.
        """
        px_per_m    = 1.0 / VISITED_MAP_RESOLUTION          # 4 px / m
        px_per_cell = int(self.cell_size * px_per_m)        # 8 px / cell
        half_px     = VISITED_MAP_SIZE // 2                 # 32

        # Rotation matrix to match local_grid_window.py frame: rotate by (-yaw - π/2)
        angle = agent_yaw - (math.pi / 2)
        cos_h = math.cos(angle)
        sin_h = math.sin(angle)

        canvas = np.zeros((VISITED_MAP_SIZE, VISITED_MAP_SIZE), dtype=np.uint8)

        for (cx, cy) in self.visited_cells:
            # World-space offset of cell centre from agent
            wx0 = cx * self.cell_size
            wy0 = cy * self.cell_size
            dx = (wx0 - agent_x) * px_per_m  # pixels, world frame
            dy = (wy0 - agent_y) * px_per_m

            # Rotate into agent-heading frame (yaw = 0 → forward = +col)
            rdx = cos_h * dx - sin_h * dy
            rdy = sin_h * dx + cos_h * dy

            col0 = int(rdx) + half_px
            row0 = int(rdy) + half_px
            col1 = col0 + px_per_cell
            row1 = row0 + px_per_cell

            # Clamp to canvas bounds
            col0c = max(col0, 0);  col1c = min(col1, VISITED_MAP_SIZE)
            row0c = max(row0, 0);  row1c = min(row1, VISITED_MAP_SIZE)
            if col0c < col1c and row0c < row1c:
                canvas[row0c:row1c, col0c:col1c] = 255

        return np.expand_dims(canvas, axis=0)  # (1, 65, 65)

    def _load_breadcrumbs(self):
        """Load breadcrumb waypoints from breadcrumbs.csv in the rl_agent directory."""
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "breadcrumbs.csv"
        )
        breadcrumbs = []
        if not os.path.exists(csv_path):
            print(f"[FireBotEnv] WARNING: breadcrumbs.csv not found at {csv_path}")
            return breadcrumbs
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                breadcrumbs.append({
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                    "size": float(row["size"]),
                })
        print(f"[FireBotEnv] Loaded {len(breadcrumbs)} breadcrumbs from {csv_path}")
        return breadcrumbs

    def _calculate_reward(self, data, action, soft_stuck: bool = False, hard_stuck: bool = False):
        wall_contact = data.get("wall_contact", False)
        
        if wall_contact:
            self.steps_since_contact = 0
            if not self.collision_active:
                self.collision_active = True
        else:
            self.steps_since_contact += 1
            if self.steps_since_contact > 5:
                self.collision_active = False
                
        # --- SCALED PENALTIES ---
        # Reduced from -10.0 to -1.0 to keep it in line with exploration gains
        collision_penalty = -1.0 if self.collision_active else 0.0

        # Constant pressure to move
        time_penalty = -0.01

        if hard_stuck:
            stuck_penalty = -5.0  # Reduced from -100.0 (prevents gradient explosion)
        elif soft_stuck:
            stuck_penalty = -0.1 * self.steps_stuck # Smoother escalation
        else:
            stuck_penalty = 0.0

        # --- SCALED REWARDS ---
        agent_x = float(data.get("agent_x", 0.0))
        agent_y = float(data.get("agent_y", 0.0))
        cell = (int(agent_x // self.cell_size), int(agent_y // self.cell_size))
        
        exploration_reward = 0.0
        if cell not in self.visited_cells:
            self.visited_cells.add(cell)
            exploration_reward = 0.5

        new_room_bonus = 0.0
        ground = data.get("ground_contact", "")
        if ground:
            tiles = [r.strip() for r in ground.split(",") if r.strip()]
            for tile in tiles:
                if tile.startswith("tile_") and tile not in self.visited_rooms:
                    self.visited_rooms.add(tile)
                    # Reduced from 500.0 to 10.0. 
                    # This is still 100x the time penalty, making it the "primary goal."
                    new_room_bonus += 10.0 

        breadcrumb_reward = 0.0
        for idx, bc in enumerate(self.breadcrumbs):
            if idx in self.claimed_breadcrumbs: continue
            dist = np.sqrt((agent_x - bc["x"])**2 + (agent_y - bc["y"])**2)
            if dist <= bc["size"]:
                self.claimed_breadcrumbs.add(idx)
                breadcrumb_reward += 2.0 # Reduced from 50.0

        # Small incentive to move forwards to offset the time penalty and prefer forward exploration
        forward_bonus = 0.02 * max(0.0, float(action[0]))

        return (time_penalty + exploration_reward + stuck_penalty + 
                collision_penalty + new_room_bonus + breadcrumb_reward + forward_bonus)
    
    def close(self):
        if self.record_data:
            self.collector.save(self.output_file)
            
        if not self.mock:
            if self.client.socket:
                self.client.socket.close()
