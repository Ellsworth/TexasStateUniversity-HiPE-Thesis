import gymnasium as gym
from gymnasium import spaces
import zmq
import numpy as np
import msgpack
import msgpack_numpy as m
import csv
import os
from .offline_dataset_recorder import OfflineDataCollector

# Patch msgpack to automatically handle numpy arrays
m.patch()

# Discrete Action Map: [linear_x, angular_z]
DISCRETE_ACTION_MAP = {
    0: [0.0, 0.0],
    1: [1.0, 0.0],
    2: [-1.0, 0.0],
    3: [0.0, 1.0],   # Spin Left
    4: [0.0, -1.0],  # Spin Right
    5: [0.5, 0.5],   # Curve Left
    6: [0.5, -0.5]   # Curve Right
}

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
        
        # Stuck detection
        self.stuck_window = 100           # Steps to look back (halved for faster recovery)
        self.stuck_threshold = 0.3     # Meters; total displacement below this = stuck
        self.position_history = []      # List of (x, y) tuples

        # Breadcrumb reward system
        self.breadcrumbs = self._load_breadcrumbs()
        self.claimed_breadcrumbs = set()
        
        # 1. Initialize ZMQ
        if not self.mock:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(f"tcp://{ip}:{port}")
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
            )
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

        # Reset breadcrumb claims so they can be collected again
        self.claimed_breadcrumbs = set()
        
        # visited_rooms is cleared AFTER we get the reset response below,
        # so we can pre-populate it with the spawn room.  Clearing it here
        # causes a race: if the first response still reports the old room,
        # the very first step fires a false +500 new-room bonus.
        
        if self.mock:
            self.visited_rooms = set()
            return self._get_mock_observation(), {}

        # Send reset command to ZMQ bridge
        payload = {
            "reset": True,
            "command": "step"
        }
        
        try:
            self.socket.send(msgpack.packb(payload))
            message = self.socket.recv()
            data = msgpack.unpackb(message)
            
            # Seed visited_rooms with wherever the agent actually spawned so
            # the first step never triggers a false new-room bonus.
            # ground_contact may be a comma-separated list (agent straddling tiles),
            # so split and store individual tile names.
            spawn_ground = data.get("ground_contact", "")
            self.visited_rooms = {r.strip() for r in spawn_ground.split(",") if r.strip()}
            
            return self._process_observation(data), {"ground_contact": spawn_ground}
            
        except zmq.ZMQError as e:
            print(f"ZMQ Error during reset: {e}")
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
            info = {}
        else:
            payload = {
                "command": "step",
                "step": 100, # 1 step = 0.001s
                "cmd_vel": cmd_vel,
                "reset": False
            }

            self.socket.send(msgpack.packb(payload))
            message = self.socket.recv()
            data = msgpack.unpackb(message)
            
            observation = self._process_observation(data)
            
            # --- Stuck detection ---
            agent_x = float(data.get("agent_x", 0.0))
            agent_y = float(data.get("agent_y", 0.0))
            self.position_history.append((agent_x, agent_y))
            stuck = False
            if len(self.position_history) >= self.stuck_window:
                self.position_history = self.position_history[-self.stuck_window:]
                oldest_x, oldest_y = self.position_history[0]
                displacement = np.sqrt((agent_x - oldest_x)**2 + (agent_y - oldest_y)**2)
                if displacement < self.stuck_threshold:
                    stuck = True
                    print(f"[FireBotEnv] Agent stuck (displacement={displacement:.3f}m over {self.stuck_window} steps). Resetting simulation.")
                    # Proactively reset the simulation NOW so we don't wait for the
                    # training loop to call env.reset() — and so subsequent steps
                    # don't repeatedly fire the stuck flag.
                    reset_payload = {"reset": True, "command": "step"}
                    self.socket.send(msgpack.packb(reset_payload))
                    self.socket.recv()  # Drain the response
                    # Clear all episode state so env.reset() is a no-op on ZMQ
                    self.position_history = []
                    self.current_step = 0
                    self.collision_active = False
                    self.steps_since_contact = 0
            
            reward = self._calculate_reward(data, cmd_vel, stuck=stuck)
            
            # Update state needed for next step
            if not isinstance(cmd_vel, np.ndarray):
                cmd_vel = np.array(cmd_vel, dtype=np.float32)

            # Update state for next step
            self.last_action = cmd_vel.copy()
            self.current_step += 1
            
            terminated = stuck  # End episode if agent is stuck
            truncated = self.current_step >= self.max_episode_steps
            info = {
                "ground_contact": data.get("ground_contact", ""),
                "wall_contact": bool(data.get("wall_contact", False)),
                "agent_x": float(data.get("agent_x", 0.0)),
                "agent_y": float(data.get("agent_y", 0.0)),
                "agent_z": float(data.get("agent_z", 0.0)),
            }

        # Record step if enabled
        if self.record_data:
             self.collector.add_step(observation, action, reward, terminated or truncated)

        return observation, reward, terminated, truncated, info

    def _get_mock_observation(self):
        """Generate random observation for testing."""
        return {
            "local_grid": np.random.randint(0, 255, (1, 65, 65), dtype=np.uint8),
            "wall_contact": np.array([0.0], dtype=np.float32)
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

        # Add channel dimension: (65, 65) -> (1, 65, 65)
        local_grid = np.expand_dims(processed_grid, axis=0)

        wall_contact = np.array([1.0 if data.get("wall_contact", False) else 0.0], dtype=np.float32)

        return {
            "local_grid": local_grid,
            "wall_contact": wall_contact
        }

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

    def _calculate_reward(self, data, action, stuck: bool = False):
        linear_x = action[0]
        angular_z = action[1]
        wall_contact = data.get("wall_contact", False)
        
        if wall_contact:
            self.steps_since_contact = 0
            if not self.collision_active:
                self.collision_active = True
        else:
            self.steps_since_contact += 1
            if self.steps_since_contact > 5:
                self.collision_active = False
                
        # If collision is active (sustained or hysteresis), apply a constant penalty
        collision_penalty = -10.0 if self.collision_active else 0.0
        
        # Reward forward progress more aggressively
        if linear_x > 0.0:  # Going forwards
            vel_reward = linear_x * 1.0   # Increased reward multiplier for forward motion
        else:  # Stationary or going backwards
            vel_reward = linear_x * 2.0      # Heavy backward penalty to discourage reverse-loops

        # Tie survival to movement: forward motion = bonus, doing nothing = steep penalty.
        # The penalty is large enough that sitting still accumulates worse than a wall hit over time.
        if linear_x > 0.05:
            survival_reward = 0.1    # Meaningful bonus to incentivize forward motion
        elif abs(linear_x) < 0.05 and abs(angular_z) < 0.05:
            survival_reward = -0.5   # Heavy "do nothing" penalty — must exceed collision fear
        else:
            survival_reward = -0.05  # Small penalty for other low-speed states

        # Allow turning without additional penalty to encourage exploration when blocked.
        angular_penalty = 0.0

        # Penalize being stuck (no displacement over the tracking window)
        stuck_penalty = -100.0 if stuck else 0.0

        # New room exploration bonus — reward entering a room not yet visited this episode.
        # ground_contact may be a comma-separated list when straddling tiles, so split it
        # and reward each individual new tile. staging_area is excluded as the spawn zone.
        new_room_bonus = 0.0
        ground = data.get("ground_contact", "")
        if ground:
            tiles = [r.strip() for r in ground.split(",") if r.strip()]
            for tile in tiles:
                if tile.startswith("tile_") and tile not in self.visited_rooms:
                    self.visited_rooms.add(tile)
                    new_room_bonus += 500.0
                    print(f"[FireBotEnv] NEW ROOM: '{tile}' (+500.0) | Visited: {len(self.visited_rooms)} rooms")

        # Breadcrumb reward — one-time reward for touching each breadcrumb waypoint.
        breadcrumb_reward = 0.0
        agent_x = float(data.get("agent_x", 0.0))
        agent_y = float(data.get("agent_y", 0.0))
        agent_z = float(data.get("agent_z", 0.0))
        for idx, bc in enumerate(self.breadcrumbs):
            if idx in self.claimed_breadcrumbs:
                continue
            dist = np.sqrt(
                (agent_x - bc["x"]) ** 2 +
                (agent_y - bc["y"]) ** 2 +
                (agent_z - bc["z"]) ** 2
            )
            if dist <= bc["size"]:
                self.claimed_breadcrumbs.add(idx)
                breadcrumb_reward += 50.0
                print(f"[FireBotEnv] BREADCRUMB {idx} claimed (+50.0) at ({bc['x']}, {bc['y']}, {bc['z']}) | "
                      f"Claimed: {len(self.claimed_breadcrumbs)}/{len(self.breadcrumbs)}")

        return vel_reward + survival_reward + angular_penalty + stuck_penalty + collision_penalty + new_room_bonus + breadcrumb_reward

    def close(self):
        if self.record_data:
            self.collector.save(self.output_file)
            
        if not self.mock:
            self.socket.close()
            self.context.term()
