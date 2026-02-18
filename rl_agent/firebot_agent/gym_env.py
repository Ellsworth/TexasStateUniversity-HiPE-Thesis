import gymnasium as gym
from gymnasium import spaces
import zmq
import numpy as np
import msgpack
import msgpack_numpy as m
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
        # - wall_distance: float
        # - wall_angle: float
        # - wall_contact: float (0.0 or 1.0) — hitting a wall/obstacle
        # ground_contact (string) is passed via the info dict, not the obs space
        self.observation_space = spaces.Dict({
            "local_grid": spaces.Box(
                low=0, high=255, 
                shape=(1, 65, 65), dtype=np.uint8
            ),
            "wall_distance": spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([100.0], dtype=np.float32), # -1 is no wall
                dtype=np.float32
            ),
            "wall_angle": spaces.Box(
                low=np.array([-np.pi], dtype=np.float32),
                high=np.array([np.pi], dtype=np.float32),
                dtype=np.float32
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
        
        if self.mock:
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
            
            return self._process_observation(data), {"ground_contact": data.get("ground_contact", "")}
            
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
            reward = self._calculate_reward(data, cmd_vel) # Use cmd_vel (mapped action) for reward calc
            
            # Update state needed for next step
            if not isinstance(cmd_vel, np.ndarray):
                cmd_vel = np.array(cmd_vel, dtype=np.float32)

            # Update state for next step
            self.last_action = cmd_vel.copy()
            self.current_step += 1
            
            terminated = False # Defining termination logic is hard without specific tasks
            truncated = self.current_step >= self.max_episode_steps
            info = {
                "ground_contact": data.get("ground_contact", ""),
                "wall_contact": bool(data.get("wall_contact", False)),
                "agent_x": float(data.get("agent_x", 0.0)),
                "agent_y": float(data.get("agent_y", 0.0)),
            }

        # Record step if enabled
        if self.record_data:
             self.collector.add_step(observation, action, reward, terminated or truncated)

        return observation, reward, terminated, truncated, info

    def _get_mock_observation(self):
        """Generate random observation for testing."""
        return {
            "local_grid": np.random.randint(0, 255, (1, 65, 65), dtype=np.uint8),
            "wall_distance": np.array([np.random.uniform(0, 10)], dtype=np.float32),
            "wall_angle": np.array([np.random.uniform(-np.pi, np.pi)], dtype=np.float32),
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

        wall_dist = np.array([float(data.get("wall_distance", -1.0))], dtype=np.float32)
        wall_ang = np.array([float(data.get("wall_angle", 0.0))], dtype=np.float32)
        wall_contact = np.array([1.0 if data.get("wall_contact", False) else 0.0], dtype=np.float32)

        return {
            "local_grid": local_grid,
            "wall_distance": wall_dist,
            "wall_angle": wall_ang,
            "wall_contact": wall_contact
        }

    def _calculate_reward(self, data, action):
        linear_x = action[0]
        angular_z = action[1]
        wall_contact = data.get("wall_contact", False)
        
        if wall_contact:
            self.steps_since_contact = 0
            if not self.collision_active:
                self.collision_active = True
                return -10.0 # Impact penalty
        else:
            self.steps_since_contact += 1
            if self.steps_since_contact > 5:
                self.collision_active = False
                
        # If collision is active (sustained or hysteresis), deny all forward reward
        if self.collision_active:
            return -0.1
        
        # Reward forward progress more aggressively
        # Removing the > 0.1 gate so the agent feels the benefit of even slow movement
        vel_reward = linear_x * 0.5 
        
        # Tie survival to movement: You only get the bonus if you are actually moving
        # This kills the "sit and jitter" strategy
        survival_reward = 0.05 if linear_x > 0.05 else -0.05
        
        # Penalize high angular velocity to prevent spinning in circles
        angular_penalty = (angular_z**2) * 0.5
        
        return vel_reward + survival_reward - angular_penalty

    def close(self):
        if self.record_data:
            self.collector.save(self.output_file)
            
        if not self.mock:
            self.socket.close()
            self.context.term()
