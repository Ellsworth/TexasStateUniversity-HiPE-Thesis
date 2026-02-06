import numpy as np
import d3rlpy
import argparse
import os

from firebot_agent.utils import convert_continuous_to_discrete
from firebot_agent.gym_env import FireBotEnv
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation

class GridObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Squeeze the channel dimension to get (H, W) for FrameStack compatibility
        # FrameStackObservation will then create (stack_size, H, W)
        shape = env.observation_space["local_grid"].shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(shape[1], shape[2]), # Assuming (1, 65, 65) -> (65, 65)
            dtype=np.uint8
        )
    
    def observation(self, obs):
        # Extract local_grid and squeeze channel dim: (1, 65, 65) -> (65, 65)
        return obs["local_grid"].squeeze(0)

def main():
    parser = argparse.ArgumentParser(description="DQN Training (Offline & Online)")
    parser.add_argument("--dataset", type=str, default="offline_dataset.npz", help="Path to the offline dataset (npz)")
    parser.add_argument("--n-steps", type=int, default=1000000, help="Number of training steps")
    parser.add_argument("--save-path", type=str, default="d3rlpy_logs/dqn_model.d3", help="Path to save the trained model")
    parser.add_argument("--n-frames", type=int, default=4, help="Number of frames to stack")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Path to checkpoint to load for fine-tuning")
    
    # Online arguments
    parser.add_argument("--online", action="store_true", help="Enable online training mode")
    parser.add_argument("--buffer-size", type=int, default=200000, help="Replay buffer size for online training")
    parser.add_argument("--mock", action="store_true", help="Use mock environment (no ZMQ)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="ZMQ server host")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ server port")
    
    args = parser.parse_args()

    # Initialize DQN
    dqn = d3rlpy.algos.DQNConfig(
        learning_rate=3e-4,
        batch_size=64,
        target_update_interval=100,
    ).create(device=False)

    if args.online:
        print("Starting ONLINE training...")
        # Create environment
        env = FireBotEnv(
            ip=args.host,
            port=args.port,
            discrete_actions=True, # DQN requires discrete actions
            mock=args.mock,
            agent_name="DQN_Agent",
            record_data=False 
        )
        env = GridObservationWrapper(env)
        
        if args.n_frames > 1:
            print(f"Stacking {args.n_frames} frames...")
            # Gymnasium v1.0 uses stack_size instead of num_stack
            env = FrameStackObservation(env, stack_size=args.n_frames)
        
        # Load weights if checkpoint is provided
        if args.load_checkpoint:
            print(f"Loading checkpoint from {args.load_checkpoint}...")
            # We need to build the model first to load weights
            # Shape depends on frame stacking: (N_FRAMES, 65, 65) or (65, 65)
            # Action size is 7 (from FireBotEnv)
            # FrameStackObservation produces (Stack, H, W)
            # GridObservationWrapper produces (H, W)
            
            if args.n_frames > 1:
                obs_shape = (args.n_frames, 65, 65)
            else:
                obs_shape = (1, 65, 65) # d3rlpy expects channels
                # NOTE: If we are not using frame stack, GridObservationWrapper returns (65,65)
                # d3rlpy normally handles (C, H, W) from pixel inputs.
                # If we passed (65, 65) to fit_online, d3rlpy infers.
                # Here we are building manually. Let's rely on what we know the online env produces.
                # Actually, let's just use the env.observation_space.shape if possible, 
                # but we need to match what d3rlpy expects (channel first).
                obs_shape = (1, 65, 65) # Default single frame
                
            # It's safer to let d3rlpy detect from a sample or use specific known shape.
            # Offline training resulted in (4, 65, 65) for 4 frames.
            # So for n_frames=4, we use (4, 65, 65).
            
            # Action size for FireBotEnv is 7
            dqn.build_with_env(env)
            dqn.load_model(args.load_checkpoint)
            print("Checkpoint loaded successfully.")

        # Create Replay Buffer
        buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=args.buffer_size, env=env)
        
        # Start Online Training
        try:
             explorer = d3rlpy.algos.LinearDecayEpsilonGreedy(
                start_epsilon=1.0,
                end_epsilon=0.1,
                duration=int(args.n_steps * 0.5) 
            )
        except AttributeError:
             explorer = None

        fit_args = {
            "env": env,
            "buffer": buffer,
            "n_steps": args.n_steps,
            "experiment_name": "DQN_Online",
        }
        if explorer:
            fit_args["explorer"] = explorer
            
        dqn.fit_online(**fit_args)
        
        env.close()
        
    else:
        print("Starting OFFLINE training...")
        if not os.path.exists(args.dataset):
            print(f"Error: Dataset not found at {args.dataset}")
            return

        print(f"Loading dataset from {args.dataset}...")
        data = np.load(args.dataset)
        
        if 'obs_local_grid' in data:
            observations = data['obs_local_grid']
        elif 'observations' in data:
            observations = data['observations']
        else:
            raise ValueError("Could not find observations in dataset. Keys: " + str(data.keys()))



        print("Observations shape:", observations.shape)
        
        actions = convert_continuous_to_discrete(data['actions'])
        
        transition_picker = None
        if args.n_frames > 1:
            print(f"Using FrameStackTransitionPicker with n_frames={args.n_frames}")
            transition_picker = d3rlpy.dataset.FrameStackTransitionPicker(n_frames=args.n_frames)

        dataset = d3rlpy.dataset.MDPDataset(
            observations=observations,
            actions=actions,
            rewards=data['rewards'],
            terminals=data['terminals'],
            transition_picker=transition_picker
        )

        dqn.fit(
            dataset,
            n_steps=args.n_steps,
            experiment_name="DQN_Offline",
        )

    # Save the model
    dqn.save_model(args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    main()