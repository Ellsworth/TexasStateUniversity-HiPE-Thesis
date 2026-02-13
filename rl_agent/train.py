import numpy as np
import d3rlpy
import argparse
import os

import torch
from firebot_agent.utils import convert_continuous_to_discrete
from firebot_agent.gym_env import FireBotEnv
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from d3rlpy.logging import TensorboardAdapterFactory

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
    parser = argparse.ArgumentParser(description="DiscreteCQL Training (Offline Pretraining + Online Fine-tuning)")
    parser.add_argument("--dataset", type=str, default="offline_dataset.npz", help="Path to the offline dataset (npz)")
    parser.add_argument("--n-steps", type=int, default=1000000, help="Number of training steps (used as default for both stages)")
    parser.add_argument("--pretrain-steps", type=int, default=None, help="Number of offline pretraining steps (overrides --n-steps for pretraining)")
    parser.add_argument("--finetune-steps", type=int, default=None, help="Number of online fine-tuning steps (overrides --n-steps for fine-tuning)")
    parser.add_argument("--save-path", type=str, default="d3rlpy_logs/cql_model.d3", help="Path to save the trained model")
    parser.add_argument("--pretrain-checkpoint", type=str, default="d3rlpy_logs/cql_pretrained.d3", help="Path to save/load pretrained model")
    parser.add_argument("--n-frames", type=int, default=4, help="Number of frames to stack")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Path to checkpoint to load for resuming training")
    
    # Online arguments
    parser.add_argument("--online", action="store_true", help="Enable online fine-tuning after offline pretraining")
    parser.add_argument("--buffer-size", type=int, default=200000, help="Replay buffer size for online training")
    parser.add_argument("--mock", action="store_true", help="Use mock environment (no ZMQ)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="ZMQ server host")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ server port")
    
    args = parser.parse_args()
    
    # Print GPU info
    print("=" * 60)
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"GPUs available: {n_gpus}")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available. Using CPU.")
    print("=" * 60)

    # Determine training steps for each stage
    pretrain_steps = args.pretrain_steps if args.pretrain_steps is not None else args.n_steps
    finetune_steps = args.finetune_steps if args.finetune_steps is not None else args.n_steps

    # Initialize DiscreteCQL (better for offline RL with discrete actions)
    cql = d3rlpy.algos.DiscreteCQLConfig(
        learning_rate=3e-4,
        batch_size=64,
        target_update_interval=100,
        alpha=1.0  # CQL regularization weight
    ).create(device=torch.cuda.is_available())

    # ============ STAGE 1: OFFLINE PRETRAINING ============
    if args.load_checkpoint:
        print(f"Loading checkpoint from {args.load_checkpoint}...")
        # For resuming, we need to know the shape. Let's load the dataset to infer it.
        if os.path.exists(args.dataset):
            data = np.load(args.dataset)
            if 'obs_local_grid' in data:
                observations = data['obs_local_grid']
            elif 'observations' in data:
                observations = data['observations']
            else:
                raise ValueError("Could not find observations in dataset. Keys: " + str(data.keys()))
            
            # Build model structure before loading
            obs_shape = observations[0].shape
            if args.n_frames > 1:
                # With frame stacking, shape will be (n_frames, H, W)
                obs_shape = (args.n_frames, obs_shape[-2], obs_shape[-1])
            
            # Create a dummy environment to build the model
            # We'll use the first observation to infer shapes
            from d3rlpy.dataset import MDPDataset
            actions = convert_continuous_to_discrete(data['actions'])
            dataset = MDPDataset(
                observations=observations[:1],
                actions=actions[:1],
                rewards=data['rewards'][:1],
                terminals=data['terminals'][:1]
            )
            cql.build_with_dataset(dataset)
            cql.load_model(args.load_checkpoint)
            print("Checkpoint loaded successfully.")
        else:
            raise ValueError(f"Cannot load checkpoint without dataset at {args.dataset}")
    else:
        # Perform offline pretraining
        print("=" * 60)
        print("STAGE 1: OFFLINE PRETRAINING WITH DISCRETECQL")
        print("=" * 60)
        
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

        print(f"Training DiscreteCQL for {pretrain_steps} steps...")
        cql.fit(
            dataset,
            n_steps=pretrain_steps,
            experiment_name="CQL_Offline_Pretrain",
            logger_adapter=TensorboardAdapterFactory(root_dir="logs"),
        )
        
        # Save pretrained model
        cql.save_model(args.pretrain_checkpoint)
        print(f"Pretrained model saved to {args.pretrain_checkpoint}")

    # ============ STAGE 2: ONLINE FINE-TUNING ============
    if args.online:
        print("\n" + "=" * 60)
        print("STAGE 2: ONLINE FINE-TUNING")
        print("=" * 60)
        
        # Create environment
        env = FireBotEnv(
            ip=args.host,
            port=args.port,
            discrete_actions=True,  # DiscreteCQL requires discrete actions
            mock=args.mock,
            agent_name="CQL_Agent",
            record_data=False 
        )
        env = GridObservationWrapper(env)
        
        if args.n_frames > 1:
            print(f"Stacking {args.n_frames} frames...")
            env = FrameStackObservation(env, stack_size=args.n_frames)
        
        # Build with environment if not already built
        if not hasattr(cql, '_impl') or cql._impl is None:
            cql.build_with_env(env)

        # Create Replay Buffer
        buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=args.buffer_size, env=env)
        
        # Start Online Fine-tuning with decayed exploration
        try:
            explorer = d3rlpy.algos.LinearDecayEpsilonGreedy(
                start_epsilon=0.3,  # Start with lower epsilon since we're fine-tuning
                end_epsilon=0.05,
                duration=int(finetune_steps * 0.5) 
            )
        except AttributeError:
            explorer = None

        fit_args = {
            "env": env,
            "buffer": buffer,
            "n_steps": finetune_steps,
            "experiment_name": "CQL_Online_Finetune",
            "logger_adapter": TensorboardAdapterFactory(root_dir="logs"),
        }
        if explorer:
            fit_args["explorer"] = explorer
        
        print(f"Fine-tuning for {finetune_steps} steps...")
        cql.fit_online(**fit_args)
        
        env.close()

    # Save the final model
    cql.save_model(args.save_path)
    print(f"\nFinal model saved to {args.save_path}")

if __name__ == "__main__":
    main()