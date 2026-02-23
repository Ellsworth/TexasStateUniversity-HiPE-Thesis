import numpy as np
import d3rlpy
import argparse
import os

import torch
from firebot_agent.utils import convert_continuous_to_discrete
from firebot_agent.gym_env import FireBotEnv
from firebot_agent.heatmap_wrapper import PositionHeatmapWrapper
from firebot_agent.log_master import FireBotLogger
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
    parser.add_argument("--pretrain-steps", type=int, default=1000000, help="Number of offline pretraining steps")
    parser.add_argument("--finetune-steps", type=int, default=0, help="Number of online fine-tuning steps (set > 0 to enable online fine-tuning)")
    parser.add_argument("--save-path", type=str, default="d3rlpy_logs/cql_model.pt", help="Path to save the trained model")
    parser.add_argument("--pretrain-checkpoint", type=str, default="d3rlpy_logs/cql_pretrained.pt", help="Path to save/load pretrained model")
    parser.add_argument("--n-frames", type=int, default=4, help="Number of frames to stack")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Path to checkpoint to load for resuming training")
    
    # Online arguments
    parser.add_argument("--buffer-size", type=int, default=200000, help="Replay buffer size for online training")
    parser.add_argument("--mix-ratio", type=float, default=0.5, help="Fraction of each batch sampled from offline data (0.0=all online, 1.0=all offline)")
    parser.add_argument("--mock", action="store_true", help="Use mock environment (no ZMQ)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="ZMQ server host")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ server port")
    
    args = parser.parse_args()

    # Create a single timestamped log directory for this run
    logger = FireBotLogger(base_dir="logs", experiment_name="CQL_Train")
    log_dir = logger.get_log_dir()

    # Derive output paths from the log directory
    pretrain_checkpoint = os.path.join(log_dir, "cql_pretrained.pt")
    save_path = os.path.join(log_dir, "cql_model.pt")

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
    pretrain_steps = args.pretrain_steps
    finetune_steps = args.finetune_steps

    evaluators = {
        "value_scale": d3rlpy.metrics.AverageValueEstimationEvaluator(),
        "td_error": d3rlpy.metrics.TDErrorEvaluator()
    }

    # Initialize DiscreteCQL (better for offline RL with discrete actions)
    cql = d3rlpy.algos.DiscreteCQLConfig(
        learning_rate=3e-4,
        batch_size=64,
        target_update_interval=100,
        alpha=1.0,  # CQL regularization weight
        observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(),  # scale 0-255 -> 0-1
    ).create(device=torch.cuda.is_available())

    # Function to load and concatenate datasets
    def load_dataset(path):
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

    # ============ STAGE 1: OFFLINE PRETRAINING ============
    if args.load_checkpoint:
        print(f"Loading checkpoint from {args.load_checkpoint}...")
        # For resuming, we need to know the shape. Let's load the dataset to infer it.
        # We can just load the first file if it's a directory to get the shape
        if os.path.isdir(args.dataset):
             files = sorted([os.path.join(args.dataset, f) for f in os.listdir(args.dataset) if f.endswith('.npz')])
             if not files:
                 raise ValueError(f"No .npz files found in {args.dataset}")
             check_file = files[0]
        elif os.path.exists(args.dataset):
             check_file = args.dataset
        else:
             raise ValueError(f"Cannot load checkpoint without dataset at {args.dataset}")
        
        data = np.load(check_file)
        if 'obs_local_grid' in data:
            observations = data['obs_local_grid']
        elif 'observations' in data:
            observations = data['observations']
        else:
            raise ValueError("Could not find observations in dataset.")
            
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
        # Perform offline pretraining
        print("=" * 60)
        print("STAGE 1: OFFLINE PRETRAINING WITH DISCRETECQL")
        print("=" * 60)
        
        observations, actions_raw, rewards, terminals = load_dataset(args.dataset)

        print("Total observations shape:", observations.shape)
        
        actions = convert_continuous_to_discrete(actions_raw)
        
        transition_picker = None
        if args.n_frames > 1:
            print(f"Using FrameStackTransitionPicker with n_frames={args.n_frames}")
            transition_picker = d3rlpy.dataset.FrameStackTransitionPicker(n_frames=args.n_frames)

        dataset = d3rlpy.dataset.MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            transition_picker=transition_picker
        )

        print(f"Training DiscreteCQL for {pretrain_steps} steps...")
        cql.fit(
            dataset,
            n_steps=pretrain_steps,
            experiment_name="CQL_Offline_Pretrain",
            logger_adapter=TensorboardAdapterFactory(root_dir=log_dir),
            evaluators=evaluators,
        )

        # Save pretrained model
        cql.save(pretrain_checkpoint)
        print(f"Pretrained model saved to {pretrain_checkpoint}")

    # ============ STAGE 2: ONLINE FINE-TUNING ============
    if finetune_steps > 0:
        print("\n" + "=" * 60)
        print("STAGE 2: ONLINE FINE-TUNING")
        print("=" * 60)
        
        # Create environment
        online_dataset_path = os.path.join(log_dir, "online_dataset.npz")
        env = FireBotEnv(
            ip=args.host,
            port=args.port,
            discrete_actions=True,  # DiscreteCQL requires discrete actions
            mock=args.mock,
            agent_name="CQL_Agent",
            record_data=True,
            output_file=online_dataset_path,
            max_episode_steps=10_000
        )
        env = PositionHeatmapWrapper(env, save_every=10_000, save_dir=log_dir)
        env = GridObservationWrapper(env)
        
        if args.n_frames > 1:
            print(f"Stacking {args.n_frames} frames...")
            env = FrameStackObservation(env, stack_size=args.n_frames)
        
        # Build with environment if not already built
        if not hasattr(cql, '_impl') or cql._impl is None:
            cql.build_with_env(env)

        # ── Frame-stacking strategy for fine-tuning ──────────────────────────
        # The online env is wrapped with FrameStackObservation, so each step
        # already returns (n_frames, H, W) observations.  Storing those in the
        # FIFO buffer and then applying FrameStackTransitionPicker on top would
        # double-stack → (n_frames*n_frames, H, W).  Instead we:
        #   1. Use the default BasicTransitionPicker for the online buffer.
        #   2. Pre-stack the offline observations to (n_frames, H, W) so they
        #      match the online format.
        # Both buffers then use BasicTransitionPicker → MixedReplayBuffer passes
        # its picker-type assertion, and batch shapes are always consistent.

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
                    out[i, f] = obs[src, 0]          # squeeze channel dim
                if terminals[i]:
                    ep_start = i + 1
            return out

        # Create online FIFO buffer (BasicTransitionPicker by default)
        online_buffer = d3rlpy.dataset.create_fifo_replay_buffer(
            limit=args.buffer_size,
            env=env,
        )

        # Load offline data to blend into training batches
        print("Loading offline dataset for replay buffer blending...")
        off_obs, off_act_raw, off_rew, off_term = load_dataset(args.dataset)
        off_act = convert_continuous_to_discrete(off_act_raw)

        if args.n_frames > 1:
            print(f"Pre-stacking offline observations to ({args.n_frames}, H, W)...")
            off_obs = prestack_frames(off_obs, off_term, args.n_frames)
            print(f"Pre-stacked shape: {off_obs.shape}")

        offline_mdp = d3rlpy.dataset.MDPDataset(
            observations=off_obs,       # (N, n_frames, H, W) or (N, 1, H, W)
            actions=off_act,
            rewards=off_rew,
            terminals=off_term,
            # BasicTransitionPicker by default — matches online buffer
        )

        buffer = d3rlpy.dataset.MixedReplayBuffer(
            primary_replay_buffer=online_buffer,
            secondary_replay_buffer=offline_mdp,
            secondary_mix_ratio=args.mix_ratio,
        )

        # FIFOBuffer.transition_count is 0 until at least one episode is clipped.
        # With max_episode_steps=10_000, random_steps alone won't complete any
        # episode, so seed the online buffer with a few offline episodes.
        seed_episodes = list(offline_mdp.episodes)[:20]
        for ep in seed_episodes:
            online_buffer.append_episode(ep)
        print(f"Pre-seeded online buffer with {len(seed_episodes)} episodes "
              f"({online_buffer.transition_count} transitions).")

        print(f"MixedReplayBuffer: {args.mix_ratio*100:.0f}% offline / "
              f"{(1-args.mix_ratio)*100:.0f}% online per batch")
        
        # Start Online Fine-tuning with minimal exploration (policy is pretrained)
        # Low epsilon: we trust the pretrained policy and want to refine, not re-explore
        try:
            explorer = d3rlpy.algos.LinearDecayEpsilonGreedy(
                start_epsilon=0.10,  # Very low: pretrained policy already knows what to do
                end_epsilon=0.02,
                duration=finetune_steps  # Decay slowly over the full run
            )
        except AttributeError:
            explorer = None

        fit_args = {
            "env": env,
            "buffer": buffer,
            "n_steps": finetune_steps,
            "n_steps_per_epoch": 1000,  # Log/checkpoint every 1000 steps
            "random_steps": 1000,  # Pre-fill online buffer before sampling begins
            "experiment_name": "CQL_Online_Finetune",
            "logger_adapter": TensorboardAdapterFactory(root_dir=log_dir),
            # NOTE: eval_env intentionally omitted — using the same env object for eval
            # would reset the environment mid-training episode, corrupting episodes.
        }
        if explorer:
            fit_args["explorer"] = explorer
        
        print(f"Fine-tuning for {finetune_steps} steps...")
        cql.fit_online(**fit_args)
        
        env.close()

    # Save the final model
    cql.save(save_path)
    print(f"\nFinal model saved to {save_path}")

if __name__ == "__main__":
    main()