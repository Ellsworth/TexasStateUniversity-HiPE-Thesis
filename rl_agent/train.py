from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gym_env import FireBotEnv
import os

def main():
    # 1. Create Environment
    # We set max_episode_steps to allow for truncation
    
    # Wrap in DummyVecEnv for VecFrameStack compatibility
    env = DummyVecEnv([lambda: FireBotEnv(max_episode_steps=1000, discrete_actions=True)])
    
    # Stack 4 frames
    env = VecFrameStack(env, n_stack=4)
    
    # 2. Define Model
    # MultiInputPolicy is required for Dict observation spaces
    model = DQN(
        "MultiInputPolicy", 
        env, 
        verbose=1,
        tensorboard_log="./dqn_firebot_tensorboard/",
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0, 
        target_update_interval=1000,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
    )

    # 3. Train
    print("Starting training...")
    try:
        # Train for a set number of timesteps. 
        # User asked for "forever", but we need a loop or huge number.
        # Let's set a large number, or we can do a loop of learn calls.
        
        # 1 Million steps for initial run
        total_timesteps = 1_000_000 
        
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        
        # 4. Save Model
        model_path = "dqn_firebot_final"
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model...")
        model.save("dqn_firebot_interrupted")
        print("Model saved.")
        
    finally:
        env.close()

if __name__ == "__main__":
    main()
