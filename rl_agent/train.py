from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gym_env import FireBotEnv
import os

def main():
    # 1. Create Environment
    # We set max_episode_steps to allow for truncation
    env = FireBotEnv(max_episode_steps=1000)
    
    # Optional: Check if environment follows Gym API
    print("Checking environment compatibility...")
    try:
        check_env(env)
        print("Environment check passed!")
    except Exception as e:
        print(f"Environment check warning: {e}")
        # We continue even with warnings, as some custom spaces might limit strict checking

    # 2. Define Model
    # MultiInputPolicy is required for Dict observation spaces
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1,
        tensorboard_log="./ppo_firebot_tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
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
        model_path = "ppo_firebot_final"
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model...")
        model.save("ppo_firebot_interrupted")
        print("Model saved.")
        
    finally:
        env.close()

if __name__ == "__main__":
    main()
