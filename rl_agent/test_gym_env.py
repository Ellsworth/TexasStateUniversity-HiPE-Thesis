import gymnasium as gym
from gym_env import FireBotEnv
import time

def main():
    print("Initializing FireBotEnv with max_episode_steps=5...")
    try:
        # Set a small limit for testing
        env = FireBotEnv(max_episode_steps=5)
        print("Environment initialized.")
        
        print("Resetting environment...")
        obs, info = env.reset()
        print("Reset successful.")
        
        print("Starting interaction loop...")
        for i in range(10): # Run more than max steps to see truncation
            # Sample random action
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {i+1}: Reward={reward:.2f}, Truncated={truncated}")
            
            if truncated:
                print("Episode truncated as expected!")
                obs, info = env.reset()
                print("Environment reset after truncation.")
            
            if terminated:
                print("Episode terminated.")
                obs, info = env.reset()
                
            time.sleep(0.1)
            
        print("Test completed successfully.")
        env.close()
        
    except Exception as e:
        print(f"Test FAILED with error: {e}")

if __name__ == "__main__":
    main()
