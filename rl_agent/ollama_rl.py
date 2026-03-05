import json
import ollama
from pydantic import BaseModel, Field
import numpy as np
import os
from PIL import Image
from datetime import datetime
import time

# Import the environment
from firebot_agent.gym_env import FireBotEnv

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

class ActionDecision(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning for why this action was chosen based on the image.")
    action_id: int = Field(description="The chosen action ID (must be an integer from 0 to 6).")

def decide_action(image_input, previous_action_id: int = None):
    """
    Passes an image to the ministral-3:14b model and asks it to choose an action
    from the DISCRETE_ACTION_MAP. Uses structured output to ensure valid JSON response.
    """
    import io
    
    if isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            print("Warning: received empty image array")
            return None, None
        img = Image.fromarray(image_input)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        image_data = buffered.getvalue()
        print("Analyzing image from memory...")
    else:
        # If it's a string path or already bytes
        image_data = image_input
        print("Analyzing image...")
    
    prompt = """
    You are a robot navigation assistant. Based on the provided image from the robot's camera, decide on the best action for the robot to take next.
    Avoid getting too close to obsticales and walls is very important. Avoid stopping unless there is an immediate danger. Spinning in place is recommended to avoid getting stuck. Try to advance towards doorways and open spaces."""
    
    if previous_action_id is not None:
        prompt += f"""

    HYSTERESIS / PREVIOUS ACTION:
    The robot's PREVIOUS action ID was {previous_action_id}. 
    To ensure smooth movement and avoid jitter, you should PREFER to continue this same action if it remains safe and viable. 
    Only change the action if continuing the previous action is no longer appropriate or safe.
"""
        
    prompt += """
    You MUST output valid JSON matching the schema, choosing one of the following action IDs:

    1: Forward
    2: Backward
    3: Spin Left
    4: Spin Right
    5: Curve Left
    6: Curve Right
    """
    
    try:
        response = ollama.chat(
            model='ministral-3:8b',
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [image_data]
                }
            ],
            # Use Pydantic to enforce the schema for structured output
            format=ActionDecision.model_json_schema(),
            options={'temperature': 0.1} # Lower temperature for more consistent, deterministic reasoning
        )
        
        # Parse the JSON response back into our Pydantic model
        result_json = response['message']['content']
        decision = ActionDecision.model_validate_json(result_json)
        
        # Map the chosen action ID to the velocities
        if decision.action_id in DISCRETE_ACTION_MAP:
            velocities = DISCRETE_ACTION_MAP[decision.action_id]
            print(f"Reasoning: {decision.reasoning}")
            print(f"Chosen Action ID: {decision.action_id}")
            print(f"Command [linear_x, angular_z]: {velocities}")
            return decision, velocities
        else:
            print(f"Error: Model returned valid JSON but an invalid action ID: {decision.action_id}")
            return decision, None
            
    except Exception as e:
        print(f"Error communicating with Ollama or parsing response: {e}")
        return None, None

if __name__ == "__main__":

    # Prepare recording path
    recording_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
    os.makedirs(recording_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(recording_dir, f"ollama_{timestamp}.npz")
    print(f"Recording will be saved to: {output_file}")

    env = FireBotEnv(discrete_actions=True, agent_name="ollama", record_data=True, output_file=output_file)
    print("Resetting environment...")
    obs, info = env.reset()
    
    previous_action_id = 0 # initially stopped
    
    # Run Parameters
    MAX_STEPS = 100_000
    MAX_DURATION_SECONDS = 3600  # 1 hour
    
    start_time = time.time()
    
    for step_num in range(MAX_STEPS):
        # Check if we've exceeded the maximum allowed wall time
        if time.time() - start_time > MAX_DURATION_SECONDS:
            print(f"\nReached maximum duration of {MAX_DURATION_SECONDS} seconds. Stopping.")
            env.close()
            break
            
        step_start_time = time.time()
        print(f"\n--- Step {step_num} ---")
        image = info.get("image")
        
        if image is None or (isinstance(image, list) and len(image) == 0) or (isinstance(image, np.ndarray) and image.size == 0):
            print("No image received yet. Stepping with Stop action.")
            action_id = 0
            obs, reward, terminated, truncated, info = env.step(action_id)
            step_duration = time.time() - step_start_time
            print(f"Step {step_num} took {step_duration:.2f} seconds")
            continue
            
        decision, cmd_vel = decide_action(image, previous_action_id)
        
        if decision:
            action_id = decision.action_id
            previous_action_id = action_id
        else:
            print("Failed to get decision, stopping.")
            action_id = 0
            
        print(f"Applying Action ID {action_id}")
        obs, reward, terminated, truncated, info = env.step(action_id)
        
        if terminated or truncated:
            print("Episode ended.")
            obs, info = env.reset()
            
        step_duration = time.time() - step_start_time
        print(f"Step {step_num} took {step_duration:.2f} seconds")
