import numpy as np
from .gym_env import DISCRETE_ACTION_MAP

def convert_continuous_to_discrete(continuous_actions):
    """
    Converts an array of continuous motor values (linear_x, angular_z) 
    back to their closest discrete action indices based on the FireBotEnv map.
    
    Args:
        continuous_actions (np.ndarray or list): Shape (N, 2) or (2,) array of floats.
        
    Returns:
        np.ndarray or int: Shape (N,) or int, containing discrete indices.
    """
    actions = np.array(continuous_actions)
    
    # Handle single action case
    is_single = False
    if actions.ndim == 1:
        is_single = True
        actions = actions.reshape(1, -1)
        
    # Prepare target vectors from map
    # Sort keys to ensure index consistency (0, 1, 2, 3...)
    sorted_keys = sorted(DISCRETE_ACTION_MAP.keys())
    target_vectors = np.array([DISCRETE_ACTION_MAP[k] for k in sorted_keys])
    
    # Calculate distances
    # actions: (N, 2)
    # target_vectors: (M, 2)
    # We want distance for each N against all M
    # diff: (N, M, 2)
    diff = actions[:, np.newaxis, :] - target_vectors[np.newaxis, :, :]
    
    # dist_sq: (N, M)
    dist_sq = np.sum(diff**2, axis=2)
    
    # Find argmin for each N
    closest_indices = np.argmin(dist_sq, axis=1)
    
    # Map back to actual keys (though keys are just 0..6 so indices match keys)
    # If keys were non-contiguous, we'd do: result = [sorted_keys[i] for i in closest_indices]
    result = np.array([sorted_keys[i] for i in closest_indices], dtype=int)
    
    if is_single:
        return result[0]
    return result
