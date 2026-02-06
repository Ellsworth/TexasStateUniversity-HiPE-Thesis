import numpy as np
import os

class OfflineDataCollector:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.terminals = []

    def add_step(self, obs, action, reward, terminal):
        """
        Adds a single step of interaction to the buffer.
        """
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)

    def save(self, filename="offline_dataset.npz"):
        """
        Converts the collected lists into numpy arrays and saves them.
        """
        if not self.observations:
            print("Warning: Buffer is empty. Nothing to save.")
            return

        # Prepare dictionary for saving
        payload = {
            "actions": np.array(self.actions, dtype=np.float32),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "terminals": np.array(self.terminals, dtype=bool)
        }

        # Handle observations
        if self.observations and isinstance(self.observations[0], dict):
            # Dict observation: save each key separately
            for key in self.observations[0].keys():
                # We let numpy infer the dtype for each key (e.g. uint8 for grid)
                payload[f"obs_{key}"] = np.array([o[key] for o in self.observations])
        else:
            # Standard observation
            payload["observations"] = np.array(self.observations, dtype=np.float32)

        # Use np.savez or np.savez_compressed
        np.savez_compressed(filename, **payload)
        
        print(f"Dataset successfully saved to {filename}")

    def clear(self):
        """Clears the current buffer memory."""
        self.__init__()