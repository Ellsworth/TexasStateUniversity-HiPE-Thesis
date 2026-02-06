import numpy as np
import d3rlpy

dataset = np.load('offline_dataset.npz')
print(dataset)

print("Obervations shape:", dataset['obs_local_grid'].shape)
print("Actions shape:", dataset['actions'].shape)
print("Rewards shape:", dataset['rewards'].shape)
print("Terminals shape:", dataset['terminals'].shape)

dataset = d3rlpy.dataset.MDPDataset(
    observations=dataset['obs_local_grid'],
    actions=dataset['actions'],
    rewards=dataset['rewards'],
    terminals=dataset['terminals'],
)

# TODO: add timeout flag array
