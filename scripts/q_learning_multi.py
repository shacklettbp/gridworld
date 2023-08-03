import numpy as np
import matplotlib.pyplot as plt

# Hack, will do proper setup.py
import sys
import torch
import pickle as pkl
import pathlib

from gridworld import GridWorld

num_worlds = int(sys.argv[1])
num_steps = int(sys.argv[2])
discount = float(sys.argv[3])
num_qs = int(sys.argv[4])

with open(pathlib.Path(__file__).parent / "world_configs/test_world.pkl", 'rb') as handle:
    start_cell, end_cell, rewards, walls = pkl.load(handle)
# Start from the end cell...
grid_world = GridWorld(num_worlds, start_cell, end_cell, rewards, walls)
#grid_world.vis_world()

print(grid_world.observations.shape)
print(start_cell, end_cell, walls)

q_dict = torch.full((num_qs, walls.shape[0], walls.shape[1], 4),-10000.) # Key: [obs, action], Value: [q]
v_dict = torch.full((num_qs, walls.shape[0], walls.shape[1]),-10000.) # Key: obs, Value: v
v_dict[:,end_cell[0,0], end_cell[0,1]] = 1.
# Also set walls to 0
v_dict[:,walls == 1] = 0.

# Create queue for DP
# curr_obs = torch.tensor([[5,4]]).repeat(num_worlds, 1)
curr_rewards = torch.zeros(num_worlds)

world_qs = torch.arange(num_worlds) % num_qs
for i in range(num_steps):
    # "Policy"
    grid_world.actions[:, 0] = torch.randint(0, 4, size=(num_worlds,))

    curr_actions = grid_world.actions.clone().flatten()
    curr_states = grid_world.observations.clone()
    # Flip actions for time reversal
    '''
    curr_actions[curr_actions == 0] = 1
    curr_actions[curr_actions == 1] = 0
    curr_actions[curr_actions == 2] = 3
    curr_actions[curr_actions == 3] = 2
    '''

    # Advance simulation across all worlds
    grid_world.step()

    dones = grid_world.dones.clone().flatten()

    next_states = grid_world.observations.clone()
    next_rewards = grid_world.rewards.clone().flatten() * (1 - dones)

    # Old loop version
    '''
    for j in range(num_worlds):
        if dones[j] == 1:
            next_states[j][0] = end_cell[0,1]
            next_states[j][1] = end_cell[0,0]
            print("Victory!")
        q_dict[curr_states[j][0], curr_states[j][1], curr_actions[j]] = max(
            q_dict[curr_states[j][0], curr_states[j][1], curr_actions[j]], curr_rewards[j] + discount * v_dict[next_states[j][0], next_states[j][1]])
        v_dict[curr_states[j][0], curr_states[j][1]] = max(
            v_dict[curr_states[j][0], curr_states[j][1]], curr_rewards[j] + discount * v_dict[next_states[j][0], next_states[j][1]])
    '''
    next_states[dones == 1,0] = end_cell[0,0]
    next_states[dones == 1,1] = end_cell[0,1]
    q_dict[world_qs, curr_states[:,0], curr_states[:,1], curr_actions] = torch.max(
        q_dict[world_qs, curr_states[:,0], curr_states[:,1], curr_actions], curr_rewards + discount * v_dict[world_qs, next_states[:,0], next_states[:,1]]
    )
    v_dict[world_qs, curr_states[:,0], curr_states[:,1]] = torch.max(
        v_dict[world_qs, curr_states[:,0], curr_states[:,1]], curr_rewards + discount * v_dict[world_qs, next_states[:,0], next_states[:,1]]
    )

    curr_rewards = next_rewards * (1 - dones)

for i in range(num_qs):
    plt.imshow(v_dict[i])
    plt.show()
