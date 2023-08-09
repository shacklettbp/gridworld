import numpy as np
import matplotlib.pyplot as plt

# Hack, will do proper setup.py
import sys
import torch
import pickle as pkl
import pathlib

from tabular_world import TabularWorld

num_worlds = int(sys.argv[1])
num_steps = int(sys.argv[2])
discount = float(sys.argv[3])

# Start from the end cell...
world = TabularWorld("/data/rl/effective-horizon/path/to/store/mdp/consolidated_ignore_screen.npz", num_worlds)
#world.vis_world()
num_states = world.transitions.shape[0]
num_actions = world.transitions.shape[1]

q_dict = torch.full((num_states, num_actions),-10000.) # Key: [obs, action], Value: [q]
v_dict = torch.full((num_states,),-10000.) # Key: obs, Value: v
v_dict[num_states - 1] = 1.

# Create queue for DP
# curr_obs = torch.tensor([[5,4]]).repeat(num_worlds, 1)
curr_rewards = torch.zeros(num_worlds, 1)

for i in range(num_steps):
    if i % 10 == 0:
        print(i)
    # "Policy"
    world.actions[:, 0] = torch.randint(0, num_actions, size=(num_worlds,))

    curr_actions = world.actions.clone().flatten()
    curr_states = world.observations.clone()
    # Flip actions for time reversal
    '''
    curr_actions[curr_actions == 0] = 1
    curr_actions[curr_actions == 1] = 0
    curr_actions[curr_actions == 2] = 3
    curr_actions[curr_actions == 3] = 2
    '''

    # Advance simulation across all worlds
    world.step()

    #print(world.observations)
    #print(world.actions)
    #print(world.dones)
    dones = world.dones.clone()#.flatten()

    next_states = world.observations.clone()
    #next_rewards = world.rewards.clone().flatten() * (1 - dones)
    next_rewards = world.rewards.clone() * (1 - dones)

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
    next_states[dones == 1] = num_states - 1
    q_dict[curr_states, curr_actions] = torch.max(
        q_dict[curr_states, curr_actions], curr_rewards + discount * v_dict[next_states]
    )
    v_dict[curr_states] = torch.max(
        v_dict[curr_states], curr_rewards + discount * v_dict[next_states]
    )

    curr_rewards = next_rewards * (1 - dones)

#plt.imshow(v_dict)
#plt.show()
print(v_dict)
print(q_dict)
