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
policy = str(sys.argv[4])
random_act_frac = float(sys.argv[5])
random_state_frac = float(sys.argv[6])
random_state_type = int(sys.argv[7])

with open(pathlib.Path(__file__).parent / "world_configs/test_world.pkl", 'rb') as handle:
    start_cell, end_cell, rewards, walls = pkl.load(handle)
# Start from the end cell...
grid_world = GridWorld(num_worlds, start_cell, end_cell, rewards, walls)
#grid_world.vis_world()

print(grid_world.observations.shape)
print(start_cell, end_cell, walls)

q_dict = torch.full((walls.shape[0], walls.shape[1], 4),0.) # Key: [obs, action], Value: [q]
v_dict = torch.full((walls.shape[0], walls.shape[1]),0.) # Key: obs, Value: v
v_dict[end_cell[0,0], end_cell[0,1]] = 1.
# Also set walls to 0
walls = torch.tensor(walls)
v_dict[walls == 1] = 0.

visit_dict = torch.zeros((walls.shape[0], walls.shape[1], 4), dtype=int) # Key: [obs, action], Value: # visits

# Create queue for DP
# curr_obs = torch.tensor([[5,4]]).repeat(num_worlds, 1)
curr_rewards = torch.zeros(num_worlds)
start_cell = torch.tensor(start_cell)[None, :]

# Valid states to visit are anything other than the end cell and walls
valid_states = torch.nonzero(1. - walls).type(torch.int)
#valid_states = valid_states[valid_states != end_cell]
#print(end_cell.shape, valid_states.shape)
valid_states = valid_states[(valid_states != torch.tensor(end_cell)).sum(axis=1) > 0]

for i in range(num_steps):
    # Account for random restarts or sampling into random or unvisited or underexplored states
    if random_state_frac > 0.:
        # Random restarts
        restarts = torch.rand(num_worlds) < random_state_frac
        if random_state_type == 0:
            grid_world.observations[restarts, :] = start_cell.repeat(torch.sum(restarts), 1).type(torch.int)
        elif random_state_type == 1:
            #grid_world.observations[restarts, :] = torch.cat([torch.randint(0, walls.shape[0], size=(torch.sum(restarts),1)), torch.randint(0, walls.shape[1], size=(torch.sum(restarts),1))], dim=1).type(torch.int)
            grid_world.observations[restarts, :] = valid_states[torch.randint(0, valid_states.shape[0], size=(torch.sum(restarts),)), :]#.squeeze()
            curr_rewards[restarts] = 0

    curr_states = grid_world.observations.clone()

    #grid_world.actions[:, 0] = torch.randint(0, 4, size=(num_worlds,))
    action_qs = q_dict[curr_states[:,0], curr_states[:,1]]
    if policy == "greedy":
        grid_world.actions[:, 0] = torch.argmax(action_qs, dim=1)
    elif policy == "ucb":
        # UCB
        visit_counts = visit_dict[curr_states[:,0], curr_states[:,1]] + 1
        print(visit_counts[0])
        ucb = action_qs + torch.sqrt((0.5 * np.log(1./0.05) / visit_counts)) # Hoeffding 95% confidence
        print(action_qs[0])
        print(curr_states[0])
        print(ucb[0])
        grid_world.actions[:, 0] = torch.argmax(ucb, dim=1)
    else:
        raise ValueError("Invalid policy")
    
    # Replace portion of actions with random
    if random_act_frac > 0.:
        random_rows = torch.rand(num_worlds) < random_act_frac
        grid_world.actions[random_rows, 0] = torch.randint(0, 4, size=(random_rows.sum()),).type(torch.int)

    curr_actions = grid_world.actions.clone().flatten()

    # Advance simulation across all worlds
    grid_world.step()

    dones = grid_world.dones.clone().flatten()

    next_states = grid_world.observations.clone()
    next_rewards = grid_world.rewards.clone().flatten() * (1 - dones)

    next_states[dones == 1,0] = end_cell[0,0]
    next_states[dones == 1,1] = end_cell[0,1]

    unique_states, states_count = torch.unique(torch.cat([curr_states, curr_actions[:,None]],dim=1), dim=0, return_counts=True)

    # Clobbering of values prioritizes last assignment so get index sort of curr_rewards
    rewards_order = torch.argsort(curr_rewards)
    q_dict[curr_states[rewards_order,0], curr_states[rewards_order,1], curr_actions] = torch.max(
        q_dict[curr_states[rewards_order,0], curr_states[rewards_order,1], curr_actions], curr_rewards[rewards_order] + discount * v_dict[next_states[rewards_order,0], next_states[rewards_order,1]]
    )
    v_dict[curr_states[rewards_order,0], curr_states[rewards_order,1]] = torch.max(
        v_dict[curr_states[rewards_order,0], curr_states[rewards_order,1]], curr_rewards[rewards_order] + discount * v_dict[next_states[rewards_order,0], next_states[rewards_order,1]]
    )
    visit_dict[unique_states[:,0], unique_states[:,1], unique_states[:,2]] += states_count

    curr_rewards = next_rewards * (1 - dones)

plt.imshow(v_dict)
plt.show()

plt.imshow(visit_dict.sum(2))
plt.colorbar()
plt.show()
