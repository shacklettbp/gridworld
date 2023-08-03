import numpy as np
import matplotlib.pyplot as plt

# Hack, will do proper setup.py
import sys
import torch
import pickle as pkl
import pathlib

from gridworld import GridWorld
from tabular_policy import TabularPolicy

#from torch_discounted_cumsum import discounted_cumsum_right

num_worlds = int(sys.argv[1])
num_steps = int(sys.argv[2])

with open(pathlib.Path(__file__).parent / "world_configs/test_world.pkl", 'rb') as handle:
    start_cell, end_cell, rewards, walls = pkl.load(handle)
grid_world = GridWorld(num_worlds, start_cell, end_cell, rewards, walls)
#grid_world.vis_world()

print(grid_world.observations.shape)

# We index into observation state by doing row_id*num_cols + col_id
num_rows = walls.shape[0]
num_cols = walls.shape[1]
policy = TabularPolicy(num_rows * num_cols, 4, greedy = False)

obs_list = []
action_list = []
reward_list = []
done_list = [torch.zeros(num_worlds, 1)]

for i in range(num_steps):
    #print(i)
    curr_obs = grid_world.observations.clone()
    obs_list.append(curr_obs)

    # "Policy"
    #grid_world.actions[:, 0] = torch.randint(0, 4, size=(num_worlds,))
    flattened_obs = curr_obs[:,0]*num_cols + curr_obs[:,1]
    #print(curr_obs[:,0], curr_obs[:,1], flattened_obs)
    acts = policy(flattened_obs)
    #print(acts)
    grid_world.actions[:,:] = acts

    action_list.append(grid_world.actions.clone())

    # Advance simulation across all worlds
    grid_world.step()
    
    reward_list.append(grid_world.rewards.clone())

    if i < num_steps - 1:
        done_list.append(grid_world.dones.clone())

obs = torch.stack(obs_list).swapaxes(0,1)
actions = torch.stack(action_list).swapaxes(0,1)
rewards = torch.stack(reward_list).swapaxes(0,1)
dones = torch.stack(done_list).swapaxes(0,1)[:,:,0]

oa = torch.cat([obs,actions], dim=2)
# Shape: (worlds, samples, o + a)
episode_ids = torch.cumsum(dones.type(torch.int), dim = 1)
unique_episodes = episode_ids.unique()

M = torch.zeros(num_worlds, (int)(episode_ids.max().item() + 1), num_steps)
M[torch.repeat_interleave(torch.arange(num_worlds), num_steps), episode_ids.flatten(), torch.arange(num_steps).repeat(num_worlds)] = 1

# Maybe do cumsum with M as a filter

filtered_rewards = M * rewards[:,:,0][:,None,:]
aggregate_rewards = torch.flip(torch.cumsum(torch.flip(filtered_rewards, dims=[2]), dim=2), dims=[2])
#aggregate_rewards = discounted_cumsum_right(filtered_rewards.reshape(-1,filtered_rewards.shape[2]), 1.0).reshape(filtered_rewards.shape)

aggregate_rewards *= M
aggregate_rewards = torch.sum(aggregate_rewards, dim = 1)

# FLATTEN WORLDS
aggregate_rewards = aggregate_rewards.flatten()
oa = oa.reshape(-1, oa.shape[-1])

# Now get average value for each state by group by and average
# we could implement this the other way from earlier too... https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335
unique_states, inverse_states, states_count = oa.unique(dim = 0, return_inverse = True, return_counts = True)

states_values = torch.zeros_like(torch.arange(unique_states.shape[0]), dtype=torch.float).scatter_add_(0, inverse_states, aggregate_rewards)
states_values = states_values / states_count.float()#.unsqueeze(1)

print(unique_states, unique_states.shape)
print(states_count, states_count.shape)
print(states_values, states_values.shape)

im = torch.zeros((torch.max(unique_states[:,0]) + 1, torch.max(unique_states[:,1]) + 1, torch.max(unique_states[:,2]) + 1))
im2 = torch.zeros((torch.max(unique_states[:,0]) + 1, torch.max(unique_states[:,1]) + 1))
for (i, state) in enumerate(unique_states):
    im[state[0], state[1], state[2]] = states_values[i]
    im2[state[0], state[1]] += states_count[i]
plt.imshow(im2)
plt.colorbar()
plt.show()
exit()
plt.imshow(im[:,:,0])
plt.show()
plt.imshow(im[:,:,1])
plt.show()
plt.imshow(im[:,:,2])
plt.show()
plt.imshow(im[:,:,3])
plt.show()
