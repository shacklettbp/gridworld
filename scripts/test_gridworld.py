import numpy as np
import matplotlib.pyplot as plt

# Hack, will do proper setup.py
import sys
import torch
import pickle as pkl

from gridworld import GridWorld

num_worlds = int(sys.argv[1])
num_samples = int(sys.argv[2])

with open("./world_configs/test_world.pkl", 'rb') as handle:
    start_cell, end_cell, rewards, walls = pkl.load(handle)
grid_world = GridWorld(num_worlds, start_cell, end_cell, rewards, walls)
#grid_world.vis_world()

print(grid_world.observations.shape)

for i in range(500):
    #print("Obs:")
    #print(grid_world.observations)

    # "Policy"
    grid_world.actions[:, 0] = torch.randint(0, 4, size=(num_worlds,))
    #grid_world.actions[:, 0] = 3 # right to win given (4, 4) start

    #print("Actions:")
    #print(grid_world.actions)

    # Advance simulation across all worlds
    grid_world.step()
    
    #print("Rewards: ")
    #print(grid_world.rewards)
    #print("Dones:   ")
    #print(grid_world.dones)
    #print()
