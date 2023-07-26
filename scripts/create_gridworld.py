import numpy as np
import torch
import pickle as pkl
#from gridworld import GridWorld

array_shape = [5,5]
walls = np.zeros(array_shape)
rewards = np.zeros(array_shape)
walls[1:4,1] = 1
walls[1,3] = 1
walls[3,3] = 1
start_cell = np.array([2,0])
end_cell = np.array([[2,4]])
rewards[2,4] = 1

#grid_world = GridWorld(start_cell, end_cell, rewards, walls, num_worlds)

with open("./scripts/world_configs/test_world.pkl", 'wb') as handle:
    pkl.dump([start_cell, end_cell, rewards, walls], handle, protocol=pkl.HIGHEST_PROTOCOL)