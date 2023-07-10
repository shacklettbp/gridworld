import numpy as np
import matplotlib.pyplot as plt

# Hack, will do proper setup.py
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent / "build"))
import gridworld_madrona
import torch

num_worlds = int(sys.argv[1])

class GridWorld:
    def __init__(self, start_cell, end_cells, rewards, walls):
        self.size = np.array(walls.shape)
        self.start_cell = start_cell
        self.end_cells = end_cells
        self.rewards = rewards
        self.walls = walls
        if np.any(start_cell < 0) or np.any(start_cell >= self.size):
            raise Exception("Start cell out of bounds")
        if np.any(end_cells < 0) or np.any(end_cells >= self.size[None,:]):
            raise Exception("End cell out of bounds")

        self.sim = gridworld_madrona.GridWorldSimulator(
                walls=walls,
                rewards=rewards,
                end_cells=end_cells,
                start_x = start_cell[0],
                start_y = start_cell[1],
                max_episode_length = 0, # No max
                exec_mode = gridworld_madrona.madrona.ExecMode.CPU,
                num_worlds = num_worlds,
            )

        self.force_reset = self.sim.reset_tensor().to_torch()
        self.actions = self.sim.action_tensor().to_torch()
        self.observations = self.sim.observation_tensor().to_torch()
        self.rewards = self.sim.reward_tensor().to_torch()
        self.dones = self.sim.done_tensor().to_torch()

    def step(self):
        self.sim.step()

    def vis_world(self):
        im = np.ones(np.append(self.size, 3))
        im[self.walls > 0] = 0
        im[self.rewards > 0] = np.array([0,1,0])
        im[self.rewards < 0] = np.array([1,0,0])
        im[self.start_cell[0], self.start_cell[1]] = np.array([0,0,1])
        plt.imshow(im)
        plt.show()

array_shape = [5,6]
walls = np.zeros(array_shape)
rewards = np.zeros(array_shape)
walls[3,2:] = 1
start_cell = np.array([4,4])
end_cell = np.array([[4,5]])
rewards[4,0] = -1
rewards[4,5] = 1

grid_world = GridWorld(start_cell, end_cell, rewards, walls)
#grid_world.vis_world()

print(grid_world.observations.shape)

for i in range(5):
    print("Obs:")
    print(grid_world.observations)

    # "Policy"
    grid_world.actions[:, 0] = torch.randint(0, 4, size=(num_worlds,))
    #grid_world.actions[:, 0] = 3 # right to win given (4, 4) start

    print("Actions:")
    print(grid_world.actions)

    # Advance simulation across all worlds
    grid_world.step()
    
    print("Rewards: ")
    print(grid_world.rewards)
    print("Dones:   ")
    print(grid_world.dones)
    print()
