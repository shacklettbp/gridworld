import numpy as np
import matplotlib.pyplot as plt
from ._gridworld_madrona import GridWorldSimulator, madrona

__all__ = ['GridWorld']

class GridWorld:
    def __init__(self, num_worlds, start_cell, end_cells, rewards, walls):
        self.size = np.array(walls.shape)
        self.start_cell = start_cell
        self.end_cells = end_cells
        self.rewards_input = rewards
        self.walls = walls
        if np.any(start_cell < 0) or np.any(start_cell >= self.size):
            raise Exception("Start cell out of bounds")
        if np.any(end_cells < 0) or np.any(end_cells >= self.size[None,:]):
            raise Exception("End cell out of bounds")

        self.sim = GridWorldSimulator(
                walls=walls,
                rewards=rewards,
                end_cells=end_cells,
                start_x = start_cell[1],
                start_y = start_cell[0],
                max_episode_length = 0, # No max
                exec_mode = madrona.ExecMode.CUDA,
                num_worlds = num_worlds,
                gpu_id = 0,
            )

    def step(self):
        self.sim.step()

    def jax(self):
        return self.sim.jax(True)
