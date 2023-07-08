import numpy as np
import matplotlib.pyplot as plt

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
start_cell = np.array([0,0])
end_cell = np.array([4,5])
rewards[4,0] = -1
rewards[4,5] = 1

x = GridWorld(start_cell, end_cell, rewards, walls)
x.vis_world()
