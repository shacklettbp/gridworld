# File for simulating tabular worlds from "transition" and "reward" matrices

import numpy as np
import torch
import pickle as pkl

# Load the npz file
def load_npz(filename):
    b = np.load(filename)
    transitions = b["transitions.npy"]
    rewards = b["rewards.npy"]

    # In both arrays, rows are indexed by state and columns are indexed by action
    num_states = transitions.shape[0]
    num_actions = transitions.shape[1]

    # Minor cleanup to account for end states
    done_state = num_states
    num_states += 1
    transitions = np.concatenate([transitions, np.zeros((1, num_actions), dtype=transitions.dtype)], axis = 0)
    transitions[transitions == -1] = done_state
    transitions[-1,:] = done_state
    rewards = np.concatenate([rewards, np.zeros((1, num_actions), dtype=rewards.dtype)], axis = 0)

    return transitions, rewards

class TabularWorld():
    def __init__(self, filename, num_worlds):
        transitions, rewards = load_npz(filename)
        # Core transition matrix
        self.transitions = torch.tensor(transitions)
        self.transition_rewards = torch.tensor(rewards)
        # Current state
        self.observations = torch.zeros((num_worlds, 1), dtype=torch.int32)
        self.actions = torch.zeros((num_worlds, 1), dtype=torch.int32)
        self.dones = torch.zeros((num_worlds, 1), dtype=torch.int32)
        self.rewards = torch.zeros((num_worlds, 1))
        # Flag for reset per world
        self.force_reset = torch.zeros(num_worlds, dtype=torch.int32)

    def step(self):
        # Apply force_reset where needed
        self.observations[self.force_reset] = 0
        self.force_reset[...] = 0
        # Assume self.actions has been set, index into transition matrix to get next state and reward
        #print("Actions", self.actions)
        #print("Observations", self.observations)
        #print(self.transition_rewards[self.observations])
        #print(self.transition_rewards[self.observations, self.actions])
        self.rewards[...] = self.transition_rewards[self.observations,self.actions]
        self.observations[...] = self.transitions[self.observations,self.actions]
        self.dones[...] = (self.observations == self.transitions.shape[0] - 1).int()
        # Reset all the dones
        #print(self.observations.shape)
        #print(self.dones.shape)
        self.observations[self.dones == 1] = 0
