import numpy as np
import matplotlib.pyplot as plt

# Hack, will do proper setup.py
import sys
import torch
import pickle as pkl
import pathlib
import time
import wandb
import argparse
from torch.utils.tensorboard import SummaryWriter

from tabular_world import TabularWorld

# New argparser here

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--world-name', type=str, required=True)
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-steps', type=int, required=True)
arg_parser.add_argument('--exploration-steps', type=int, required=True)
arg_parser.add_argument('--device', type=str, default='cuda')
arg_parser.add_argument('--use-logging', action='store_true')
args = arg_parser.parse_args()

# New code

class BinInfo:
    def __init__(self, device):
        self.count = torch.tensor(0, dtype=torch.int, device=device)
        self.score = torch.tensor(0, dtype=torch.float32, device=device)
        self.states = {}

def sum_by_label(samples, labels):
    ''' select mean(samples), count() from samples group by labels order by labels asc '''
    #print(samples, labels)
    weight = torch.zeros(labels.max()+1, samples.shape[0], device=samples.device) # L, N
    weight[labels, torch.arange(samples.shape[0])] = 1.
    label_count = weight.sum(dim=1)
    mean = torch.mm(weight, samples[:,None].type(torch.float)) # L, F
    index = torch.arange(mean.shape[0], device=samples.device)[label_count > 0]
    return mean[index].flatten().type(torch.int)

def sum_by_label_optimized(samples, labels):
    # Get unique labels and sort them
    unique_labels, inverse_indices = labels.unique(return_inverse=True, sorted=True)

    # Initialize sums array for unique labels
    sums = torch.zeros_like(unique_labels, dtype=torch.float)

    # Vectorized operation to sum samples by label
    sums.index_add_(0, inverse_indices, samples.float())

    # Count occurrences of each label
    counts = torch.bincount(inverse_indices, minlength=len(unique_labels))

    return sums.flatten().type(torch.int)

# Function to replace keys with values in a provided vector using PyTorch, without loops
def replace_keys_with_values(vector, keys, values):
    # Create a tensor for mapping keys to values
    max_val = max(vector.max(), keys.max()) + 1
    mapper = torch.arange(max_val, device=vector.device)

    # Replace keys in the mapper with corresponding values
    mapper[keys] = values

    # Map the vector using the updated mapper
    return mapper[vector]

class GoExplore:
    def __init__(self, world_name, num_worlds, exploration_steps, device):
        self.worlds = TabularWorld(world_name, num_worlds, device)
        self.num_worlds = num_worlds
        self.num_states = self.worlds.transitions.shape[0]
        self.num_actions = self.worlds.transitions.shape[1]
        print("Num states: ", self.num_states)
        print("Num actions: ", self.num_actions)
        self.curr_returns = torch.zeros(num_worlds, device = device) # For tracking cumulative return of each state/bin
        self.num_exploration_steps = exploration_steps
        self.binning = "no-op"
        self.num_bins = self.num_states # We can change this later
        self.device = device
        self.state_score = torch.zeros(self.num_states, device=device)
        self.state_count = torch.zeros(self.num_states, device=device)
        self.state_count[0] = 1
        self.state_bins = torch.full((self.num_states,), self.num_states + 1, device=device) # num_states + 1 = unassigned, 0+ = bin number
        self.state_bins[torch.tensor([0], device=device)] = self.map_states_to_bins(torch.tensor([0], device=device)) # Initialize
        self.max_return = 0

    # Corrected approach to get the first element of each group without using a for loop
    def get_first_elements_unsorted_groups(self, states, groups):
        # Sort groups and states based on groups
        sorted_groups, indices = groups.sort()
        sorted_states = states[indices]

        # Find the unique groups and the first occurrence of each group
        unique_groups, first_occurrences = torch.unique(sorted_groups, return_inverse=True)
        # Mask to identify first occurrences in the sorted array
        first_occurrence_mask = torch.zeros_like(sorted_groups, dtype=torch.bool).scatter_(0, first_occurrences, 1)

        return unique_groups, sorted_states[first_occurrence_mask]

    # Step 1: Select state from archive
    # Uses: self.archive
    # Output: states
    def select_state(self):
        print("About to select state")
        # First select from visited bins with go-explore weighting function
        unique_bins, bin_inverse = self.get_first_elements_unsorted_groups(torch.arange(self.num_states, device=self.device), self.state_bins)
        bin_inverse = bin_inverse[unique_bins < self.num_states + 1]
        unique_bins = unique_bins[unique_bins < self.num_states + 1]
        # Compute bin_count and weights
        bin_count = sum_by_label_optimized(self.state_count[self.state_bins < self.num_states + 1], self.state_bins[self.state_bins < self.num_states + 1]
        )
        weights = 1./(torch.log(bin_count) + 1)
        # Sample bins
        sampled_bins = unique_bins[torch.multinomial(weights, num_samples=self.num_worlds, replacement=True).type(torch.int)]
        # Sample states from bins: either sample first occurrence in each bin (what's in the paper), or something better...
        # sampled_states = bin_inverse[sampled_bins] # Replace with scatter
        sampled_states = replace_keys_with_values(sampled_bins, unique_bins, bin_inverse)
        self.curr_returns = self.state_score[sampled_states]
        return sampled_states

    # Step 2: Go to state
    # Input: states, worlds
    # Logic: For each state, set world to state
    # Output: None
    def go_to_state(self, states):
        self.worlds.observations[:,0] = states
        return None

    # Step 3: Explore from state
    def explore_from_state(self):
        for i in range(self.num_exploration_steps):
            # Select actions either at random or by sampling from a trained policy
            self.worlds.actions[:,0] = torch.randint(0, self.num_actions, size=(self.num_worlds,), device = self.device).type(torch.int)
            self.worlds.step()
            # Map states to bins
            new_states = self.worlds.observations.clone().flatten()
            new_bins = self.map_states_to_bins(new_states)
            # Update archive
            self.curr_returns += self.worlds.rewards[:,0]
            self.max_return = max(self.max_return, torch.max(self.curr_returns).item())
            self.curr_returns *= (1 - self.worlds.dones.clone().flatten()) # Account for dones, this is working!
            #print("Max return", torch.max(self.curr_returns), self.worlds.observations[torch.argmax(self.curr_returns)])
            self.update_archive(new_bins, new_states, self.curr_returns)
        return None

    def apply_binning_function(self, states):
        if self.binning == "no-op":
            return states
        elif self.binning == "random":
            return torch.randint(0, self.num_bins, size=states.shape, device=self.device)
        else:
            raise NotImplementedError

    # Step 4: Map encountered states to bins
    def map_states_to_bins(self, states):
        # For all newly-visited states, add to state_bins
        new_states = torch.nonzero(self.state_bins[states] == self.num_states + 1).flatten()
        # Apply binning function to define bin for new states
        self.state_bins[states[new_states]] = self.apply_binning_function(states[new_states])
        # Now return the binning of all states
        return self.state_bins[states]

    # Step 5: Update archive
    def update_archive(self, bins, states, scores):
        # For each unique bin, update count in archive and update best score
        new_state_counts = torch.bincount(states, minlength=self.num_states)
        self.state_count += new_state_counts
        self.state_score[states] = torch.maximum(self.state_score[states], scores)
        return None

    # Compute best score from archive
    def compute_best_score(self):
        return torch.max(self.state_score)

# Run training loop
def train(args):
    # Create GoExplore object from args
    goExplore = GoExplore(args.world_name, args.num_worlds, args.exploration_steps, args.device)
    # Set up wandb
    run_name = f"{args.world_name}_go_explore_{int(time.time())}"
    if args.use_logging:
        wandb.init(
            project="go_explore_tabularworld",
            entity=None,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    best_score = 0
    start_time = time.time()
    for i in range(args.num_steps):
        # Step 1: Select state from archive
        states = goExplore.select_state()
        # Step 2: Go to state
        goExplore.go_to_state(states)
        # Step 3: Explore from state
        goExplore.explore_from_state()
        # Compute best score from archive
        # best_score = max(best_score, goExplore.compute_best_score())
        print(goExplore.max_return)
        # Log the step
        global_step = (i + 1)*args.num_worlds
        if args.use_logging:
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("charts/best_score", goExplore.max_return, global_step)
            # Compute number of unvisited and underexplored states from archive
            unvisited_states = torch.sum(goExplore.state_count == 0) # Need to fix this when num_states != num_bins
            underexplored_states = torch.sum(goExplore.state_count < 10) # Need to fix this 
            # Log it all
            writer.add_scalar("charts/unvisited_states", unvisited_states, global_step)
            writer.add_scalar("charts/underexplored_states", underexplored_states, global_step)
    # Return best score
    return best_score

train(args)