import numpy as np
import matplotlib.pyplot as plt
import os

from typing import Optional

# Hack, will do proper setup.py
import sys
import torch
import pickle as pkl
import pathlib
import time
import wandb
import argparse
from torch.utils.tensorboard import SummaryWriter
from toolbox.printing import debug

from tabular_world import TabularWorld
from ssim import SSIM
import faiss

# New argparser here

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--world-name", type=str, required=True)
arg_parser.add_argument("--mdp_path", type=str, required=True)
arg_parser.add_argument("--render_path", type=str, required=True)
arg_parser.add_argument("--num-worlds", type=int, required=True)
arg_parser.add_argument("--num-steps", type=int, required=True)
arg_parser.add_argument("--exploration-steps", type=int, required=True)
arg_parser.add_argument("--device", type=str, default="cuda")
arg_parser.add_argument("--use-logging", action="store_true")
arg_parser.add_argument("--binning-method", type=str, default="none")
arg_parser.add_argument(
    "--num-bins",
    type=int,
    default=-1,
    help="Number of bins (set to -1 to have one bin per state)",
)
arg_parser.add_argument("--run-name", type=str, default=None)
args = arg_parser.parse_args()


# New code
def hash_batch(data: torch.Tensor) -> torch.Tensor:
    # assert data.dtype == torch.uint8
    assert len(data.shape) >= 2

    # Non efficient hashing: convert to batch size lists of integers, then use Python's hash function
    # print("Data shape", data.shape)
    flattened_data = data.flatten(start_dim=1)
    data_list = [tuple(row.tolist()) for row in flattened_data]
    hash_values = torch.tensor([hash(x) for x in data_list], device=data.device)
    assert hash_values.shape == (data.shape[0],)
    # print("Hashed values: ", hash_values)
    return hash_values


def sum_by_label(samples, labels):
    """select mean(samples), count() from samples group by labels order by labels asc"""
    # print(samples, labels)
    weight = torch.zeros(
        labels.max() + 1, samples.shape[0], device=samples.device
    )  # L, N
    weight[labels, torch.arange(samples.shape[0])] = 1.0
    label_count = weight.sum(dim=1)
    mean = torch.mm(weight, samples[:, None].type(torch.float))  # L, F
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
    def __init__(
        self,
        world_path: str,
        render_path: str,
        num_worlds: int,
        exploration_steps: int,
        device: str,
        binning_method: str = "none",
        num_bins: Optional[int] = None,
    ):
        self.worlds = TabularWorld(world_path, num_worlds, device)
        self.num_worlds = num_worlds
        self.num_states = self.worlds.transitions.shape[0]
        self.num_actions = self.worlds.transitions.shape[1]
        print("Num states: ", self.num_states)
        print("Num actions: ", self.num_actions)
        self.curr_returns = torch.zeros(
            num_worlds, device=device
        )  # For tracking cumulative return of each state/bin
        self.num_exploration_steps = exploration_steps
        self.binning = binning_method
        self.num_bins = num_bins if num_bins is not None else self.num_states
        self.device = device
        self.state_score = torch.zeros(self.num_states, device=device)
        self.state_count = torch.zeros(self.num_states, device=device)
        self.state_count[0] = 1
        self.state_bins = torch.full(
            (self.num_states,), self.num_states + 1, device=device
        )  # num_states + 1 = unassigned, 0+ = bin number

        # Check if render data exists
        if os.path.exists(render_path) and self.binning in ["pixel", "ssim"]:
            self.frames = np.load(render_path)
            assert (
                self.frames.shape[0] == self.num_states
            ), f"Render data has {self.frames.shape[0]} states, but the MDP has {self.num_states} states."
            print("Rendering data exists.")
            # Move to device
            self.frames = torch.tensor(self.frames, device=device)
            if len(self.frames.shape) == 3:
                self.frames = self.frames.unsqueeze(1)
            self.frames = self.frames.float() / 255.0

            if self.binning == "ssim":
                self.ssim_obj = SSIM(window_size=11, size_average=False)
                self.ssim_refs = torch.zeros_like(self.frames[:num_bins])
                self.num_bins_used = 0
        elif self.binning == "clip":
            self.clip_indices = np.load(
                render_path.replace(".npy", "_clip_indices.npy")
            )
            self.clip_indices = torch.tensor(self.clip_indices, device=device)
        elif self.binning == "clip_live":
            self.clip_values = np.load(render_path.replace(".npy", "_clip.npy"))
            self.clip_values = torch.tensor(self.clip_values, device=device)
            self.recompute_every = self.num_bins // 2
            self.num_centroids = 0
            self.centroids = torch.zeros(
                (self.num_bins, self.clip_values.shape[1]), device=device
            )
            self.last_update = -self.recompute_every
        else:
            print("Rendering data does not exist.")
            if self.binning not in ["none", "random"]:
                raise ValueError(
                    f"Binning method {self.binning} not supported without rendering data."
                )

        self.state_bins[torch.tensor([0], device=device)] = self.map_states_to_bins(
            torch.tensor([0], device=device)
        )  # Initialize
        self.max_return = 0

    # Corrected approach to get the first element of each group without using a for loop
    def get_first_elements_unsorted_groups(self, states, groups):
        # Sort groups and states based on groups
        sorted_groups, indices = groups.sort()
        sorted_states = states[indices]

        # Find the unique groups and the first occurrence of each group
        unique_groups, group_to_unique_group = torch.unique(
            sorted_groups, return_inverse=True
        )

        # Sample
        sampled_states = torch.zeros_like(unique_groups)
        for i, group in enumerate(unique_groups):
            idx_part_of_group = torch.nonzero(sorted_groups == group).flatten()
            # Sample from the indices that are part of the group (randomly)
            sampled_states[i] = sorted_states[
                idx_part_of_group[torch.randint(0, idx_part_of_group.shape[0], (1,))]
            ]
        inverse_bin = sampled_states
        unique_bins = unique_groups
        # # debug(first_occurrences)
        # # debug(sorted_groups[first_occurrences])
        # # Mask to identify first occurrences in the sorted array
        # first_occurrence_mask = torch.zeros_like(
        #     sorted_groups, dtype=torch.bool
        # ).scatter_(0, first_occurrences, 1)
        # # debug(sorted_groups[first_occurrence_mask])

        # return unique_groups, sorted_states[first_occurrence_mask]

        # # debug(states)
        # # debug(groups)
        # # Remove value len(groups) + 1 from groups
        # groups = groups[groups < self.num_states + 1]
        # # Sort groups and states based on groups
        # sorted_bins, indices = groups.sort()
        # # debug(sorted_bins)
        # # debug(indices)
        # sorted_states = states[indices]

        # # Find the unique groups and the first occurrence of each group
        # unique_bins, state_to_bin = torch.unique(sorted_bins, return_inverse=True)
        # # debug(unique_bins)
        # # debug(state_to_bin)

        # # Shuffle so that we dont always get the first occurrence
        # shuffled_indices = torch.randperm(state_to_bin.shape[0], device=groups.device)

        # # Mask to identify first occurrences in the sorted array
        # first_occurrence_mask = torch.zeros_like(
        #     sorted_bins, dtype=torch.bool
        # ).scatter_(0, state_to_bin, 1)
        # inverse_bin = sorted_states[first_occurrence_mask]
        # # debug(unique_bins)
        # # debug(inverse_bin)
        # # debug(self.state_bins[inverse_bin])

        return unique_bins, inverse_bin

    # Step 1: Select state from archive
    # Uses: self.archive
    # Output: states
    def select_state(self):
        print("About to select state")
        # First select from visited bins with go-explore weighting function
        unique_bins, bin_inverse = self.get_first_elements_unsorted_groups(
            torch.arange(self.num_states, device=self.device), self.state_bins
        )
        # # debug(unique_bins)
        # # debug(bin_inverse)
        bin_inverse = bin_inverse[unique_bins < self.num_states + 1]
        unique_bins = unique_bins[unique_bins < self.num_states + 1]
        # Compute bin_count and weights
        bin_count = sum_by_label_optimized(
            self.state_count[self.state_bins < self.num_states + 1],
            self.state_bins[self.state_bins < self.num_states + 1],
        )
        weights = 1.0 / (torch.sqrt(bin_count) + 1)
        # Sample bins
        sampled_bins = unique_bins[
            torch.multinomial(
                weights, num_samples=self.num_worlds, replacement=True
            ).type(torch.int)
        ]
        # Sample states from bins: either sample first occurrence in each bin (what's in the paper), or something better...
        # sampled_states = bin_inverse[sampled_bins] # Replace with scatter
        sampled_states = replace_keys_with_values(
            sampled_bins, unique_bins, bin_inverse
        )
        self.curr_returns = self.state_score[sampled_states]
        return sampled_states

    # Step 2: Go to state
    # Input: states, worlds
    # Logic: For each state, set world to state
    # Output: None
    def go_to_state(self, states):
        self.worlds.observations[:, 0] = states
        return None

    # Step 3: Explore from state
    def explore_from_state(self):
        for i in range(self.num_exploration_steps):
            # Select actions either at random or by sampling from a trained policy
            self.worlds.actions[:, 0] = torch.randint(
                0, self.num_actions, size=(self.num_worlds,), device=self.device
            ).type(torch.int)
            self.worlds.step()
            # Map states to bins
            new_states = self.worlds.observations.clone().flatten()
            new_bins = self.map_states_to_bins(new_states)
            # Update archive
            self.curr_returns += self.worlds.rewards[:, 0]
            self.max_return = max(self.max_return, torch.max(self.curr_returns).item())
            self.curr_returns *= (
                1 - self.worlds.dones.clone().flatten()
            )  # Account for dones, this is working!
            # print("Max return", torch.max(self.curr_returns), self.worlds.observations[torch.argmax(self.curr_returns)])
            self.update_archive(new_bins, new_states, self.curr_returns)
        return None

    def apply_binning_function(self, states):
        if self.binning == "none":
            return states % self.num_bins
        elif self.binning == "random":
            return torch.randint(
                0, self.num_bins, size=states.shape, device=self.device
            )
        elif self.binning == "pixel":
            assert self.frames is not None, "Rendering data not provided."
            # Get frames for states
            frames = self.frames[states]
            # Hash the images to produce a tensor of shape (states.shape[0],)
            hashed_frames = hash_batch(frames)
            return hashed_frames % self.num_bins
        elif self.binning == "ssim":
            bins = torch.zeros_like(states)
            for i, state in enumerate(states):
                new_bin = False
                bin = None
                frame = self.frames[state]
                if self.num_bins_used == 0:  # First bin
                    new_bin = True
                else:
                    ssim_values = self.ssim_obj(
                        frame.unsqueeze(0), self.ssim_refs[: self.num_bins_used]
                    )
                    if (
                        torch.max(ssim_values) < 0.97
                        and self.num_bins_used < self.num_bins
                    ):
                        new_bin = True
                    else:
                        bin = torch.argmax(ssim_values)
                if new_bin:
                    self.ssim_refs[self.num_bins_used] = frame
                    bin = self.num_bins_used
                    self.num_bins_used += 1
                bins[i] = bin
            return bins
        elif self.binning == "clip":
            assert self.clip_indices is not None, "Rendering data not provided."
            return self.clip_indices[states]
        elif self.binning == "clip_live":
            new_bins = torch.zeros_like(states)
            # debug(states)
            unique_states = torch.unique(states)
            # debug(unique_states)

            num_new_states = len(unique_states)
            num_centroids_still_available = self.num_bins - self.num_centroids
            num_centroids_before_recompute = self.last_update + self.recompute_every
            # debug(num_new_states)
            # debug(num_centroids_still_available)
            # debug(num_centroids_before_recompute)

            if (
                num_new_states
                >= num_centroids_still_available + num_centroids_before_recompute
            ):
                print("RETRAIN")
                # Needs to recompute KMeans
                self.last_update = 0

                # Prepare the data to kmeans i.e. the state_to_bin
                # that do not map to self.num_bins + 1
                # plus the new states
                # Prepare two arrays:
                # - clip_values: the values of the states that are not
                #   mapped to self.num_bins + 1
                # - state_indices: the indices of the states that are
                #   not mapped to self.num_bins + 1
                states_indices = torch.nonzero(
                    self.state_bins != self.num_states + 1
                ).flatten()
                # Add the new states
                states_indices = torch.cat(
                    (states_indices, torch.tensor(unique_states, device=self.device))
                )
                states_indices = states_indices[states_indices < self.num_states]
                clip_values = self.clip_values[states_indices]
                # debug(clip_values)
                # debug(states_indices)

                # Recompute centroids
                ncentroids = self.num_bins
                # debug(ncentroids)
                niter = 50
                verbose = False
                d = self.clip_values.shape[1]
                self.kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
                self.kmeans.train(clip_values.cpu().numpy())
                self.centroids = torch.tensor(self.kmeans.centroids, device=self.device)
                # debug(self.centroids)

                # Update bins
                bins = torch.full(
                    (self.num_states,), self.num_states + 1, device=self.device
                )
                _, I = self.kmeans.index.search(clip_values.cpu().numpy(), 1)
                # debug(torch.unique(torch.tensor(I.squeeze())))
                bins[states_indices] = torch.tensor(I.squeeze(), device=self.device)
                self.state_bins = bins.clone()
                # debug(I)
                # debug(bins[states_indices])
                # debug(torch.unique(bins))

                # COunt number of states not assigned to self.num_states + 1
                num = torch.sum(bins != self.num_states + 1)
                # debug(num)

                # Get the bins for the new states
                new_bins = bins[states]
                self.num_centroids = ncentroids
                print("New bins computed")
                # # debug(new_bins)
                return new_bins

            # No recompute will be required
            # First let's add centroids if we still can
            if num_centroids_still_available > 0:
                # debug(self.clip_values[unique_states])
                temp = self.centroids[
                    self.num_centroids : self.num_centroids + num_new_states
                ]
                # debug(temp)
                # debug(self.num_centroids)
                # debug(self.num_centroids + num_new_states)
                # debug(self.centroids)
                self.centroids[
                    self.num_centroids : self.num_centroids + num_new_states
                ] = self.clip_values[unique_states]
                self.num_centroids += num_new_states
                # debug(self.centroids)
            else:
                clip_values = self.clip_values[states]
                _, I = self.kmeans.index.search(clip_values.cpu().numpy(), 1)
                new_bins = torch.tensor(I.flatten(), device=self.device)
                self.last_update -= num_new_states

            # # debug(new_bins)
            return new_bins
        else:
            raise NotImplementedError

    # Step 4: Map encountered states to bins
    def map_states_to_bins(self, states):
        # For all newly-visited states, add to state_bins
        new_states = torch.nonzero(
            self.state_bins[states] == self.num_states + 1
        ).flatten()
        # Apply binning function to define bin for new states
        if new_states.shape[0] > 0:
            new_state_bins = self.apply_binning_function(states[new_states])
            self.state_bins[states[new_states]] = new_state_bins
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
    world_name = args.world_name
    print(f"Running Go-Explore on {world_name}")
    mdp_path = os.path.join(args.mdp_path, f"{world_name}/consolidated.npz")
    render_path = os.path.join(args.render_path, f"render_data_{world_name}.npy")
    num_bins = args.num_bins if args.num_bins > 0 else None
    goExplore = GoExplore(
        mdp_path,
        render_path,
        args.num_worlds,
        args.exploration_steps,
        args.device,
        args.binning_method,
        num_bins,
    )
    # Set up wandb
    run_name = (
        args.run_name
        if args.run_name
        else f"{args.world_name}_go_explore_{int(time.time())}"
    )
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
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
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
        print(f"Step: {i} - Max return: {goExplore.max_return}")
        # Log the step
        global_step = (i + 1) * args.num_worlds
        if args.use_logging:
            writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )
            writer.add_scalar("charts/best_score", goExplore.max_return, global_step)
            # Compute number of unvisited and underexplored states from archive
            unvisited_states = torch.sum(
                goExplore.state_count == 0
            )  # Need to fix this when num_states != num_bins
            underexplored_states = torch.sum(
                goExplore.state_count < 10
            )  # Need to fix this
            # Compute the same for bins
            visited_bins = torch.unique(goExplore.state_bins).shape[0]
            # Log it all
            writer.add_scalar("charts/unvisited_states", unvisited_states, global_step)
            writer.add_scalar(
                "charts/underexplored_states", underexplored_states, global_step
            )
            writer.add_scalar(
                "charts/unvisited_bins",
                goExplore.num_bins - visited_bins + 1,
                global_step,
            )

    # Analysis of bins
    print("\nBinning analysis:")
    print(" - Binning method: ", goExplore.binning)
    print(" - Number of bins: ", goExplore.num_bins)
    print(" - Number of bins used: ", torch.unique(goExplore.state_bins).shape[0])

    # Return best score
    return best_score


train(args)
