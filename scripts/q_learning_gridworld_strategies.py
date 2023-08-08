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

from gridworld import GridWorld

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-steps', type=int, required=True)
arg_parser.add_argument('--discount', type=float, default=0.99)
arg_parser.add_argument('--policy', type=str, default='ucb')
arg_parser.add_argument('--random-act-frac', type=float, default=0.)
arg_parser.add_argument('--random-state-frac', type=float, default=0.)
arg_parser.add_argument('--random-state-type', type=int, default=0)
arg_parser.add_argument('--seed', type=int, default=0)
arg_parser.add_argument('--tag', type=str, default=None)
args = arg_parser.parse_args()

num_worlds = args.num_worlds
num_steps = args.num_steps
discount = args.discount
policy = args.policy
random_act_frac = args.random_act_frac
random_state_frac = args.random_state_frac
random_state_type = args.random_state_type

run_name = f"qlearngrid__{num_worlds}__{num_steps}__{args.seed}__{int(time.time())}_torch"

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
visit_dict[start_cell[0], start_cell[1], :] = 1

# Create queue for DP
# curr_obs = torch.tensor([[5,4]]).repeat(num_worlds, 1)
curr_rewards = torch.zeros(num_worlds)
start_cell = torch.tensor(start_cell)[None, :]

# Valid states to visit are anything other than the end cell and walls
valid_states = torch.nonzero(1. - walls).type(torch.int)
#valid_states = valid_states[valid_states != end_cell]
#print(end_cell.shape, valid_states.shape)
valid_states = valid_states[(valid_states != torch.tensor(end_cell)).sum(axis=1) > 0]

wandb.init(
    project="cleanRL",
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

start_time = time.time()

for i in range(num_steps):
    # Account for random restarts or sampling into random or unvisited or underexplored states
    if random_state_frac > 0.:
        # Random restarts
        restarts = torch.rand(num_worlds) < random_state_frac
        if random_state_type == 0:
            grid_world.observations[restarts, :] = start_cell.repeat(torch.sum(restarts), 1).type(torch.int)
        elif random_state_type == 1:
            #grid_world.observations[restarts, :] = torch.cat([torch.randint(0, walls.shape[0], size=(torch.sum(restarts),1)), torch.randint(0, walls.shape[1], size=(torch.sum(restarts),1))], dim=1).type(torch.int)
            grid_world.observations[restarts, :] = valid_states[torch.randint(0, valid_states.shape[0], size=(torch.sum(restarts),)), :]
        elif random_state_type == 2:
            # Sample only from already-visited but underexplored states
            visited_states = torch.nonzero(visit_dict).type(torch.int)
            grid_world.observations[restarts, :] = visited_states[torch.randint(0, visited_states.shape[0], size=(torch.sum(restarts),)), :2]
        curr_rewards[restarts] = 0

    curr_states = grid_world.observations.clone()

    #grid_world.actions[:, 0] = torch.randint(0, 4, size=(num_worlds,))
    action_qs = q_dict[curr_states[:,0], curr_states[:,1]]
    if policy == "greedy":
        grid_world.actions[:, 0] = torch.argmax(action_qs, dim=1)
    elif policy == "ucb":
        # UCB
        visit_counts = visit_dict[curr_states[:,0], curr_states[:,1]] + 1
        #print(visit_counts[0])
        ucb = action_qs + torch.sqrt((0.5 * np.log(1./0.05) / visit_counts)) # Hoeffding 95% confidence
        #print(action_qs[0])
        #print(curr_states[0])
        #print(ucb[0])
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

    # Write stats
    global_step = (i + 1)*num_worlds
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    writer.add_scalar("charts/avg_value", v_dict[curr_states[rewards_order,0], curr_states[rewards_order,1]].mean().item(), global_step)
    writer.add_scalar("charts/rollout_value", v_dict[start_cell[0,0], start_cell[0,1]], global_step)
    writer.add_scalar("charts/unvisited_states", visit_dict.sum(2).eq(0).sum().item(), global_step)
    writer.add_scalar("charts/underexplored_states", (visit_dict.sum(2) < 10).sum().item(), global_step)
    writer.add_scalar("charts/exploration_variance", visit_dict.sum(2).float().std().item() / visit_dict.sum(2).float().mean().item(), global_step)

plt.imshow(v_dict)
plt.show()

plt.imshow(visit_dict.sum(2))
plt.colorbar()
plt.show()
