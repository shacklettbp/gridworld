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

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-steps', type=int, required=True)
arg_parser.add_argument('--reward-scaling', type=float, default=1.)
arg_parser.add_argument('--discount', type=float, default=0.99)
arg_parser.add_argument('--policy', type=str, default='ucb')
arg_parser.add_argument('--random-act-frac', type=float, default=0.)
arg_parser.add_argument('--random-state-frac', type=float, default=0.)
arg_parser.add_argument('--random-state-type', type=int, default=0)
arg_parser.add_argument('--seed', type=int, default=0)
arg_parser.add_argument('--tag', type=str, default=None)
arg_parser.add_argument('--world-name', type=str, default="/data/rl/effective-horizon/path/to/store/mdp/consolidated_ignore_screen.npz")
args = arg_parser.parse_args()

num_worlds = args.num_worlds
num_steps = args.num_steps
discount = args.discount
policy = args.policy
random_act_frac = args.random_act_frac
random_state_frac = args.random_state_frac
random_state_type = args.random_state_type

run_name = f"qlearngrid__{num_worlds}__{num_steps}__{args.seed}__{int(time.time())}_torch"

# Start from the end cell...
device = 'cuda'
world = TabularWorld(args.world_name, num_worlds, device)
#world.vis_world()
num_states = world.transitions.shape[0]
num_actions = world.transitions.shape[1]
max_reward = torch.max(world.transition_rewards) # Critical to UCB...
print(world.transitions)
print(world.transition_rewards)
print(max_reward)

# MODIFICATIONS FOR ATARI
last_reward = max_reward
end_reward = 0
world.transition_rewards /= args.reward_scaling
# END MODIFICATIONS

q_dict = torch.full((num_states, num_actions),0., device = device) # Key: [obs, action], Value: [q]
v_dict = torch.full((num_states,),0., device = device) # Key: obs, Value: v
v_dict[num_states - 1] = end_reward # SHOULD THIS BE COMMENTED?

visit_dict = torch.zeros((num_states, num_actions), dtype=int, device = device) # Key: [obs, action], Value: # visits
visit_dict[0, :] = 1

# Create queue for DP
# curr_obs = torch.tensor([[5,4]]).repeat(num_worlds, 1)
curr_rewards = torch.zeros(num_worlds, device = device)

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
        #print("We should not be here")
        # Random restarts
        restarts = torch.rand(num_worlds, device = device) < random_state_frac
        if random_state_type == 0:
            world.observations[restarts, :] = torch.zeros(torch.sum(restarts), 1, device = device).type(torch.int)
        elif random_state_type == 1:
            #grid_world.observations[restarts, :] = torch.cat([torch.randint(0, walls.shape[0], size=(torch.sum(restarts),1)), torch.randint(0, walls.shape[1], size=(torch.sum(restarts),1))], dim=1).type(torch.int)
            world.observations[restarts, :] = torch.randint(0, num_states - 1, size=(torch.sum(restarts),), device = device).type(torch.int)[:,None]
        elif random_state_type == 2:
            # Sample only from already-visited but underexplored states
            visited_states = torch.nonzero(visit_dict).type(torch.int)
            world.observations[restarts, :] = visited_states[torch.randint(0, visited_states.shape[0], size=(torch.sum(restarts),), device = device), :1]
        curr_rewards[restarts] = 0

    curr_states = world.observations.clone()[:,0]

    #grid_world.actions[:, 0] = torch.randint(0, 4, size=(num_worlds,))
    action_qs = q_dict[curr_states]
    if policy == "greedy":
        world.actions[:, 0] = torch.argmax(action_qs, dim=1)
    elif policy == "ucb":
        # UCB
        visit_counts = visit_dict[curr_states] + 1
        #print(visit_counts[0])
        ucb = action_qs + torch.sqrt((0.5 * np.log(1./0.05) / visit_counts)) # Hoeffding 95% confidence
        #print(action_qs[0])
        #print(curr_states[0])
        #print(ucb[0])
        #print(world.observations)
        #print(world.actions)
        #print(ucb, ucb.shape)
        world.actions[:, 0] = torch.argmax(ucb, dim=1)
    else:
        raise ValueError("Invalid policy")
    
    # Replace portion of actions with random
    if random_act_frac > 0.:
        #print("We should not be here")
        random_rows = torch.rand(num_worlds, device = device) < random_act_frac
        world.actions[random_rows, 0] = torch.randint(0, num_actions, size=(random_rows.sum(),), device = device).type(torch.int)

    curr_actions = world.actions.clone().flatten()

    # Advance simulation across all worlds
    world.step()
    #if i == 0: # bug fix
    #    continue

    dones = world.dones.clone().flatten()
    #print(dones[0])

    next_states = world.observations.clone()[:,0]
    next_rewards = world.rewards.clone().flatten()# * (1 - dones)
    #if next_rewards.sum() > 0:
    #    print("Wow we made it")
    #if dones.sum() > 0:
    #    print("Weird")

    next_states[dones == 1] = num_states - 1

    unique_states, states_count = torch.unique(torch.cat([curr_states[:,None], curr_actions[:,None]],dim=1), dim=0, return_counts=True)

    #print(unique_states, states_count)

    # Clobbering of values prioritizes last assignment so get index sort of curr_rewards
    rewards_order = torch.argsort(curr_rewards)
    #print(rewards_order, q_dict[curr_states[rewards_order], curr_actions], v_dict[next_states[rewards_order]])
    q_dict[curr_states[rewards_order], curr_actions] = torch.max(
        q_dict[curr_states[rewards_order], curr_actions], next_rewards[rewards_order] + discount * v_dict[next_states[rewards_order]] * (1 - dones)
    )
    v_dict[curr_states[rewards_order]] = torch.max(
        v_dict[curr_states[rewards_order]], next_rewards[rewards_order] + discount * v_dict[next_states[rewards_order]] * (1 - dones)
    )
    visit_dict[unique_states[:,0], unique_states[:,1]] += states_count

    curr_rewards = next_rewards * (1 - dones)

    # Write stats
    global_step = (i + 1)*num_worlds
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    writer.add_scalar("charts/avg_value", v_dict[curr_states[rewards_order]].mean().item(), global_step)
    writer.add_scalar("charts/rollout_value", v_dict[0], global_step)
    writer.add_scalar("charts/unvisited_states", visit_dict.sum(1).eq(0).sum().item(), global_step)
    writer.add_scalar("charts/underexplored_states", (visit_dict.sum(1) < 10).sum().item(), global_step)
    writer.add_scalar("charts/exploration_variance", visit_dict.sum(1).float().std().item() / visit_dict.sum(1).float().mean().item(), global_step)

