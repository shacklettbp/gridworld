import argparse
import madrona_learn
import pathlib
import pickle as pkl
from gridworld import GridWorld
from tabular_policy import TabularPolicy, TabularValue
import torch

class PPOTabularActor(madrona_learn.ActorCritic.DiscreteActor):
    def __init__(self, num_states, num_actions):
        tbl = TabularPolicy(num_states, num_actions, False)
        eval_policy = lambda states: tbl.policy[states.squeeze(-1)]
        super().__init__(eval_policy, [num_actions])
        self.tbl = tbl

class PPOTabularCritic(madrona_learn.ActorCritic.Critic):
    def __init__(self, num_states):
        tbl = TabularValue(num_states)
        eval_V = lambda states: tbl.V[states]
        super().__init__(eval_V)
        self.tbl = tbl

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-updates', type=int, required=True)
arg_parser.add_argument('--lr', type=float, default=3e-4)
arg_parser.add_argument('--gamma', type=float, default=0.998)
arg_parser.add_argument('--steps-per-update', type=int, default=100)
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--cpu-sim', action='store_true')

args = arg_parser.parse_args()

with open(pathlib.Path(__file__).parent / "world_configs/test_world.pkl", 'rb') as handle:
    start_cell, end_cell, rewards, walls = pkl.load(handle)

# Start from the end cell...
world = GridWorld(args.num_worlds, start_cell, end_cell, rewards, walls)

if torch.cuda.is_available():
    dev = torch.device(f'cuda:{args.gpu_id}')
elif torch.backends.mps.is_available() and False:
    dev = torch.device('mps')
else:
    dev = torch.device('cpu')

num_rows = walls.shape[0]
num_cols = walls.shape[1]
num_states = num_rows * num_cols
num_actions = 4 

def to1D(obs):
    with torch.no_grad():
        obs_1d = obs[:, 1] * num_cols + obs[:, 0]
        return obs_1d.view(*obs.shape[:-1], 1)

policy = madrona_learn.ActorCritic(
    backbone = lambda obs: to1D(obs),
    actor = PPOTabularActor(num_states, num_actions),
    critic = PPOTabularCritic(num_states),
)

madrona_learn.train(madrona_learn.SimInterface(
        step = lambda: world.step(),
        obs = [world.observations],
        actions = world.actions,
        dones = world.dones,
        rewards = world.rewards,
    ),
    madrona_learn.TrainConfig(
        num_updates = args.num_updates,
        gamma = args.gamma,
        gae_lambda = 0.95,
        lr = args.lr,
        steps_per_update = args.steps_per_update,
        ppo = madrona_learn.PPOConfig(
            num_mini_batches=1,
            clip_coef=0.2,
            value_loss_coef=1.0,
            entropy_coef=0.01,
            max_grad_norm=5,
            num_epochs=1,
            clip_value_loss=True,
        ),
    ),
    policy,
    dev = dev,
)
