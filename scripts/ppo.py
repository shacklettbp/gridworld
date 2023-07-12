import argparse
import madrona_learn
import pathlib
import pickle as pkl
from gridworld import GridWorld
from tabular_policy import TabularPolicy, TabularValue
import torch
import warnings
warnings.filterwarnings("error")
import matplotlib.pyplot as plt

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
arg_parser.add_argument('--lr', type=float, default=0.1)
arg_parser.add_argument('--gamma', type=float, default=0.9)
arg_parser.add_argument('--steps-per-update', type=int, default=50)
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--cpu-sim', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')
arg_parser.add_argument('--plot', action='store_true')

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
        obs_1d = obs[:, 0] * num_cols + obs[:, 1]
        return obs_1d.view(*obs.shape[:-1], 1)

policy = madrona_learn.ActorCritic(
    backbone = to1D,
    actor = PPOTabularActor(num_states, num_actions),
    critic = PPOTabularCritic(num_states),
)

trained = madrona_learn.train(madrona_learn.SimInterface(
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
        mixed_precision = args.fp16,
    ),
    policy,
    dev = dev,
)

print("\nAction probs:")
for i in range(policy.actor.tbl.policy.shape[0]):
    probs = madrona_learn.DiscreteActionDistributions([num_actions], policy.actor.tbl.policy[i].unsqueeze(0)).dists[0].probs.detach().numpy()[0]

    row = i // num_cols
    col = i % num_cols

    print(f"  {row}, {col}: [{probs[0]:.2f} {probs[1]:.2f} {probs[2]:.2f} {probs[3]:.2f}]")

print(f"Grid size: {num_rows} x {num_cols}")
print(rewards)
print(walls)
print("\nV:")

V = policy.critic.tbl.V.view(num_rows, num_cols)
for r in range(num_rows):
    for c in range(num_cols):
        print(f"{V[r, c]: .2f} ", end='')
    print()

world.force_reset[0] = 1
world.step()
print("Initial Obs: ", world.observations[0])
print()

with torch.no_grad():
    for i in range(10):
        trained.actor.infer(to1D(world.observations[0:1]), world.actions[0:1])
        print("Action:", world.actions[0].cpu().numpy())
        world.step()
        print("Obs:   ",   world.observations[0].cpu().numpy())
        print("Reward:", world.rewards[0].cpu().numpy())
        print()

if args.plot:
    plt.imshow(policy.actor.tbl.policy[:,0].reshape(num_rows, num_cols).cpu().detach().numpy())
    plt.show()
    plt.imshow(policy.actor.tbl.policy[:,1].reshape(num_rows, num_cols).cpu().detach().numpy())
    plt.show()
    plt.imshow(policy.actor.tbl.policy[:,2].reshape(num_rows, num_cols).cpu().detach().numpy())
    plt.show()
    plt.imshow(policy.actor.tbl.policy[:,3].reshape(num_rows, num_cols).cpu().detach().numpy())
    plt.show()
    print(policy.actor.tbl.policy[:,0])
    print(policy.actor.tbl.policy[:,0].detach().numpy().reshape(num_cols, num_rows).swapaxes(0,1).copy().flatten())
    '''
    plt.imshow(policy.actor.tbl.policy[:,0].detach().numpy().reshape(num_cols, num_rows).swapaxes(0,1).copy().reshape(num_cols, num_rows))
    plt.show()
    plt.imshow(policy.critic.tbl.V.reshape(num_rows, num_cols).cpu().detach().numpy())
    plt.show()
    '''
