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
        super().__init__([num_actions], eval_policy)
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
arg_parser.add_argument('--lr', type=float, default=0.01)
arg_parser.add_argument('--gamma', type=float, default=0.998)
arg_parser.add_argument('--steps-per-update', type=int, default=50)
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--entropy-loss-coef', type=float, default=0.3)
arg_parser.add_argument('--value-loss-coef', type=float, default=0.5)
arg_parser.add_argument('--cpu-sim', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')
arg_parser.add_argument('--plot', action='store_true')
arg_parser.add_argument('--dnn', action='store_true')
# Args for DNN:
# --num-worlds 1024 --num-updates 1000 --dnn --lr 0.001 --entropy-loss-coef 0.1

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

if args.dnn:
    def process_obs(obs):
        div = torch.tensor([[1 / float(num_rows), 1 / float(num_cols)]],
            dtype=torch.float32, device=obs.device)
        return obs.float() * div

    policy = madrona_learn.ActorCritic(
        backbone = madrona_learn.models.SmallMLPBackbone(
            process_obs_fn = process_obs,
            input_dim = 2,
            num_channels = 1024,
        ),
        actor = madrona_learn.models.LinearLayerDiscreteActor([num_actions], 1024),
        critic = madrona_learn.models.LinearLayerCritic(1024),
    )
else:
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
            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_loss_coef,
            max_grad_norm=0.5,
            num_epochs=1,
            clip_value_loss=False,
        ),
        mixed_precision = args.fp16,
    ),
    policy,
    dev = dev,
)

world.force_reset[0] = 1
world.step()
print()

V = torch.zeros(num_rows, num_cols,
                dtype=torch.float32, device=torch.device('cpu'))
action_probs = torch.zeros(num_rows, num_cols, num_actions,
                            dtype=torch.float32, device=torch.device('cpu'))

logits = torch.zeros(num_rows, num_cols, num_actions,
                            dtype=torch.float32, device=torch.device('cpu'))

with torch.no_grad():
    for r in range(num_rows):
        for c in range(num_cols):
            action_dist, value, *rnn = trained(torch.tensor([[r, c]]).cpu())
            V[r, c] = value[0, 0]
            action_probs[r, c, :] = action_dist.probs()[0][0]
            logits[r, c, :] = action_dist.dists[0].logits[0]

    for i in range(10):
        print("Obs:   ", world.observations[0])
        trained.eval_infer(world.actions[0:1], world.observations[0:1])
        print("Action:", world.actions[0].cpu().numpy())
        world.step()
        print("Reward:", world.rewards[0].cpu().numpy())
        print()

print(f"Grid size: {num_rows} x {num_cols}")
print(rewards)
print(walls)
print("\nV:")

for r in range(num_rows):
    for c in range(num_cols):
        print(f"{V[r, c]: .2f} ", end='')
    print()

print("\nAction probs:")
for r in range(num_rows):
    for c in range(num_cols):
        probs = action_probs[r, c]
        print(f"  {r}, {c}: [{probs[0]:.2f} {probs[1]:.2f} {probs[2]:.2f} {probs[3]:.2f}]")

print("\nLogits:")
for r in range(num_rows):
    for c in range(num_cols):
        l = logits[r, c]
        print(f"  {r}, {c}: [{l[0]:.2f} {l[1]:.2f} {l[2]:.2f} {l[3]:.2f}]")

if args.plot and not args.dnn:
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
