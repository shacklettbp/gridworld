import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn
from flax.core import frozen_dict

import madrona_learn
from madrona_learn import (
    TrainConfig, CustomMetricConfig, PPOConfig,
)

from madrona_learn.rnn import LSTM
from madrona_learn.utils import init_recurrent_states
from madrona_learn.eval import InferenceState, inference_loop

from madrona_learn import (
    ActorCritic, BackboneShared, BackboneSeparate,
    BackboneEncoder, RecurrentBackboneEncoder,
)

from madrona_learn.models import (
    EMANormalizeTree,
    MLP,
    EgocentricSelfAttentionNet,
    DenseLayerDiscreteActor,
    DenseLayerCritic,
)

from madrona_learn.action import DiscreteActionDistributions

import argparse
import pathlib
import pickle as pkl
from gridworld import GridWorld
import warnings
#warnings.filterwarnings("error")
import matplotlib.pyplot as plt
import time
from functools import partial

class PPOTabularActor(nn.Module):
    dtype: jnp.dtype
    num_states: int
    num_actions: int

    @nn.compact
    def __call__(self, states, train=False):
        tbl = self.param('tbl', lambda rng, shape: jnp.zeros(shape),
                         (self.num_states, self.num_actions))
        logits = jnp.asarray(tbl[states.squeeze(-1)], dtype=self.dtype)

        return DiscreteActionDistributions(
            actions_num_buckets = [ self.num_actions ],
            all_logits = logits,
        )


class PPOTabularCritic(nn.Module):
    dtype: jnp.dtype
    num_states: int

    @nn.compact
    def __call__(self, states, train=False):
        tbl = self.param('tbl', lambda rng, shape: jnp.zeros(shape),
                         (self.num_states,))
        return jnp.asarray(tbl[states], dtype=self.dtype)

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
arg_parser.add_argument('--num-channels', type=int, default=1024)
arg_parser.add_argument('--separate-value', action='store_true')
arg_parser.add_argument('--actor-rnn', action='store_true')
arg_parser.add_argument('--critic-rnn', action='store_true')
arg_parser.add_argument('--num-bptt-chunks', type=int, default=1)
arg_parser.add_argument('--pbt-ensemble-size', type=int, default=1)
arg_parser.add_argument('--profile-report', action='store_true')
arg_parser.add_argument('--seed', type=int, default=0)
arg_parser.add_argument('--tag', type=str, default=None)
# Working DNN hyperparams:
# --num-worlds 1024 --num-updates 1000 --dnn --lr 0.001 --entropy-loss-coef 0.1
# --num-worlds 1024 --num-updates 1000 --dnn --lr 0.001 --entropy-loss-coef 0.3 --separate-value
# Alternatives (fast):
# --num-worlds 1024 --num-updates 1000 --dnn --lr 0.001 --entropy-loss-coef 0.3 --steps-per-update 10 --separate-value --num-channels 64 --gamma 0.9 
# --num-worlds 1024 --num-updates 1000 --dnn --lr 0.001 --entropy-loss-coef 0.3 --steps-per-update 10 --separate-value --num-channels 256 --gamma 0.998

args = arg_parser.parse_args()

with open(pathlib.Path(__file__).parent / "world_configs/test_world.pkl", 'rb') as handle:
    start_cell, end_cell, rewards, walls = pkl.load(handle)

world = GridWorld(args.num_worlds, start_cell, end_cell, rewards, walls)

dev = jax.devices()[0]

num_rows = walls.shape[0]
num_cols = walls.shape[1]

run_name = f"ppogrid__{args.num_worlds}__{args.steps_per_update}__{args.seed}__{int(time.time())}_torch"

num_states = num_rows * num_cols
num_actions = 4 

start_time = time.time()

def to1D(obs, train):
    self_obs = obs['self']

    obs_1d = self_obs[:, 0] * num_cols + self_obs[:, 1]
    return obs_1d.reshape(*self_obs.shape[:-1], 1)

sim_step, init_sim_data = world.jax()
init_sim_data_copy = jax.tree_map(jnp.copy, init_sim_data)

init_sim_data = frozen_dict.freeze(init_sim_data)
init_sim_data_copy = frozen_dict.freeze(init_sim_data_copy)

def metrics_cb(metrics, epoch, mb, train_state):
    return metrics

def host_cb(update_idx, metrics, train_state_mgr):
    print(f"Update: {update_idx}")

    metrics.pretty_print()

    return ()

def iter_cb(update_idx, update_time, metrics, train_state_mgr):
    cb = partial(jax.experimental.io_callback, host_cb, ())
    noop = lambda *args: ()

    update_id = update_idx + 1
    lax.cond(jnp.logical_or(update_id == 1, update_id % 10 == 0), cb, noop,
             update_id, metrics, train_state_mgr)
    #cb(update_id, metrics, train_state_mgr)

if args.dnn:
    def process_obs(obs):
        div = torch.tensor([[1 / float(num_rows), 1 / float(num_cols)]],
            dtype=torch.float32, device=obs.device)
        return obs.float() * div

    def make_rnn_encoder(num_channels):
        return RecurrentBackboneEncoder(
            net = MLP(
                input_dim = 2,
                num_channels = num_channels // 2,
                num_layers = 2,
            ),
            rnn = LSTM(
                in_channels = num_channels // 2,
                hidden_channels = num_channels,
                num_layers = 1,
            )
        )

    def make_normal_encoder(num_channels):
        return BackboneEncoder(MLP(
            input_dim = 2,
            num_channels = num_channels,
            num_layers = 2,
        ))

    if args.separate_value:
        # Use different channel dims just to make sure everything is being passed correctly
        backbone = BackboneSeparate(
            process_obs = process_obs,
            actor_encoder = make_rnn_encoder(args.num_channels) if args.actor_rnn else make_normal_encoder(args.num_channels),
            critic_encoder = make_rnn_encoder(args.num_channels // 2) if args.critic_rnn else make_normal_encoder(args.num_channels // 2),
        )

        actor_input = args.num_channels 
        critic_input = args.num_channels // 2
    else:
        assert(args.actor_rnn == args.critic_rnn)

        backbone = BackboneShared(
            process_obs = process_obs,
            encoder = make_rnn_encoder(args.num_channels) if args.actor_rnn else make_normal_encoder(args.num_channels),
        )

        actor_input = args.num_channels
        critic_input = args.num_channels

    policy = ActorCritic(
        backbone = backbone,
        actor = LinearLayerDiscreteActor([num_actions], actor_input),
        critic = LinearLayerCritic(critic_input),
    )
else:
    policy = ActorCritic(
        backbone = BackboneShared(
            prefix = to1D,
            encoder = BackboneEncoder(lambda x, train: x),
        ),
        actor = PPOTabularActor(dtype=(jnp.float16 if args.fp16 else jnp.float32),
                                num_states=num_states, num_actions=num_actions),
        critic = PPOTabularCritic(dtype=(jnp.float16 if args.fp16 else jnp.float32),
            num_states=num_states),
    )

cfg = TrainConfig(
    num_worlds = args.num_worlds,
    team_size = 1,
    num_teams = 1,
    num_updates = args.num_updates,
    steps_per_update = args.steps_per_update,
    num_bptt_chunks = args.num_bptt_chunks,
    lr = args.lr,
    gamma = args.gamma,
    gae_lambda = 0.95,
    algo = PPOConfig(
        num_mini_batches=1,
        clip_coef=0.2,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_loss_coef,
        max_grad_norm=0.5,
        num_epochs=1,
        clip_value_loss=False,
    ),
    mixed_precision = args.fp16,
    seed = 5,
    pbt_ensemble_size = args.pbt_ensemble_size,
    pbt_history_len = 1,
)

trained = madrona_learn.train(dev, cfg, sim_step, init_sim_data, policy,
    iter_cb, CustomMetricConfig(register_metrics = lambda metrics: metrics))

# Reset sim state
init_sim_data = init_sim_data_copy.copy({
    'resets': jnp.ones((args.num_worlds, 1), dtype=jnp.int32),
})
init_sim_data = sim_step(init_sim_data)
init_sim_data = frozen_dict.freeze(init_sim_data)
init_sim_data = init_sim_data.copy({
    'resets': jnp.zeros((args.num_worlds, 1), dtype=jnp.int32),
})

cur_rnn_states = init_recurrent_states(
    policy, args.num_worlds, args.pbt_ensemble_size)

def policy_fwd(state, rnn_states, obs):
    return state.apply_fn(
        {
            'params': state.params,
            'batch_stats': state.batch_stats,
        },
        rnn_states,
        obs,
        train=False,
        method='debug',
    )

def sweep_states(state, rnn_states):
    # Note these collected values are pretty much meaningless with a recurrent policy
    V = jnp.zeros((num_rows, num_cols),
                  dtype=jnp.float32)
    action_probs = jnp.zeros((num_rows, num_cols, num_actions),
                             dtype=jnp.float32)
    
    logits = jnp.zeros((num_rows, num_cols, num_actions),
                        dtype=jnp.float32)

    for r in range(num_rows):
        for c in range(num_cols):
            obs = {
                'self': jnp.array([[r, c]], dtype=jnp.int32),
            }

            step_actions, step_action_probs, step_action_logits, value, rnn_states = policy_fwd(state, rnn_states, obs)
            V = V.at[r, c].set(value[0, 0])
            action_probs = action_probs.at[r, c, :].set(step_action_probs[0][0])
            logits = logits.at[r, c, :].set(step_action_logits[0][0])

    return V, action_probs, logits, rnn_states

sweep_states = jax.jit(jax.vmap(sweep_states))

V, action_probs, logits, cur_rnn_states = sweep_states(
    trained.train_states, cur_rnn_states)

reset_rnn_state_fn = jax.jit(jax.vmap(trained.train_states.rnn_reset_fn, axis_size=cfg.pbt_ensemble_size))

cur_rnn_states = reset_rnn_state_fn(cur_rnn_states,
                                    jnp.ones((args.num_worlds, 1), dtype=jnp.bool_))

def infer_host_cb(obs, actions, action_probs, values, dones, rewards):
    print(obs['self'][0, 0])
    print("Action:", actions[0, 0])
    print("Action Probs:", action_probs[0][0, 0])
    print("Reward:", rewards[0, 0])
    print()

def infer_step_cb(obs, actions, action_probs, action_logits, values, dones, rewards):
    cb = partial(jax.experimental.io_callback, infer_host_cb, ())
    cb(obs, actions, action_probs, values, dones, rewards)

inference_state = InferenceState(
    sim_step_fn = sim_step,
    user_step_cb = infer_step_cb,
    rnn_states = cur_rnn_states,
    sim_data = init_sim_data,
    reorder_idxs = None,
)

inference_loop_wrapper = partial(
    inference_loop, 16, args.pbt_ensemble_size, trained)

jax.jit(inference_loop_wrapper)(inference_state)

print(f"Grid size: {num_rows} x {num_cols}")
print(rewards)
print(walls)

for policy_idx in range(cfg.pbt_ensemble_size):
    print(f"Policy: {policy_idx}")

    print("\nV:")
    for r in range(num_rows):
        for c in range(num_cols):
            print(f"{V[policy_idx, r, c]: .2f} ", end='')
        print()
    
    print("\nAction probs:")
    for r in range(num_rows):
        for c in range(num_cols):
            probs = action_probs[policy_idx, r, c]
            print(f"  {r}, {c}: [{probs[0]:.2f} {probs[1]:.2f} {probs[2]:.2f} {probs[3]:.2f}]")
    
    print("\nLogits:")
    for r in range(num_rows):
        for c in range(num_cols):
            l = logits[policy_idx, r, c]
            print(f"  {r}, {c}: [{l[0]:.2f} {l[1]:.2f} {l[2]:.2f} {l[3]:.2f}]")

#if args.plot and not args.dnn:
#    plt.imshow(policy.actor.tbl.policy[:,0].reshape(num_rows, num_cols).cpu().detach().numpy())
#    plt.show()
#    plt.imshow(policy.actor.tbl.policy[:,1].reshape(num_rows, num_cols).cpu().detach().numpy())
#    plt.show()
#    plt.imshow(policy.actor.tbl.policy[:,2].reshape(num_rows, num_cols).cpu().detach().numpy())
#    plt.show()
#    plt.imshow(policy.actor.tbl.policy[:,3].reshape(num_rows, num_cols).cpu().detach().numpy())
#    plt.show()
#    print(policy.actor.tbl.policy[:,0])
#    print(policy.actor.tbl.policy[:,0].detach().numpy().reshape(num_cols, num_rows).swapaxes(0,1).copy().flatten())
#    '''
#    plt.imshow(policy.actor.tbl.policy[:,0].detach().numpy().reshape(num_cols, num_rows).swapaxes(0,1).copy().reshape(num_cols, num_rows))
#    plt.show()
#    plt.imshow(policy.critic.tbl.V.reshape(num_rows, num_cols).cpu().detach().numpy())
#    plt.show()
#    '''
