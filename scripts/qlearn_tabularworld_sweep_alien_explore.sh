#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for seed in {1..5}
do
    echo $seed
    for random_act_frac in 0.1
    do
        for num_worlds in 10000 100000 1000000
        do
            for random_state_type in 3
            do
                for policy_eval in 0 1
                do
                    for rollout_steps in 1 5 10
                    do
                        let num_steps=50000*10000/$num_worlds
                        MADRONA_MWGPU_KERNEL_CACHE=/tmp/gridworldcache python "./scripts/q_learning_tabularworld_strategies.py" \
                            --num-worlds $num_worlds \
                            --num-steps $num_steps \
                            --seed $seed \
                            --policy ucb \
                            --discount 0.93 \
                            --random-act-frac $random_act_frac \
                            --random-state-frac 0.1 \
                            --random-state-type $random_state_type \
                            --random-state-pow 1.0 \
                            --rollout-steps $rollout_steps \
                            --tag "alien_sweep_explore_2" \
                            --reward-scaling 160. \
                            --world-name "/data/rl/effective-horizon/downloaded_tables/bridge_dataset/mdps/alien_10_fs30/consolidated.npz" \
                            --policy-eval $policy_eval
                    done
                done
            done
        done
    done
done