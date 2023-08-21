#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for seed in {1..5}
do
    echo $seed
    for random_act_frac in 0.1
    do
        for num_worlds in 10 100 1000 10000
        do
            for random_state_type in 2 3
            do
                for policy_eval in False True
                do
                    MADRONA_MWGPU_KERNEL_CACHE=/tmp/gridworldcache python "./scripts/q_learning_tabularworld_strategies.py" \
                        --num-worlds $num_worlds \
                        --num-steps 5000 \
                        --seed $seed \
                        --policy ucb \
                        --discount 0.965 \
                        --random-act-frac $random_act_frac \
                        --random-state-frac 0.1 \
                        --random-state-type $random_state_type \
                        --random-state-pow 1.0 \
                        --tag "freeway_sweep_explore" \
                        --reward-scaling 1. \
                        --world-name "/data/rl/effective-horizon/downloaded_tables/bridge_dataset/mdps/freeway_10_fs30/consolidated.npz" \
                        --policy-eval $policy_eval
                done
            done
        done
    done
done