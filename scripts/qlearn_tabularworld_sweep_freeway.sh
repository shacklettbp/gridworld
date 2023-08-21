#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for seed in {1..5}
do
    echo $seed
    for random_act_frac in 0.0 0.05 0.1 0.3 1.0
    do
        for num_worlds in 10 100 1000 10000
        do
            MADRONA_MWGPU_KERNEL_CACHE=/tmp/gridworldcache python "./scripts/q_learning_tabularworld_strategies.py" \
                --num-worlds $num_worlds \
                --num-steps 5000 \
                --seed $seed \
                --policy ucb \
                --discount 0.965 \
                --random-act-frac $random_act_frac \
                --random-state-frac 0.0 \
                --random-state-type 0 \
                --tag "freeway_sweep_2" \
                --reward-scaling 1. \
                --world-name "/data/rl/effective-horizon/downloaded_tables/bridge_dataset/mdps/freeway_10_fs30/consolidated.npz"
        done
    done
    
    for random_act_frac in 0.0 0.05 0.1 0.3 1.0
    do
        for num_worlds in 10 100 1000 10000
        do
            for random_restart_frac in 0.1 1.0
            do
                for random_restart_type in 0 1 2
                do
                    MADRONA_MWGPU_KERNEL_CACHE=/tmp/gridworldcache python "./scripts/q_learning_tabularworld_strategies.py" \
                        --num-worlds $num_worlds \
                        --num-steps 5000 \
                        --seed $seed \
                        --policy ucb \
                        --discount 0.965 \
                        --random-act-frac $random_act_frac \
                        --random-state-frac $random_restart_frac \
                        --random-state-type $random_restart_type \
                        --tag "freeway_sweep_2" \
                        --reward-scaling 1. \
                        --world-name "/data/rl/effective-horizon/downloaded_tables/bridge_dataset/mdps/freeway_10_fs30/consolidated.npz"
                done
            done
        done
    done
done