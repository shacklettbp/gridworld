#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for seed in {1..5}
do
    echo $seed
    for random_act_frac in 0.0 0.05 0.1 0.3 1.0
    do
        for num_worlds in 100 1000 10000 100000
        do
            let num_steps=50000*100/$num_worlds
            MADRONA_MWGPU_KERNEL_CACHE=/tmp/gridworldcache python "./scripts/q_learning_tabularworld_strategies.py" \
                --num-worlds $num_worlds \
                --num-steps $num_steps \
                --seed $seed \
                --policy ucb \
                --discount 1.0 \
                --random-act-frac $random_act_frac \
                --random-state-frac 0.0 \
                --random-state-type 0 \
                --tag "montezuma_sweep" \
                --reward-scaling 100. \
                --world-name "/data/rl/effective-horizon/downloaded_tables/bridge_dataset/mdps/montezuma_revenge_15_fs24/consolidated.npz"
        done
    done
    
    for random_act_frac in 0.0 0.05 0.1 0.3 1.0
    do
        for num_worlds in 100 1000 10000 100000
        do
            let num_steps=50000*100/$num_worlds
            for random_restart_frac in 0.1 1.0
            do
                for random_restart_type in 0 1 2
                do
                    MADRONA_MWGPU_KERNEL_CACHE=/tmp/gridworldcache python "./scripts/q_learning_tabularworld_strategies.py" \
                        --num-worlds $num_worlds \
                        --num-steps $num_steps \
                        --seed $seed \
                        --policy ucb \
                        --discount 1.0 \
                        --random-act-frac $random_act_frac \
                        --random-state-frac $random_restart_frac \
                        --random-state-type $random_restart_type \
                        --tag "montezuma_sweep" \
                        --reward-scaling 100. \
                        --world-name "/data/rl/effective-horizon/downloaded_tables/bridge_dataset/mdps/montezuma_revenge_15_fs24/consolidated.npz"
                done
            done
        done
    done
done