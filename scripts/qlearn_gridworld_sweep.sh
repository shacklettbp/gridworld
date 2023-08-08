#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for seed in {1..5}
do
    echo $seed
    for num_worlds in 1 10 100 1000 10000
    do
        let num_steps=10000/$num_worlds
        echo $num_steps
        for random_act_frac in 0.0 0.2 0.4 0.6 0.8 1.0
        do
            echo $random_act_frac
            for random_restart_frac in 0.0 0.02 0.04 0.06 0.08 0.1
            do
                for random_restart_type in 0 1 2
                do
                    MADRONA_MWGPU_KERNEL_CACHE=/tmp/gridworldcache python "$SCRIPT_DIR/q_learning_gridworld_strategies.py" \
                        --num-worlds $num_worlds \
                        --num-steps $num_steps \
                        --seed $seed \
                        --policy ucb \
                        --discount 0.9 \
                        --random-act-frac $random_act_frac \
                        --random-state-frac $random_restart_frac \
                        --random-state-type $random_restart_type \
                        --tag "qlearn_sweep"
                done
            done
        done
    done
done