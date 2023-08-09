#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for seed in {1..5}
do
    echo $seed
    for num_worlds in 100 1000 10000
    do
        for entropy_loss_coef in 0.0 0.03 0.06 0.09 0.12 0.15
        do
            python "$SCRIPT_DIR/ppo.py" \
                --num-worlds $num_worlds \
                --num-updates 1000 \
                --seed $seed \
                --entropy-loss-coef $entropy_loss_coef \
                --gamma 0.9 \
                --tag "ppo_grid_sweep_1"
        done
    done
done