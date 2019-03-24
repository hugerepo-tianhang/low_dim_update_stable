#!/bin/bash

machine='prod'
alg='ppo2'
repeat_num=2
time_steps=675000
cores_to_use=-1
xnum=50
ynum=50
eval_num_timesteps=10000
padding_fraction=0.4

run_and_plot_once () {
    local run=$1
    local env=$2

    echo "Welcome run number $run"
#    python -m baselines.run --machine=$machine --alg=$alg --env=$env --network=mlp --num_timesteps=$time_steps

    python -m baselines.low_dim_analysis.plot_return_landscape --machine=$machine --alg=$alg \
                                    --num_timesteps=$time_steps --run_num=$run --env=$env\
                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
}


for (( run_num=0; run_num<$repeat_num; run_num++ ))
do
    sleep 5; run_and_plot_once $run_num 'Hopper-v2' ; sleep 1; ps
done

for (( run_num=0; run_num<$repeat_num; run_num++ ))
do
    sleep 5; run_and_plot_once $run_num 'Walker2d-v2' ; sleep 1; ps
done
