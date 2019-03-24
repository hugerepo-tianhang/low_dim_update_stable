#!/bin/bash

repeat_num=1
start_index=0

time_steps=20000
cores_to_use=-1
xnum=70
ynum=70
eval_num_timesteps=5000
padding_fraction=0.4

run_and_plot_once () {
    local run=$1
    local env=$2

    echo "Welcome run number $run"
    python -m stable_baselines.ppo2.run_mujoco --env=$env --num-timesteps=$time_steps

    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
}


for (( run_num=$start_index; run_num<$repeat_num; run_num++ ))
do
    sleep 5; run_and_plot_once $run_num 'Hopper-v2' ; sleep 1; ps
done

for (( run_num=$start_index; run_num<$repeat_num; run_num++ ))
do
    sleep 5; run_and_plot_once $run_num 'Walker2d-v2' ; sleep 1; ps
done
