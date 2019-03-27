#!/bin/bash

repeat_num=3
start_index=0

time_steps=675000
cores_to_use=-1
xnum=50
ynum=50

padding_fraction=0.4
n_components=15

n_comp_to_use=15
cma_num_timesteps=1000000
eval_num_timesteps=2048
even_check_point_num=5
normalize=True

plot_final_param_plane () {
    local run=$1
    local env=$2

    echo "Welcome to plot_final_param_plane: run number $env  $run"

    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps\
                                    --n_comp_to_use=$n_comp_to_use --n_components=$n_components --normalize=$normalize
#    python -m stable_baselines.low_dim_analysis.just_pca \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --n_components=$n_components --n_comp_to_use=$n_comp_to_use

}
plot_mean_param_plane () {
    local run=$1
    local env=$2

    echo "Welcome to plot_mean_param_plane: run number  $env $run"

    python -m stable_baselines.low_dim_analysis.plot_return_real_pca_plane \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps\
                                    --n_comp_to_use=$n_comp_to_use --n_components=$n_components --normalize=$normalize
#    python -m stable_baselines.low_dim_analysis.just_pca \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --n_components=$n_components --n_comp_to_use=$n_comp_to_use

}
plot_final_plane_with_9_10 () {
    local run=$1
    local env=$2

    echo "Welcome to plot_mean_param_plane: run number  $env $run"

    python -m stable_baselines.low_dim_analysis.plot_other_pca_plane_return_landscape \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps\
                                    --n_comp_to_use=$n_comp_to_use --n_components=$n_components --normalize=$normalize
#    python -m stable_baselines.low_dim_analysis.just_pca \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --n_components=$n_components --n_comp_to_use=$n_comp_to_use

}
run () {
    local run=$1
    local env=$2

    echo "Welcome to RUN: run number  $env $run"
    python -m stable_baselines.ppo2.run_mujoco --env=$env --num-timesteps=$time_steps --run_num=$run --normalize=$normalize

}

cma_once () {
    local run=$1
    local env=$2

    echo "Welcome to cma: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.cmaes.cmaes \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
                                    --cores_to_use=$cores_to_use \
                                    --n_components=$n_components --cma_num_timesteps=$cma_num_timesteps\
                                    --n_comp_to_use=$n_comp_to_use --eval_num_timesteps=$eval_num_timesteps\
                                     --normalize=$normalize
}

next_n_once () {
    local run=$1
    local env=$2

    echo "Welcome to next_n: run number $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.low_dim_analysis.next_n \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
                                    --n_comp_to_use=$n_comp_to_use --n_components=$n_components \
                                    --even_check_point_num=$even_check_point_num --normalize=$normalize
}


for (( run_num=$start_index; run_num<$repeat_num; run_num++ ))
do
    sleep 1; run $run_num 'Hopper-v2' & sleep 1; ps
    sleep 1; run $run_num 'Walker2d-v2' & sleep 1; ps

done

wait

for (( run_num=$start_index; run_num<$repeat_num; run_num++ ))
do

    sleep 1; plot_final_plane_with_9_10 $run_num 'Hopper-v2' ; sleep 1; ps
    sleep 1; plot_final_plane_with_9_10 $run_num 'Walker2d-v2' ; sleep 1; ps

    sleep 1; plot_mean_param_plane $run_num 'Hopper-v2' ; sleep 1; ps
    sleep 1; plot_mean_param_plane $run_num 'Walker2d-v2' ; sleep 1; ps

    sleep 1; plot_final_param_plane $run_num 'Hopper-v2' ; sleep 1; ps
    sleep 1; plot_final_param_plane $run_num 'Walker2d-v2' ; sleep 1; ps

    sleep 1; cma_once $run_num 'Hopper-v2' ; sleep 1; ps
    sleep 1; cma_once $run_num 'Walker2d-v2' ; sleep 1; ps

    sleep 1; next_n_once $run_num 'Hopper-v2' ; sleep 1; ps
    sleep 1; next_n_once $run_num 'Walker2d-v2' ; sleep 1; ps
done
