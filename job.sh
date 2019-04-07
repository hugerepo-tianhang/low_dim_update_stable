#!/bin/bash


cores_to_use=-1
xnum=3
ynum=3

padding_fraction=0.4
n_components=15

n_comp_to_use=2
cma_num_timesteps=15000
ppos_num_timesteps=1500000
eval_num_timesteps=1024
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
    local nminibatches=$3
    local n_steps=$4
    local time_steps=$5

    echo "Welcome to RUN: run number  $env $run"
    python -m stable_baselines.ppo2.run_mujoco --env=$env --num-timesteps=$time_steps\
            --run_num=$run --normalize=$normalize --nminibatches=$nminibatches\
            --n_steps=$n_steps

}

cma_once () {
    local run=$1
    local env=$2
    local nminibatches=$3
    local n_steps=$4
    local time_steps=$5
    local use_IPCA=$6
    local chunk_size=$7

    echo "Welcome to cma: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.cmaes.cmaes \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
                                    --cores_to_use=$cores_to_use \
                                    --xnum=$xnum --ynum=$ynum\
                                    --padding_fraction=$padding_fraction \
                                    --n_components=$n_components --cma_num_timesteps=$cma_num_timesteps\
                                    --n_comp_to_use=$n_comp_to_use --eval_num_timesteps=$eval_num_timesteps\
                                     --normalize=$normalize --nminibatches=$nminibatches\
                                     --n_steps=$n_steps --use_IPCA=$use_IPCA --chunk_size=$chunk_size
}
ppos_once () {
    local run=$1
    local env=$2
    local nminibatches=$3
    local n_steps=$4
    local time_steps=$5

    echo "Welcome to cma: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.ppo_subspace.ppo_sub \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
                                    --cores_to_use=$cores_to_use \
                                    --n_components=$n_components --ppos_num_timesteps=$ppos_num_timesteps\
                                    --n_comp_to_use=$n_comp_to_use \
                                     --normalize=$normalize --nminibatches=$nminibatches\
                                    --n_steps=$n_steps
}
next_n_once () {
    local run=$1
    local env=$2
    local nminibatches=$3
    local n_steps=$4
    local time_steps=$5

    echo "Welcome to next_n: run number $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.low_dim_analysis.next_n \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
                                    --n_comp_to_use=$n_comp_to_use --n_components=$n_components \
                                    --even_check_point_num=$even_check_point_num --normalize=$normalize \
                                    --nminibatches=$nminibatches --n_steps=$n_steps
}

#sleep 1; ppos_once 0 'Walker2d-v2' 8 2048; sleep 1; ps
#
#sleep 1; ppos_once 0 'Hopper-v2' 8 2048; sleep 1; ps
#sleep 1; ppos_once 0 'Walker2d-v2' 8 2048; sleep 1; ps

#
#sleep 1; run 0 'DartHopper-v1' 512 2048 1000000& sleep 1; ps
#sleep 1; run 0 'DartHopper-v1' 2 2048& sleep 1; ps
#sleep 1; run 0 'DartHopper-v1' 32 2048 1000000& sleep 1; ps
##sleep 1; run 0 'DartHopper-v1' 256 2048 1000000& sleep 1; ps
##sleep 1; run 0 'Hopper-v2' 32 2048& sleep 1; ps
##
#sleep 1; run 0 'DartWalker2d-v1' 512 2048 675000& sleep 1; ps
###sleep 1; run 0 'DartWalker2d-v1' 2 2048& sleep 1; ps
##sleep 1; run 0 'DartWalker2d-v1' 32 2048& sleep 1; ps
##sleep 1; run 0 'DartWalker2d-v1' 256 2048& sleep 1; ps
##sleep 1; run 0 'Walker2d-v2' 32 2048& sleep 1; ps
##sleep 1; run 0 'Walker2d-v2' 16 2048& sleep 1; ps
##sleep 1; run 0 'Walker2d-v2' 8 2048& sleep 1; ps
#sleep 1; run 0 'DartWalker2d-v1' 32 2048 675000& sleep 1; ps

wait

#sleep 1; ppos_once 0 'Hopper-v2' 8 2048; sleep 1; ps

#sleep 1; cma_once 0 'DartHopper-v1' 512 2048 1000000 True 50000; sleep 1; ps
sleep 1; cma_once 0 'DartHopper-v1' 32 2048 1000000 True 50000; sleep 1; ps
#sleep 1; cma_once 0 'DartHopper-v1' 32 2048 1000000; sleep 1; ps

#sleep 1; ppos_once 0 'Walker2d-v2' 8 2048; sleep 1; ps


#sleep 1; cma_once 0 'DartWalker2d-v1' 512 2048 675000; sleep 1; ps
sleep 1; cma_once 0 'DartWalker2d-v1' 32 2048 675000; sleep 1; ps
#sleep 1; cma_once 0 'DartWalker2d-v1' 32 2048 675000; sleep 1; ps



#
#sleep 1; next_n_once 0 'DartHopper-v1' 512 2048; sleep 1; ps
#sleep 1; next_n_once 0 'DartHopper-v1' 256 2048; sleep 1; ps
##sleep 1; cma_once 0 'DartHopper-v1' 512 2048; sleep 1; ps
#
##sleep 1; next_n_once 0 'Hopper-v2' 32 2048; sleep 1; ps
#
#sleep 1; next_n_once 0 'DartWalker2d-v1' 512 2048; sleep 1; ps
#sleep 1; next_n_once 0 'DartWalker2d-v1' 256 2048; sleep 1; ps
#sleep 1; cma_once 0 'DartWalker2d-v1' 512 2048; sleep 1; ps

#sleep 1; next_n_once 0 'Walker2d-v2' 32 2048; sleep 1; ps
#sleep 1; next_n_once 0 'Walker2d-v2' 16 2048; sleep 1; ps
#sleep 1; next_n_once 0 'Walker2d-v2' 8 2048; sleep 1; ps

