#!/bin/bash


cores_to_use=-1
xnum=50
ynum=50

padding_fraction=0.4
n_components=500

n_comp_to_use=2
cma_num_timesteps=1000000
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
    local origin=$8
    local n_comp_to_use=$9

    echo "Welcome to cma: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.cmaes.cma_n_comp \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
                                    --cores_to_use=$cores_to_use \
                                    --xnum=$xnum --ynum=$ynum\
                                    --padding_fraction=$padding_fraction \
                                    --n_components=$n_components --cma_num_timesteps=$cma_num_timesteps\
                                    --n_comp_to_use=$n_comp_to_use --eval_num_timesteps=$eval_num_timesteps\
                                     --normalize=$normalize --nminibatches=$nminibatches\
                                     --n_steps=$n_steps --use_IPCA=$use_IPCA --chunk_size=$chunk_size\
                                     --origin=$origin
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
final_projection_on_mean_performance () {
    local run=$1
    local env=$2
    local nminibatches=$3
    local n_steps=$4
    local time_steps=$5
    local use_IPCA=$6
    local chunk_size=$7
    local n_components=$8

    echo "Welcome to final_projection_on_mean_performance: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.low_dim_analysis.final_projection_on_mean_performance \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env --normalize=$normalize \
                                    --nminibatches=$nminibatches --n_steps=$n_steps\
                                    --cores_to_use=$cores_to_use \
                                    --n_components=$n_components \
                                    --eval_num_timesteps=$eval_num_timesteps\
                                     --use_IPCA=$use_IPCA --chunk_size=$chunk_size
}


first_comp_angle_with_diff () {
    local run=$1
    local env=$2
    local nminibatches=$3
    local n_steps=$4
    local time_steps=$5
    local use_IPCA=$6
    local chunk_size=$7
    local n_comp_to_use=$8
    local n_components=$9

    echo "Welcome to first_comp_angle_with_diff: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.low_dim_analysis.first_comp_angle_with_diff \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env --normalize=$normalize \
                                    --nminibatches=$nminibatches --n_steps=$n_steps\
                                    --cores_to_use=$cores_to_use \
                                    --n_components=$n_components --n_comp_to_use=$n_comp_to_use\
                                    --eval_num_timesteps=$eval_num_timesteps\
                                     --use_IPCA=$use_IPCA --chunk_size=$chunk_size
}

how_many_steps_can_you_go () {
    local run=$1
    local env=$2
    local nminibatches=$3
    local n_steps=$4
    local time_steps=$5
    local use_IPCA=$6
    local chunk_size=$7
    local n_comp_to_use=$8
    local n_components=$9


    echo "Welcome to how_many_steps_can_you_go: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.low_dim_analysis.how_many_steps_can_you_go \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env --normalize=$normalize \
                                    --nminibatches=$nminibatches --n_steps=$n_steps\
                                    --cores_to_use=$cores_to_use \
                                    --n_components=$n_components --n_comp_to_use=$n_comp_to_use\
                                    --eval_num_timesteps=$eval_num_timesteps\
                                     --use_IPCA=$use_IPCA --chunk_size=$chunk_size
}


pcn_vs_final_minus_start () {
    local run=$1
    local env=$2
    local nminibatches=$3
    local n_steps=$4
    local time_steps=$5

    local pc1_chunk_size=$6
    local n_comp_to_use=$7


    echo "Welcome to pcn_vs_final_minus_start: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.low_dim_analysis.pcn_vs_final_minus_start \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env --normalize=$normalize \
                                    --nminibatches=$nminibatches --n_steps=$n_steps\
                                    --n_comp_to_use=$n_comp_to_use\
                                    --pc1_chunk_size=$pc1_chunk_size
}

weighted_pcn_vs_final () {
    local run=$1
    local env=$2
    local nminibatches=$3
    local n_steps=$4
    local time_steps=$5

    local pc1_chunk_size=$6
    local n_comp_to_use=$7


    echo "Welcome to weighted_pcn_vs_final: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.low_dim_analysis.weighted_pcn_vs_final \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env --normalize=$normalize \
                                    --nminibatches=$nminibatches --n_steps=$n_steps\
                                    --n_comp_to_use=$n_comp_to_use\
                                    --pc1_chunk_size=$pc1_chunk_size
}
pcn_latest_vs_final () {
    local run=$1
    local env=$2
    local nminibatches=$3
    local n_steps=$4
    local time_steps=$5

    local pc1_chunk_size=$6
    local n_comp_to_use=$7
    local deque_len=$8

    echo "Welcome to pcn_latest_vs_final: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.low_dim_analysis.pcn_latest_vs_final \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env --normalize=$normalize \
                                    --nminibatches=$nminibatches --n_steps=$n_steps\
                                    --n_comp_to_use=$n_comp_to_use\
                                    --pc1_chunk_size=$pc1_chunk_size --deque_len=$deque_len
}

so_far_pcn_vs_final_minus_current () {
    local run=$1
    local env=$2
    local nminibatches=$3
    local n_steps=$4
    local time_steps=$5

    local pc1_chunk_size=$6
    local n_comp_to_use=$7

    echo "Welcome to so_far_pcn_vs_final_minus_current: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.low_dim_analysis.so_far_pcn_vs_final_minus_current \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env --normalize=$normalize \
                                    --nminibatches=$nminibatches --n_steps=$n_steps\
                                    --n_comp_to_use=$n_comp_to_use\
                                    --pc1_chunk_size=$pc1_chunk_size
}
#sleep 1; ppos_once 0 'Walker2d-v2' 8 2048; sleep 1; ps
#
#sleep 1; ppos_once 0 'Hopper-v2' 8 2048; sleep 1; ps
#sleep 1; ppos_once 0 'Walker2d-v2' 8 2048; sleep 1; ps

#
sleep 1; run 0 'DartReacher-v1' 32 2048 5000& sleep 1; ps
sleep 1; run 0 'DartHalfCheetah-v1' 32 2048 5000& sleep 1; ps
sleep 1; run 0 'DartSnake7Link-v1' 32 2048 5000& sleep 1; ps
#sleep 1; run 0 'DartHopper-v1' 64 2048 1000000& sleep 1; ps
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
#sleep 1; cma_once 0 'DartHopper-v1' 32 2048 1000000 True 50000 "mean_param" 300; sleep 1; ps
#sleep 1; final_projection_on_mean_performance 0 'DartHopper-v1' 32 2048 1000000 True 50000 $n_components; sleep 1; ps

#sleep 1; cma_once 0 'DartHopper-v1' 512 2048 1000000 True 50000; sleep 1; ps
#sleep 1; cma_once 0 'DartHopper-v1' 32 2048 1000000 True 10000 "mean_param" $n_components; sleep 1; ps
#sleep 1; cma_once 0 'DartHopper-v1' 32 2048 1000000 True 10000 "mean_param" 15; sleep 1; ps
##sleep 1; cma_once 0 'DartHopper-v1' 32 2048 1000000 True 10000 "mean_param" $n_components; sleep 1; ps
#
##sleep 1; ppos_once 0 'Walker2d-v2' 8 2048; sleep 1; ps
#sleep 1; weighted_pcn_vs_final 0 'DartWalker2d-v1' 32 2048 675000 1000 100; sleep 1;

sleep 1; so_far_pcn_vs_final_minus_current 0 'DartWalker2d-v1' 32 2048 5000 100 20; sleep 1;
#sleep 1; so_far_pcn_vs_final_minus_current 0 'DartWalker2d-v1' 32 2048 675000 3000 50; sleep 1;
#sleep 1; so_far_pcn_vs_final_minus_current 0 'DartWalker2d-v1' 32 2048 675000 5000 50; sleep 1;
#
#sleep 1; so_far_pcn_vs_final_minus_current 0 'DartWalker2d-v1' 32 2048 675000 1000 10; sleep 1;
#sleep 1; so_far_pcn_vs_final_minus_current 0 'DartWalker2d-v1' 32 2048 675000 3000 10; sleep 1;
#sleep 1; so_far_pcn_vs_final_minus_current 0 'DartWalker2d-v1' 32 2048 675000 5000 10; sleep 1;
#
#sleep 1; so_far_pcn_vs_final_minus_current 0 'DartWalker2d-v1' 32 2048 675000 1000 3; sleep 1;
#sleep 1; so_far_pcn_vs_final_minus_current 0 'DartWalker2d-v1' 32 2048 675000 3000 3; sleep 1;
#sleep 1; so_far_pcn_vs_final_minus_current 0 'DartWalker2d-v1' 32 2048 675000 5000 3; sleep 1;
#
#sleep 1; so_far_pcn_vs_final_minus_current 0 'DartWalker2d-v1' 32 2048 675000 1000 2; sleep 1;
#sleep 1; so_far_pcn_vs_final_minus_current 0 'DartWalker2d-v1' 32 2048 675000 3000 2; sleep 1;
#sleep 1; so_far_pcn_vs_final_minus_current 0 'DartWalker2d-v1' 32 2048 675000 5000 2; sleep 1;
#sleep 1; pcn_latest_vs_final 0 'DartWalker2d-v1' 32 2048 675000 1000 1 20000; sleep 1;
#sleep 1; pcn_latest_vs_final 0 'DartWalker2d-v1' 32 2048 675000 1000 1 10000; sleep 1;
#sleep 1; pcn_latest_vs_final 0 'DartWalker2d-v1' 32 2048 675000 1000 1 5000; sleep 1;
#sleep 1; pcn_latest_vs_final 0 'DartWalker2d-v1' 32 2048 675000 1000 1 3000; sleep 1;
#sleep 1; pcn_latest_vs_final 0 'DartWalker2d-v1' 32 2048 675000 1000 1 1000; sleep 1;
#
#
#sleep 1; pcn_vs_final_minus_start 0 'DartWalker2d-v1' 32 2048 675000 1000 2; sleep 1; ps
#sleep 1; pcn_vs_final_minus_start 0 'DartWalker2d-v1' 32 2048 675000 3000 1; sleep 1; ps
#sleep 1; pcn_vs_final_minus_start 0 'DartWalker2d-v1' 32 2048 675000 3000 2; sleep 1; ps
#sleep 1; first_comp_angle_with_diff 0 'DartWalker2d-v1' 32 2048 1000000 True 10000 $n_components $n_components; sleep 1; ps
#sleep 1; how_many_steps_can_you_go 0 'DartWalker2d-v1' 32 2048 1000000 True 1000 100 $n_components; sleep 1; ps
#sleep 1; final_projection_on_mean_performance 0 'DartWalker2d-v1' 32 2048 675000 True 20000 $n_components; sleep 1; ps
#sleep 1; final_projection_on_mean_performance 0 'DartWalker2d-v1' 512 2048 675000 True 10000 $n_components; sleep 1; ps

#sleep 1; cma_once 0 'DartWalker2d-v1' 512 2048 675000; sleep 1; ps
#sleep 1; cma_once 0 'DartWalker2d-v1' 32 2048 675000 False 0 "mean_param" 50; sleep 1; ps
#sleep 1; cma_once 0 'DartWalker2d-v1' 32 2048 675000 False 0 "mean_param" $n_components; sleep 1; ps
#sleep 1; cma_once 0 'DartWalker2d-v1' 32 2048 675000 True 20000 "mean_param" 500; sleep 1; ps
#sleep 1; cma_once 0 'DartWalker2d-v1' 32 2048 675000 True 20000 "mean_param" 450; sleep 1; ps
#sleep 1; cma_once 0 'DartWalker2d-v1' 32 2048 675000 True 20000 "mean_param" 400; sleep 1; ps
#sleep 1; cma_once 0 'DartWalker2d-v1' 32 2048 675000 False 0 "mean_param" $n_components; sleep 1; ps
#sleep 1; first_comp_angle_with_diff 0 'DartWalker2d-v1' 32 2048 675000 True 10000 $n_components $n_components; sleep 1; ps
#
#
#sleep 1; how_many_steps_can_you_go 0 'DartWalker2d-v1' 32 2048 675000 True 1000 100 $n_components; sleep 1; ps
#sleep 1; how_many_steps_can_you_go 0 'DartWalker2d-v1' 32 2048 675000 True 3000 100 $n_components; sleep 1; ps
#sleep 1; final_projection_on_mean_performance 0 'DartWalker2d-v1' 32 2048 675000 True 10000 $n_components; sleep 1; ps
#sleep 1; how_many_steps_can_you_go 0 'DartWalker2d-v1' 32 2048 1000000 True 3000 100 $n_components; sleep 1; ps

#sleep 1; cma_once 0 'DartWalker2d-v1' 512 2048 675000 True 50000 "mean_param" 50; sleep 1; ps

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

