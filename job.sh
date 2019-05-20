#!/bin/bash


cores_to_use=-1
xnum=50
ynum=50

nminibatches=32
n_steps=2048

padding_fraction=0.4
n_components=500

n_comp_to_use=2
cma_num_timesteps=500000
ppos_num_timesteps=2000000
eval_num_timesteps=1024
even_check_point_num=5
normalize=True
optimizer=adam

use_IPCA=True
chunk_size=10000

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
plot_other_plane_return_landscape () {
    local run=$1
    local env=$2
    local time_steps=$3

    local other_pca_index=$4
    local origin=$5


    echo "Welcome to plot_other_plane_return_landscape: run number  $env $run"

    python -m stable_baselines.low_dim_analysis.plot_other_plane_return_landscape \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
                                    --nminibatches=$nminibatches --n_steps=$n_steps\
                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps\
                                    --n_components=$n_components --normalize=$normalize\
                                    --other_pca_index=$other_pca_index --origin=$origin

}
run () {
    local run=$1
    local env=$2
    local nminibatches=$3
    local n_steps=$4
    local time_steps=$5

    local optimizer=$6
    local use_run_num_start=$7

    echo "Welcome to RUN: run number  $env $run"
    python -m stable_baselines.ppo2.run_mujoco --env=$env --num-timesteps=$time_steps\
            --run_num=$run --normalize=$normalize --nminibatches=$nminibatches\
            --n_steps=$n_steps --optimizer=$optimizer --use_run_num_start=$use_run_num_start

}

are_final_parameters_the_same () {
    local env=$1
    local nminibatches=$2
    local n_steps=$3
    local time_steps=$4

    local optimizer=$5
    local run_nums_to_check=$6

    echo "Welcome to are_final_parameters_the_same: run number  $env $run"
    python -m stable_baselines.low_dim_analysis.are_final_parameters_the_same --env=$env --num-timesteps=$time_steps\
            --normalize=$normalize --nminibatches=$nminibatches\
            --n_steps=$n_steps --optimizer=$optimizer --run_nums_to_check=$run_nums_to_check

}

cma_redo () {
    local run=$1
    local env=$2

    local time_steps=$3
    local use_IPCA=$4
    local chunk_size=$5
    local origin=$6
    local cma_var=$7

    echo "Welcome to cma_redo: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.cmaes.cma_redo \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
                                    --cores_to_use=$cores_to_use \
                                    --xnum=$xnum --ynum=$ynum\
                                    --padding_fraction=$padding_fraction \
                                     --cma_num_timesteps=$cma_num_timesteps\
                                     --eval_num_timesteps=$eval_num_timesteps\
                                     --normalize=$normalize --nminibatches=$nminibatches\
                                     --n_steps=$n_steps --use_IPCA=$use_IPCA --chunk_size=$chunk_size\
                                     --origin=$origin
}


cma_once () {
    local run=$1
    local env=$2

    local time_steps=$3
    local use_IPCA=$4
    local chunk_size=$5
    local origin=$6
    local n_comp_to_use=$7
    local cma_var=$8

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


cma_and_then_ppo2 () {
    local run=$1
    local env=$2

    local time_steps=$3
    local use_IPCA=$4
    local chunk_size=$5
    local origin=$6
    local other_pca_index=$7
    local cma_var=$8
    local ppo_num_timesteps=$9


    echo "Welcome to cma_and_then_ppo2: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.cmaes.cma_and_then_ppo2 \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
                                    --cores_to_use=$cores_to_use \
                                    --xnum=$xnum --ynum=$ynum\
                                    --padding_fraction=$padding_fraction \
                                    --n_components=$n_components --cma_num_timesteps=$cma_num_timesteps\
                                    --other_pca_index=$other_pca_index --eval_num_timesteps=$eval_num_timesteps\
                                     --normalize=$normalize --nminibatches=$nminibatches\
                                     --n_steps=$n_steps --use_IPCA=$use_IPCA --chunk_size=$chunk_size\
                                     --origin=$origin --ppo_num_timesteps=$ppo_num_timesteps
}

ppos_once () {
    local run=$1
    local env=$2
    local time_steps=$3

    local use_IPCA=$4
    local chunk_size=$5
    local origin=$6
    local n_comp_to_use=$7
    echo "Welcome to ppos: run number  $env $run"

    python -m stable_baselines.ppo_subspace.ppo_sub \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
                                    --cores_to_use=$cores_to_use \
                                    --xnum=$xnum --ynum=$ynum\
                                    --padding_fraction=$padding_fraction \
                                    --n_components=$n_components --ppos_num_timesteps=$ppos_num_timesteps\
                                    --n_comp_to_use=$n_comp_to_use\
                                     --normalize=$normalize --nminibatches=$nminibatches\
                                     --n_steps=$n_steps --use_IPCA=$use_IPCA --chunk_size=$chunk_size\
                                     --origin=$origin
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


first_n_pc1_vs_final_minus_start () {
    local run=$1
    local env=$2
    local nminibatches=$3
    local n_steps=$4
    local time_steps=$5

    local pc1_chunk_size=$6


    echo "Welcome to first_n_pc1_vs_final_minus_start: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.low_dim_analysis.first_n_pc1_vs_final_minus_start \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env --normalize=$normalize \
                                    --nminibatches=$nminibatches --n_steps=$n_steps\
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

    echo "Welcome to so_far_pcn_vs_final_minus_current: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.low_dim_analysis.so_far_pcn_vs_final_minus_current \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env --normalize=$normalize \
                                    --nminibatches=$nminibatches --n_steps=$n_steps\
                                    --pc1_chunk_size=$pc1_chunk_size
}


skip_first_n_chunks () {
    local run=$1
    local env=$2
    local nminibatches=$3
    local n_steps=$4
    local time_steps=$5

    local pc1_chunk_size=$6
    local skipped_chunks=$7

    echo "Welcome to skip_first_n_chunks: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.low_dim_analysis.skip_first_n_chunks \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env --normalize=$normalize \
                                    --nminibatches=$nminibatches --n_steps=$n_steps\
                                    --pc1_chunk_size=$pc1_chunk_size --skipped_chunks=$skipped_chunks
}

grad_vs_V () {
    local run=$1
    local env=$2
    local nminibatches=$3
    local n_steps=$4
    local time_steps=$5

    local pc1_chunk_size=$6
    local optimizer=$7

    echo "Welcome to grad_vs_V: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.low_dim_analysis.grad_vs_V \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env --normalize=$normalize \
                                    --nminibatches=$nminibatches --n_steps=$n_steps\
                                    --pc1_chunk_size=$pc1_chunk_size --optimizer=$optimizer
}
update_dir_vs_final_min_start () {
    local run=$1
    local env=$2
    local nminibatches=$3
    local n_steps=$4
    local time_steps=$5

    local pc1_chunk_size=$6
    local optimizer=$7

    echo "Welcome to update_dir_vs_final_min_start: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.low_dim_analysis.update_dir_vs_final_min_start \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env --normalize=$normalize \
                                    --nminibatches=$nminibatches --n_steps=$n_steps\
                                    --pc1_chunk_size=$pc1_chunk_size --optimizer=$optimizer
}


dup_last_part_to_approx_pc1 () {
    local run=$1
    local env=$2
    local nminibatches=$3
    local n_steps=$4
    local time_steps=$5

    local pc1_chunk_size=$6
    local chunk_size=$7
    local optimizer=$8

    echo "Welcome to dup_last_part_to_approx_pc1: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.low_dim_analysis.dup_last_part_to_approx_pc1 \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env --normalize=$normalize \
                                    --nminibatches=$nminibatches --n_steps=$n_steps\
                                    --pc1_chunk_size=$pc1_chunk_size\
                                    --chunk_size=$chunk_size --optimizer=$optimizer
}


pc1_vs_V () {
    local run=$1
    local env=$2
    local nminibatches=$3
    local n_steps=$4
    local time_steps=$5

    local num_comp_to_load=$6


    echo "Welcome to pc1_vs_V: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.low_dim_analysis.pc1_vs_V \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env --normalize=$normalize \
                                    --nminibatches=$nminibatches --n_steps=$n_steps\
                                    --num_comp_to_load=$num_comp_to_load
}



first_n_2d_plane_angle_vs_final_2d_plane () {
    local run=$1
    local env=$2
    local time_steps=$3

    local pc1_chunk_size=$4
    local chunk_size=$5
    local n_comp_to_use=$6
    local n_components=$7

    echo "Welcome to first_n_2d_plane_angle_vs_final_2d_plane: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.low_dim_analysis.first_n_2d_plane_angle_vs_final_2d_plane \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env --normalize=$normalize \
                                    --nminibatches=$nminibatches --n_steps=$n_steps\
                                    --pc1_chunk_size=$pc1_chunk_size\
                                    --chunk_size=$chunk_size --optimizer=$optimizer\
                                    --n_comp_to_use=$n_comp_to_use --n_components=$n_components
}


skip_m_chunks_first_n_plane_vs_final_plane_angle () {
    local run=$1
    local env=$2
    local time_steps=$3

    local pc1_chunk_size=$4
    local chunk_size=$5
    local n_comp_to_use=$6
    local n_components=$7
    local skipped_chunks=$8

    echo "Welcome to skip_m_chunks_first_n_plane_vs_final_plane_angle: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.low_dim_analysis.skip_m_chunks_first_n_plane_vs_final_plane_angle \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env --normalize=$normalize \
                                    --nminibatches=$nminibatches --n_steps=$n_steps\
                                    --pc1_chunk_size=$pc1_chunk_size\
                                    --chunk_size=$chunk_size --optimizer=$optimizer\
                                    --n_comp_to_use=$n_comp_to_use --n_components=$n_components\
                                    --skipped_chunks=$skipped_chunks
}


WPCA_first_n_VS_last_plane () {
    local run=$1
    local env=$2
    local time_steps=$3

    local pc1_chunk_size=$4
    local chunk_size=$5
    local n_comp_to_use=$6
    local n_components=$7
    local func_index_to_use=$8

    echo "Welcome to WPCA_first_n_VS_last_plane: run number  $env $run"

#    python -m stable_baselines.low_dim_analysis.plot_return_landscape \
#                                    --num-timesteps=$time_steps --run_num=$run --env=$env\
#                                    --cores_to_use=$cores_to_use --xnum=$xnum --ynum=$ynum\
#                                    --padding_fraction=$padding_fraction --eval_num_timesteps=$eval_num_timesteps
    python -m stable_baselines.low_dim_analysis.WPCA_first_n_VS_last_plane \
                                    --num-timesteps=$time_steps --run_num=$run --env=$env --normalize=$normalize \
                                    --nminibatches=$nminibatches --n_steps=$n_steps\
                                    --pc1_chunk_size=$pc1_chunk_size\
                                    --chunk_size=$chunk_size --optimizer=$optimizer\
                                    --n_comp_to_use=$n_comp_to_use --n_components=$n_components\
                                    --func_index_to_use=$func_index_to_use
}


#sleep 1; ppos_once 0 'Walker2d-v2' 8 2048; sleep 1; ps
#
#sleep 1; ppos_once 0 'Hopper-v2' 8 2048; sleep 1; ps
#sleep 1; ppos_once 0 'Walker2d-v2' 8 2048; sleep 1; ps

#
#sleep 1; run 0 'DartHopper-v1' 512 2048 1000000& sleep 1; ps
#sleep 1; run 0 'DartHopper-v1' 2 2048& sleep 1; ps
#sleep 1; run 0 'DartHopper-v1' 32 2048 1000000& sleep 1; ps
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
#sleep 1; run_sgd 0 'DartWalker2d-v1' 32 2048 5000& sleep 1; ps
#sleep 1; run 0 'DartWalker2d-v1' 32 2048 1000000 'sgd'& sleep 1; ps
#sleep 1; run 1 'DartWalker2d-v1' 32 2048 1000000 'sgd'& sleep 1; ps
#sleep 1; run 1 'DartWalker2d-v1' 32 2048 675000 'adam'& sleep 1; ps
#sleep 1; run 10 'DartWalker2d-v1' 32 2048 675000 'adam' -1& sleep 1; ps
#sleep 1; run 11 'DartWalker2d-v1' 32 2048 675000 'adam' 10& sleep 1; ps
#sleep 1; run 12 'DartWalker2d-v1' 32 2048 675000 'adam' 10& sleep 1; ps
#sleep 1; run 13 'DartWalker2d-v1' 32 2048 675000 'adam' 10& sleep 1; ps
#sleep 1; run 3 'DartWalker2d-v1' 32 2048 675000 'adam'& sleep 1; ps
sleep 1; run 1 'DartWalker2d-v1' 64 4096 3000000 'adam' -1& sleep 1; ps
sleep 1; run 2 'DartWalker2d-v1' 64 4096 3000000 'adam' -1& sleep 1; ps
sleep 1; run 1 'DartHopper-v1' 64 4096 3000000 'adam' -1& sleep 1; ps
sleep 1; run 2 'DartHopper-v1' 64 4096 3000000 'adam' -1& sleep 1; ps

#
#sleep 1; run 0 'DartReacher-v1' 32 2048 675000 'adam'& sleep 1; ps
#sleep 1; run 1 'DartReacher-v1' 32 2048 675000 'adam'& sleep 1; ps
#sleep 1; run 0 'DartHalfCheetah-v1' 32 2048 675000 'adam'& sleep 1; ps
#sleep 1; run 1 'DartHalfCheetah-v1' 32 2048 675000 'adam'& sleep 1; ps
#sleep 1; run 0 'DartSnake7Link-v1' 32 2048 675000 'adam'& sleep 1; ps
#sleep 1; run 1 'DartSnake7Link-v1' 32 2048 675000 'adam'& sleep 1; ps


wait

#sleep 1; are_final_parameters_the_same 'DartWalker2d-v1' 32 2048 675000 'adam' 0:1:2:3:10:11:12:13& sleep 1; ps


#sleep 1; ppos_once 0 'DartWalker2d-v1' 675000 True 10000 "final_param" 2; sleep 1; ps
#sleep 1; ppos_once 0 'DartWalker2d-v1' 675000 True 10000 "mean_param" 2; sleep 1; ps
#sleep 1; ppos_once 1 'DartWalker2d-v1' 675000 True 10000 "final_param" 2; sleep 1; ps
#sleep 1; ppos_once 1 'DartWalker2d-v1' 675000 True 10000 "mean_param" 2; sleep 1; ps


#sleep 1; ppos_once 0 'Hopper-v2' 8 2048; sleep 1; ps
#sleep 1; cma_once 0 'DartHopper-v1' 32 2048 1000000 True 50000 "mean_param" 300; sleep 1; ps
#sleep 1; final_projection_on_mean_performance 0 'DartHopper-v1' 32 2048 1000000 True 50000 $n_components; sleep 1; ps
#sleep 1; first_n_2d_plane_angle_vs_final_2d_plane 0 'DartWalker2d-v1' 675000 5000 20000 100 500; sleep 1; ps
#sleep 1; first_n_2d_plane_angle_vs_final_2d_plane 0 'DartWalker2d-v1' 675000 5000 20000 500 500; sleep 1; ps
#sleep 1; skip_m_chunks_first_n_plane_vs_final_plane_angle 0 'DartWalker2d-v1' 675000 1000 20000 1 500 1; sleep 1; ps
#sleep 1; skip_m_chunks_first_n_plane_vs_final_plane_angle 0 'DartWalker2d-v1' 675000 1000 20000 1 500 10; sleep 1; ps
#sleep 1; skip_m_chunks_first_n_plane_vs_final_plane_angle 0 'DartWalker2d-v1' 675000 1000 20000 1 500 50; sleep 1; ps
#sleep 1; first_n_2d_plane_angle_vs_final_2d_plane 0 'DartWalker2d-v1' 675000 5000 20000 1 500; sleep 1; ps

#sleep 1; first_n_2d_plane_angle_vs_final_2d_plane 0 'DartWalker2d-v1' 5000 5000 20000 500 10; sleep 1; ps
#sleep 1; first_n_2d_plane_angle_vs_final_2d_plane 0 'DartWalker2d-v1' 5000 5000 20000 500 500; sleep 1; ps

#sleep 1; first_n_pc1_vs_final_minus_start 1 'DartWalker2d-v1' 32 2048 675000 100; sleep 1; ps
#sleep 1; first_n_pc1_vs_final_minus_start 0 'DartHalfCheetah-v1' 32 2048 675000 100; sleep 1; ps
#sleep 1; first_n_pc1_vs_final_minus_start 1 'DartHalfCheetah-v1' 32 2048 675000 100; sleep 1; ps
#sleep 1; pc1_vs_V 0 'DartHopper-v1' 32 2048 1000000 150; sleep 1; ps

#sleep 1; plot_other_plane_return_landscape 0 'DartWalker2d-v1' 675000 0:1 "start_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartWalker2d-v1' 675000 2:3 "start_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartWalker2d-v1' 675000 4:5 "start_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartWalker2d-v1' 675000 200:201 "start_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartWalker2d-v1' 675000 498:499 "start_param"; sleep 1; ps
#
#sleep 1; plot_other_plane_return_landscape 0 'DartWalker2d-v1' 675000 0:1 "mean_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartWalker2d-v1' 675000 2:3 "mean_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartWalker2d-v1' 675000 4:5 "mean_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartWalker2d-v1' 675000 200:201 "mean_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartWalker2d-v1' 675000 498:499 "mean_param"; sleep 1; ps
#
#sleep 1; plot_other_plane_return_landscape 1 'DartWalker2d-v1' 675000 0:1 "start_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 1 'DartWalker2d-v1' 675000 2:3 "start_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 1 'DartWalker2d-v1' 675000 4:5 "start_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 1 'DartWalker2d-v1' 675000 200:201 "start_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 1 'DartWalker2d-v1' 675000 498:499 "start_param"; sleep 1; ps
#
#sleep 1; plot_other_plane_return_landscape 1 'DartWalker2d-v1' 675000 0:1 "mean_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 1 'DartWalker2d-v1' 675000 2:3 "mean_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 1 'DartWalker2d-v1' 675000 4:5 "mean_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 1 'DartWalker2d-v1' 675000 200:201 "mean_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 1 'DartWalker2d-v1' 675000 498:499 "mean_param"; sleep 1; ps
##
#
#sleep 1; plot_other_plane_return_landscape 0 'DartHalfCheetah-v1' 675000 0:1 "start_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartHalfCheetah-v1' 675000 2:3 "start_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartHalfCheetah-v1' 675000 4:5 "start_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartHalfCheetah-v1' 675000 200:201 "start_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartHalfCheetah-v1' 675000 498:499 "start_param"; sleep 1; ps
#
#sleep 1; plot_other_plane_return_landscape 0 'DartHalfCheetah-v1' 675000 0:1 "mean_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartHalfCheetah-v1' 675000 2:3 "mean_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartHalfCheetah-v1' 675000 4:5 "mean_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartHalfCheetah-v1' 675000 200:201 "mean_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartHalfCheetah-v1' 675000 498:499 "mean_param"; sleep 1; ps
##
#sleep 1; plot_other_plane_return_landscape 0 'DartReacher-v1' 675000 0:1 "start_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartReacher-v1' 675000 2:3 "start_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartReacher-v1' 675000 4:5 "start_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartReacher-v1' 675000 200:201 "start_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartReacher-v1' 675000 498:499 "start_param"; sleep 1; ps
#
#sleep 1; plot_other_plane_return_landscape 0 'DartReacher-v1' 675000 0:1 "mean_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartReacher-v1' 675000 2:3 "mean_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartReacher-v1' 675000 4:5 "mean_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartReacher-v1' 675000 200:201 "mean_param"; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartReacher-v1' 675000 498:499 "mean_param"; sleep 1; ps
#
##
#
#sleep 1; plot_other_plane_return_landscape 0 'DartHalfCheetah-v1' 675000 0:1:2; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 1 'DartHalfCheetah-v1' 675000 0:1:2; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartHalfCheetah-v1' 675000 3:4:5; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 1 'DartHalfCheetah-v1' 675000 3:4:5; sleep 1; ps
#
#
#sleep 1; plot_other_plane_return_landscape 0 'DartReacher-v1' 675000 0:1:2; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 1 'DartReacher-v1' 675000 0:1:2; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartReacher-v1' 675000 3:4:5; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 1 'DartReacher-v1' 675000 3:4:5; sleep 1; ps
#
#
#sleep 1; plot_other_plane_return_landscape 0 'DartSnake7Link-v1' 675000 0:1:2; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 1 'DartSnake7Link-v1' 675000 0:1:2; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 0 'DartSnake7Link-v1' 675000 3:4:5; sleep 1; ps
#sleep 1; plot_other_plane_return_landscape 1 'DartSnake7Link-v1' 675000 3:4:5; sleep 1; ps


#sleep 1; cma_once 0 'DartHopper-v1' 512 2048 1000000 True 50000; sleep 1; ps
##sleep 1; cma_once 0 'DartHopper-v1' 32 2048 1000000 True 10000 "mean_param" $n_components; sleep 1; ps
##sleep 1; cma_once 0 'DartHopper-v1' 32 2048 1000000 True 10000 "mean_param" 15; sleep 1; ps
#sleep 1; cma_redo 0 'DartWalker2d-v1' 675000 True 5000 "mean_param" 10; sleep 1; ps
#sleep 1; cma_redo 0 'DartWalker2d-v1' 675000 True 5000 "mean_param" 10; sleep 1; ps
#sleep 1; cma_and_then_ppo2 0 'DartWalker2d-v1' 675000 True 5000 "start_param" 1 10 600000; sleep 1; ps
#sleep 1; cma_and_then_ppo2 0 'DartWalker2d-v1' 675000 True 5000 "start_param" 1 10 600000; sleep 1; ps

#sleep 1; cma_and_then_ppo2 0 'DartWalker2d-v1' 675000 True 20000 "start_param" 10 10 600000; sleep 1; ps
#sleep 1; cma_and_then_ppo2 3 'DartWalker2d-v1' 675000 True 5000 "mean_param" 0:1 10 200000; sleep 1; ps
#sleep 1; cma_and_then_ppo2 3 'DartWalker2d-v1' 675000 True 5000 "mean_param" 3:4 10 200000; sleep 1; ps
#sleep 1; cma_and_then_ppo2 3 'DartWalker2d-v1' 675000 True 5000 "mean_param" 480:481 10 200000; sleep 1; ps
#
#sleep 1; cma_and_then_ppo2 3 'DartWalker2d-v1' 675000 True 5000 "start_param" 0:1 10 200000; sleep 1; ps
#sleep 1; cma_and_then_ppo2 3 'DartWalker2d-v1' 675000 True 5000 "start_param" 3:4 10 200000; sleep 1; ps
#sleep 1; cma_and_then_ppo2 3 'DartWalker2d-v1' 675000 True 5000 "start_param" 480:481 10 200000; sleep 1; ps
#
#sleep 1; cma_and_then_ppo2 0 'DartWalker2d-v1' 675000 True 5000 "mean_param" 0:1 10 200000; sleep 1; ps
#sleep 1; cma_and_then_ppo2 0 'DartWalker2d-v1' 675000 True 5000 "mean_param" 3:4 10 200000; sleep 1; ps
#sleep 1; cma_and_then_ppo2 0 'DartWalker2d-v1' 675000 True 5000 "mean_param" 480:481 10 200000; sleep 1; ps
#
#sleep 1; cma_and_then_ppo2 0 'DartWalker2d-v1' 675000 True 5000 "start_param" 0:1 10 200000; sleep 1; ps
#sleep 1; cma_and_then_ppo2 0 'DartWalker2d-v1' 675000 True 5000 "start_param" 3:4 10 200000; sleep 1; ps
#sleep 1; cma_and_then_ppo2 0 'DartWalker2d-v1' 675000 True 5000 "start_param" 480:481 10 200000; sleep 1; ps
#sleep 1; cma_once 0 'DartWalker2d-v1' 675000 True 5000 "mean_param" 100 100; sleep 1; ps
#sleep 1; cma_once 0 'DartWalker2d-v1' 675000 True 5000 "mean_param" 2 10; sleep 1; ps
#sleep 1; cma_once 0 'DartWalker2d-v1' 675000 True 5000 "mean_param" 2 100; sleep 1; ps
#sleep 1; cma_once 0 'DartWalker2d-v1' 675000 True 5000 "mean_param" 2 5; sleep 1; ps

##sleep 1; ppos_once 0 'Walker2d-v2' 8 2048; sleep 1; ps
#sleep 1; weighted_pcn_vs_final 0 'DartWalker2d-v1' 32 2048 675000 1000 100; sleep 1;

#sleep 1; so_far_pcn_vs_final_minus_current 0 'DartWalker2d-v1' 32 2048 675000 1000; sleep 1;
#sleep 1; so_far_pcn_vs_final_minus_current 1 'DartWalker2d-v1' 32 2048 675000 1000; sleep 1;
#sleep 1; so_far_pcn_vs_final_minus_current 2 'DartWalker2d-v1' 32 2048 675000 1000; sleep 1;
#sleep 1; so_far_pcn_vs_final_minus_current 0 'DartWalker2d-v1' 32 2048 675000 5000 50; sleep 1;
#
#sleep 1; so_far_pcn_vs_final_minus_current 0 'DartWalker2d-v1' 32 2048 675000 1000 10; sleep 1;
#sleep 1; so_far_pcn_vs_final_minus_current 0 'DartWalker2d-v1' 32 2048 675000 3000 10; sleep 1;
#sleep 1; so_far_pcn_vs_final_minus_current 0 'DartWalker2d-v1' 32 2048 675000 5000 10; sleep 1;

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

#sleep 1; dup_last_part_to_approx_pc1 0 'DartWalker2d-v1' 32 2048 1000000 3000 20000 'sgd'; sleep 1; ps
##sleep 1; dup_last_part_to_approx_pc1 1 'DartWalker2d-v1' 32 2048 1000000 100 10000 'sgd'; sleep 1; ps
#sleep 1; dup_last_part_to_approx_pc1 0 'DartWalker2d-v1' 32 2048 675000 1000 20000 'adam'; sleep 1; ps
#sleep 1; WPCA_first_n_VS_last_plane 0 'DartWalker2d-v1' 675000 200 20000 1 500 0; sleep 1; ps

#sleep 1; dup_last_part_to_approx_pc1 1 'DartWalker2d-v1' 32 2048 675000 3000 20000 'adam'; sleep 1; ps
##sleep 1; dup_last_part_to_approx_pc1 2 'DartWalker2d-v1' 32 2048 675000 100 10000 'adam'; sleep 1; ps

#sleep 1; grad_vs_V 0 'DartWalker2d-v1' 32 2048 675000 100 'sgd'; sleep 1; ps

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

