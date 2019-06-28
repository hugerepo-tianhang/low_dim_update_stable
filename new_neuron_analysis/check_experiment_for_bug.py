import sys
import os
d = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..'))
od = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
sys.path.append(d)
sys.path.append(od)

from new_neuron_analysis.run_trained_policy import eval_trained_policy_and_collect_data
from new_neuron_analysis.analyse_data import crunch_and_plot_data
from new_neuron_analysis.experiment_augment_input import run_experiment, get_result_dir, run_check_experiment
from stable_baselines.ppo2.run_mujoco import train
from new_neuron_analysis.plot_result import plot, get_results
from new_neuron_analysis.dir_tree_util import *
import warnings
warnings.filterwarnings("ignore")

def complete_run(policy_num_timesteps, policy_run_num, policy_seed, eval_seed,
                 eval_run_num, augment_env, augment_num_timesteps, top_num_to_include, augment_seed,
                 policy_env, augment_run_num, network_size):



    experiment_label, log_dir = run_experiment(augment_env, augment_num_timesteps, top_num_to_include, augment_seed,
                                               augment_run_num, network_size,
                                               policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed,
                                               eval_run_num)
    return [experiment_label, log_dir]

def crunch_correlation_data(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed,
                         eval_run_num):
    eval_trained_policy_and_collect_data(seed=eval_seed, run_num=eval_run_num, policy_env=policy_env,
                                         policy_num_timesteps=policy_num_timesteps,
                                         policy_run_num=policy_run_num, policy_seed=policy_seed)

    crunch_and_plot_data(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed,
                         eval_run_num)


def main():
    import multiprocessing as mp

    policy_num_timesteps = 5000000
    policy_env = "DartWalker2d-v1"
    policy_seeds = [3, 4, 5]
    policy_run_nums = [0, 1]

    eval_seeds = [4]
    eval_run_nums = [4]

    augment_seeds = [0,1]
    augment_run_nums = range(20)
    augment_num_timesteps = 1000000
    top_num_to_includes = [0]
    network_sizes = [64, 16]



    # policy_seeds = [3, 4]
    # policy_run_nums = [0]
    # policy_num_timesteps = 5000
    # policy_env = "DartWalker2d-v1"
    #
    # eval_seeds = [4]
    # eval_run_nums = [4]
    #
    # augment_seeds = [0]
    # augment_run_nums = range(2)
    # augment_num_timesteps = 5000
    # top_num_to_includes = [0]
    # network_sizes = [16]
    #



    with mp.Pool(mp.cpu_count()) as pool:

        run_test_args = [(augment_num_timesteps, augment_seed, augment_run_num, network_size,
                         policy_env, learning_rate)

                         for augment_seed in augment_seeds
                         for augment_run_num in augment_run_nums
                         for network_size in network_sizes
                         for learning_rate in
                         [64 / network_size * 3e-4,
                          (64 / network_size + 64 / network_size) * 3e-4]]
        pool.starmap(run_check_experiment, run_test_args)

        for policy_seed in policy_seeds:
            for policy_run_num in policy_run_nums:
                for eval_seed in eval_seeds:
                    for eval_run_num in eval_run_nums:

                        run_experiment_args = [(augment_num_timesteps, top_num_to_include, augment_seed,
                                augment_run_num, network_size,
                                policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed,
                                eval_run_num, learning_rate, True)


                                for augment_seed in augment_seeds
                                for augment_run_num in augment_run_nums
                                for top_num_to_include in top_num_to_includes
                                for network_size in network_sizes
                                for learning_rate in
                                [64 / network_size * 3e-4, (64 / network_size + 64 / network_size) * 3e-4]]


                        pool.starmap(run_experiment, run_experiment_args)

        # results = [pool.apply(howmany_within_range, args=(row, 4, 8)) for row in data]

        # result_dir = get_result_dir(policy_env, policy_num_timesteps, policy_run_num,
        #                             policy_seed, eval_seed, eval_run_num)
        # plot(result_dir)






main()