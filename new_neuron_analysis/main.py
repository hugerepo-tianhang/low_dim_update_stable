import sys
import os
d = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..'))
od = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
sys.path.append(d)
sys.path.append(od)

from new_neuron_analysis.run_trained_policy import eval_trained_policy_and_collect_data
from new_neuron_analysis.analyse_data import crunch_and_plot_data
from new_neuron_analysis.experiment_augment_input import run_experiment, get_result_dir, get_test_dir
from stable_baselines.ppo2.run_mujoco import train
from new_neuron_analysis.plot_result import plot, get_results
import warnings
from new_neuron_analysis.dir_tree_util import *
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
                         eval_run_num, additional_note):
    eval_trained_policy_and_collect_data(eval_seed=eval_seed, eval_run_num=eval_run_num, policy_env=policy_env,
                                         policy_num_timesteps=policy_num_timesteps,
                                         policy_run_num=policy_run_num, policy_seed=policy_seed, additional_note=additional_note)

    crunch_and_plot_data(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed,
                         eval_run_num, additional_note=additional_note)


def main():
    import multiprocessing as mp

    policy_num_timesteps = 5000000
    policy_env = "DartWalker2d-v1"
    policy_seeds = [4, 5]
    policy_run_nums = [0, 1]

    eval_seeds = [4]
    eval_run_nums = [4]

    augment_seeds = range(30)
    augment_run_nums = [0]
    augment_num_timesteps = 1500000
    top_num_to_includes = [slice(0,0),slice(0,10),slice(0,20),slice(10,20)]
    network_sizes = [64]
    additional_note = "largebatchlineartestforotherruns"

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
    # top_num_to_includes = [slice(0,10)]
    # network_sizes = [16]
    # additional_note = "largebatchtestforotherruns"

    test=False
    # policy_num_timesteps = 2000000
    # policy_env = "DartWalker2d-v1"
    # policy_seeds = [0]
    # policy_run_nums = [0]
    #
    # eval_seeds = [3]
    # eval_run_nums = [3]
    #
    # augment_seeds = range(1)
    # augment_run_nums = [0]
    # augment_num_timesteps = 5000
    # top_num_to_includes = [slice(0, 10), slice(0,0)]
    # network_sizes = [64]
    # additional_note = "non_linear"



    with mp.Pool(mp.cpu_count()) as pool:

        # # ============================================================
        #
        # train_policy_args = [(["--num-timesteps", str(policy_num_timesteps), "--run_num", str(policy_run_num), "--seed",
        #             str(policy_seed)]) for policy_seed in policy_seeds for policy_run_num in policy_run_nums]
        #
        # pool.map(train, train_policy_args)
        #
        #
        #============================================================
        correlation_data_args = [(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num, additional_note)
                                for policy_seed in policy_seeds
                                for policy_run_num in policy_run_nums
                                for eval_seed in eval_seeds
                                for eval_run_num in eval_run_nums]

        pool.starmap(crunch_correlation_data, correlation_data_args)

        #============================================================

        for policy_seed in policy_seeds:
            for policy_run_num in policy_run_nums:
                for eval_seed in eval_seeds:
                    for eval_run_num in eval_run_nums:

                        # if not test:
                        result_dir = get_result_dir(policy_env, policy_num_timesteps, policy_run_num,
                                                        policy_seed, eval_seed, eval_run_num, additional_note)
                        # else:
                        #     result_dir = get_test_dir(policy_env, policy_num_timesteps, policy_run_num, policy_seed,
                        #                               eval_seed, eval_run_num, augment_seed, additional_note)

                        create_dir_if_not(result_dir)


                        run_experiment_args = [(augment_num_timesteps, top_num_to_include, augment_seed,
                                augment_run_num, network_size,
                                policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed,
                                eval_run_num, learning_rate, additional_note, result_dir, test)


                                for augment_seed in augment_seeds
                                for augment_run_num in augment_run_nums
                                for top_num_to_include in top_num_to_includes
                                for network_size in network_sizes
                                for learning_rate in
                                [64 / network_size * 3e-4, (64 / network_size + 64 / network_size) * 3e-4]]

                        pool.starmap(run_experiment, run_experiment_args)


#################################################################
        #
        # policy_num_timesteps = 2000000
        # policy_env = "DartWalker2d-v1"
        # policy_seeds = [0]
        # policy_run_nums = [0]
        #
        # eval_seeds = [3]
        # eval_run_nums = [3]
        #
        # augment_seeds = range(30)
        # augment_run_nums = [0]
        # augment_num_timesteps = 1500000
        # top_num_to_includes = [slice(0, 30)]
        # network_sizes = [64]
        # additional_note = "testtop30linearshouldoutperformsinceitsadditive"
        # test = False
        #
        # #===================================
        # correlation_data_args = [(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num, additional_note)
        #                         for policy_seed in policy_seeds
        #                         for policy_run_num in policy_run_nums
        #                         for eval_seed in eval_seeds
        #                         for eval_run_num in eval_run_nums]
        #
        # pool.starmap(crunch_correlation_data, correlation_data_args)
        # #=====================================
        #
        # for policy_seed in policy_seeds:
        #     for policy_run_num in policy_run_nums:
        #         for eval_seed in eval_seeds:
        #             for eval_run_num in eval_run_nums:
        #                 # if not test:
        #                 result_dir = get_result_dir(policy_env, policy_num_timesteps, policy_run_num,
        #                                                 policy_seed, eval_seed, eval_run_num, additional_note)
        #                 # else:
        #                 #     result_dir = get_test_dir(policy_env, policy_num_timesteps, policy_run_num, policy_seed,
        #                 #                               eval_seed, eval_run_num, augment_seed, additional_note)
        #
        #                 create_dir_if_not(result_dir)
        #                 run_experiment_args = [(augment_num_timesteps, top_num_to_include, augment_seed,
        #                         augment_run_num, network_size,
        #                         policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed,
        #                         eval_run_num, learning_rate, additional_note, result_dir, test)
        #
        #
        #                         for augment_seed in augment_seeds
        #                         for augment_run_num in augment_run_nums
        #                         for top_num_to_include in top_num_to_includes
        #                         for network_size in network_sizes
        #                         for learning_rate in
        #                         [64 / network_size * 3e-4]]
        #
        #                 pool.starmap(run_experiment, run_experiment_args)
        #
        #
        #




main()