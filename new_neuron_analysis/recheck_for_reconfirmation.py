import sys
import os
d = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..'))
od = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
sys.path.append(d)
sys.path.append(od)

from new_neuron_analysis.run_trained_policy import eval_trained_policy_and_collect_data
from new_neuron_analysis.analyse_data import crunch_and_plot_data
from new_neuron_analysis.experiment_augment_input import run_experiment, _run_experiment, get_result_dir, get_test_dir
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
    policy_seeds = [3]
    policy_run_nums = [1]

    eval_seeds = [4]
    eval_run_nums = [4]

    augment_seeds = range(30)
    augment_run_nums = [0]
    augment_num_timesteps = 1500000
    network_sizes = [64]
    additional_note = "check_these_working_Ms_against_those_other_envs"

    policy_num_timesteps = 9000000
    policy_envs = ["DartWalker2d-v1", "DartSnake7Link-v1", "DartHopper-v1"]
    policy_seeds = [0]
    policy_run_nums = [0]

    eval_seeds = [3]
    eval_run_nums = [3]

    augment_seeds = range(30)
    augment_run_nums = [0]
    augment_num_timesteps = 1500000
    network_sizes = [64]
    additional_note = "reconfirm_check_these_working_Ms_against_those_other_envs"

    # policy_num_timesteps = 5000000
    # policy_env = "DartWalker2d-v1"
    # policy_seeds = [4]
    # policy_run_nums = [0]
    #
    # eval_seeds = [4]
    # eval_run_nums = [4]
    #
    # augment_seeds = range(1)
    # augment_run_nums = [0]
    # augment_num_timesteps = 1500000
    # top_num_to_includes = [slice(0,20)]
    # network_sizes = [64]
    # additional_note = "tee"

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

    test_or_train=False
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


        #============================================================
        checks = [(slice(0,0), []),
                  (slice(0,10), [("M",6), ("M",7), ("M",8), ("M",12), ("M",16), ("M",20), ("M",24), ("M",25), ("M",26), ("COM", 1)]),
                 (slice(0,20), [("M",2),("M",3),("M",6),("M",7),("M",8),("M",11),("M",12),("M",15),("M",16),("M",20),("M",21),("M",22),("M",24),("M",25),("M",26),("M",30),("M",31),("M",40),("M",41), ("COM", 1)])]
        for policy_env in policy_envs:
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

                            for check in checks:
                                top_num_to_include, lagrangian_inds_to_include = check

                                run_experiment_args = [(augment_num_timesteps, top_num_to_include, augment_seed,
                                        augment_run_num, network_size,
                                        policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed,
                                        eval_run_num, learning_rate, additional_note, result_dir,
                                                        lagrangian_inds_to_include)


                                        for augment_seed in augment_seeds
                                        for augment_run_num in augment_run_nums
                                        for network_size in network_sizes
                                        for learning_rate in
                                        [64 / network_size * 3e-4]]

                                pool.starmap(_run_experiment, run_experiment_args)

                                try:
                                    plot(result_dir, augment_num_timesteps)
                                except Exception as e:
                                    print(e)


main()