import sys
import os
d = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..'))
od = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
sys.path.append(d)
sys.path.append(od)

from new_neuron_analysis.run_trained_policy import eval_trained_policy_and_collect_data
from new_neuron_analysis.analyse_data import crunch_and_plot_data
from new_neuron_analysis.experiment_augment_input import run_experiment, get_result_dir
from stable_baselines.ppo2.run_mujoco import train
from new_neuron_analysis.plot_result import plot, get_results
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

def main():
    import multiprocessing as mp

    seeds = [0, 1, 2]
    run_nums = [0, 1, 2]
    policy_num_timesteps = 2000000
    policy_env = "DartWalker2d-v1"

    augment_num_timesteps = 1000000
    top_num_to_includes = [0, 5, 10, 20]
    network_sizes = [16, 32, 64]

    # seeds = [0, 1]
    # run_nums = [0, 1]
    # policy_num_timesteps = 1000
    # policy_env = "DartWalker2d-v1"
    #
    # augment_num_timesteps = 1000
    # top_num_to_includes = [0, 10, 30]
    # network_sizes = [16, 32]



    with mp.Pool(mp.cpu_count()) as pool:

        for policy_seed in [0,1]:
            for policy_run_num in [0,1]:
                cmd_line = ["--num-timesteps", str(policy_num_timesteps), "--run_num", str(policy_run_num), "--seed",
                            str(policy_seed)]

                train(cmd_line)

                for eval_seed in [4]:
                    for eval_run_num in [4]:
                        eval_trained_policy_and_collect_data(seed=eval_seed, run_num=eval_run_num, policy_env=policy_env,
                                                             policy_num_timesteps=policy_num_timesteps,
                                                             policy_run_num=policy_run_num, policy_seed=policy_seed)

                        crunch_and_plot_data(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed,
                                             eval_run_num)

                        args = [(augment_num_timesteps, top_num_to_include, augment_seed,
                                                augment_run_num, network_size,
                                                policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed,
                                                eval_run_num, learning_rate)

                                                for augment_seed in seeds
                                                for augment_run_num in run_nums
                                                for top_num_to_include in top_num_to_includes
                                                for network_size in network_sizes
                                                for learning_rate in
                                                [64 / network_size * 3e-4, (64 / network_size + 64 / network_size) * 3e-4]]
                        results = pool.starmap(run_experiment, args)

                        # results = [pool.apply(howmany_within_range, args=(row, 4, 8)) for row in data]

                        # result_dir = get_result_dir(policy_env, policy_num_timesteps, policy_run_num,
                        #                             policy_seed, eval_seed, eval_run_num)
                        # plot(result_dir)






main()