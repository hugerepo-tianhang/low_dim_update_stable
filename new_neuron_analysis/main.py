import sys
import os
d = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..'))
od = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
sys.path.append(d)
sys.path.append(od)

from new_neuron_analysis.run_trained_policy import eval_trained_policy_and_collect_data
from new_neuron_analysis.analyse_data import crunch_and_plot_data
from new_neuron_analysis.experiment_augment_input import run_experiment
from stable_baselines.ppo2.run_mujoco import train
from new_neuron_analysis.plot_result import plot, get_results


def complete_run(policy_num_timesteps, policy_run_num, policy_seed, eval_seed,
                 eval_run_num, augment_env, augment_num_timesteps, top_num_to_include, augment_seed,
                 policy_env, augment_run_num, network_size):



    experiment_label, log_dir = run_experiment(augment_env, augment_num_timesteps, top_num_to_include, augment_seed,
                                               augment_run_num, network_size,
                                               policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed,
                                               eval_run_num)
    return [experiment_label, log_dir]

def main():
    from joblib import Parallel, delayed

    seeds = [0]
    run_nums = [0]
    policy_num_timesteps = 10000
    policy_env = "DartWalker2d-v1"
    augment_env = 'DartWalker2d_aug_input_current_trial-v1'

    # num_timesteps = 1000000
    # total_num_to_includes = [10, 30, 60]
    # trained_policy_seeds = [0,1,2]
    # trained_policy_run_nums = [0,1,2]
    # network_sizes = [16, 32, 64, 128]

    augment_num_timesteps = 5000
    top_num_to_includes = [10, 20]
    network_sizes = [16]

    for policy_seed in seeds:
        for policy_run_num in run_nums:
            cmd_line = ["--num-timesteps", str(policy_num_timesteps), "--run_num", str(policy_run_num), "--seed",
                        str(policy_seed)]

            train(cmd_line)

            for eval_seed in seeds:
                for eval_run_num in run_nums:
                    eval_trained_policy_and_collect_data(seed=eval_seed, run_num=eval_run_num, policy_env=policy_env,
                                                         policy_num_timesteps=policy_num_timesteps,
                                                         policy_run_num=policy_run_num, policy_seed=policy_seed)

                    crunch_and_plot_data(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed,
                                         eval_run_num)

                    results = Parallel(n_jobs=-1)(delayed(run_experiment)(augment_env, augment_num_timesteps, top_num_to_include, augment_seed,
                                               augment_run_num, network_size,
                                               policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed,
                                               eval_run_num)

                            for augment_seed in seeds
                            for augment_run_num in run_nums
                            for top_num_to_include in top_num_to_includes
                            for network_size in network_sizes)

                    labels, log_dirs = zip(*results)
                    name = f"policy_seed_{policy_seed}_policy_run_num_{policy_run_num}_eval_seed_{eval_seed}_eval_run_num{eval_run_num}"
                    plot(labels, log_dirs, name)




main()