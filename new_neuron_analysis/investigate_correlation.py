from new_neuron_analysis.analyse_data import *

if __name__ =="__main__":

    trained_policy_env = "DartWalker2d-v1"
    trained_policy_num_timesteps = 2000000
    policy_run_nums = [0]
    policy_seeds = [0]
    eval_seed = 3
    eval_run_num = 3
    aug_num_timesteps=1500000
    for policy_run_num in policy_run_nums:
        for policy_seed in policy_seeds:
            result_dir = get_result_dir(trained_policy_env, trained_policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num)

            plot(result_dir)