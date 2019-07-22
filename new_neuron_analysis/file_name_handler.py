import os
from new_neuron_analysis.experiment_augment_input import get_experiment_path_for_this_run, \
     get_log_dir, get_result_dir, AttributeDict, os, get_project_dir, get_save_dir

def change_name(result_dir):
    new_names = []
    for label in os.listdir(result_dir):
        try:
            old_top_num_to_include = label.split("top_num_to_include_")[1].split("_")[0]
        except Exception as e:
            continue
        if ":" not in old_top_num_to_include:
            top_num_to_include = f"0:{old_top_num_to_include}"

            new_name  = label.replace(f"top_num_to_include_{old_top_num_to_include}", f"top_num_to_include_{top_num_to_include}")
            new_names.append((label,new_name))


    for label, new_name in new_names:
        os.rename(src=f"{result_dir}/{label}", dst=f"{result_dir}/{new_name}")

if __name__ == "__main__":
    trained_policy_env = "DartWalker2d-v1"
    trained_policy_num_timesteps = 2000000
    policy_run_nums = [0]
    policy_seeds = [0]
    eval_seed = 3
    eval_run_num = 3
    aug_num_timesteps = 1500000
    additional_note = " (copy)"
    for policy_run_num in policy_run_nums:
        for policy_seed in policy_seeds:
            result_dir = get_result_dir(trained_policy_env, trained_policy_num_timesteps, policy_run_num, policy_seed,
                                        eval_seed, eval_run_num, additional_note=additional_note)

            change_name(result_dir)



