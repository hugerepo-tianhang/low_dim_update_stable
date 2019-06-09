from stable_baselines.results_plotter import plot_results, X_TIMESTEPS, plt
from new_neuron_analysis.experiment_augment_input import get_experiment_path_for_this_run, \
     get_log_dir, get_result_dir, AttributeDict, os, get_proj_dir

env = 'DartWalker2d_aug_input_current_trial-v1'
trained_policy_env = "DartWalker2d-v1"
trained_policy_num_timesteps = 10000
policy_run_num = 0
policy_seed = 0
run_seed = 0
run_run_num = 0
# num_timesteps = 5000
# top_num_to_includes = [10]
# trained_policy_seeds = [0, 1]
# trained_policy_run_nums = [0, 1]
# network_sizes = [16]


num_timesteps = 5000
top_num_to_includes = [10]
trained_policy_seeds = [0]
trained_policy_run_nums = [0]
network_sizes = [16]

def get_results(trained_policy_seeds, trained_policy_run_nums, top_num_to_includes, network_sizes):
    labels, total_log_dirs = [], []
    for trained_policy_run_num in trained_policy_run_nums:
        for trained_policy_seed in trained_policy_seeds:
            for top_num_to_include in top_num_to_includes:
                for network_size in network_sizes:
                    experiment_label, log_dir = get_result(trained_policy_seed, trained_policy_run_num,
                                                           top_num_to_include, network_size)
                    labels.append(experiment_label)

                    total_log_dirs.append(log_dir)
    return labels, total_log_dirs

def get_result(trained_policy_seed, trained_policy_run_num, top_num_to_include, network_size):

    args = AttributeDict()
    seed = 0
    run_num = 0 #HARD CODE!
    args.normalize = True
    args.env = env
    args.num_timesteps = num_timesteps
    args.run_num = run_num
    args.alg = "ppo2"
    args.seed = seed


    result_dir = get_result_dir(trained_policy_env, trained_policy_num_timesteps, policy_run_num, policy_seed, run_seed, run_run_num)
    this_run_dir = get_experiment_path_for_this_run(args, top_num_to_include=top_num_to_include,
                                                    result_dir=result_dir, network_size=network_size)
    log_dir = get_log_dir(this_run_dir)

    experiment_label = f"trained_policy_seed{trained_policy_seed}, " \
                       f"trained_policy_run_num{trained_policy_run_num},top_num_to_include{top_num_to_include}, " \
                       f"network_size{network_size}"

    return experiment_label, log_dir




def plot(labels, total_log_dirs, name):
    task_name = "augmented_input"

    fig, figlegend = plot_results(dirs=total_log_dirs, num_timesteps=num_timesteps, xaxis=X_TIMESTEPS, task_name=task_name, labels=labels)
    # save_dir = os.path.abspath(os.path.join(total_log_dirs[0], '..', '..'))
    outer_result_dir = f"{get_proj_dir()}/result/"
    fig.savefig(f"{outer_result_dir}/result_{name}.png")
    figlegend.savefig(f"{outer_result_dir}/result_{name}_legend.png")

if __name__ =="__main__":


    labels, total_log_dirs = get_results(trained_policy_seeds, trained_policy_run_nums, top_num_to_includes,
                                         network_sizes)

    plot(labels, total_log_dirs)