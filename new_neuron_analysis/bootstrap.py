from stable_baselines.results_plotter import *
from new_neuron_analysis.experiment_augment_input import get_experiment_path_for_this_run, \
     get_log_dir, get_result_dir, AttributeDict, os, get_project_dir, get_save_dir
import scipy.stats as stats
import random
import pandas as pd
import numpy as np
import bootstrapped.bootstrap as bas
import bootstrapped.stats_functions as bs_stats
import bootstrapped.compare_functions as bs_compare
import bootstrapped.power as bs_power
from scipy.stats import ks_2samp


def get_results(result_dir):
    labels = []
    log_dirs = []
    for label in os.listdir(result_dir):
        this_run_dir = f"{result_dir}/{label}"
        log_dir = get_log_dir(this_run_dir)
        if not os.path.exists(log_dir):
            continue

        labels.append(label)
        log_dirs.append(log_dir)


    return labels, log_dirs

def extra(label):
    top_num_to_include = label.split("top_num_to_include_")[1].split("_")[0]
    lr = float(label.split("learning_rate_")[1].split("_")[0])
    network_size = int(label.split("network_size_")[1])
    return top_num_to_include, network_size, lr



def significance_analysis(result_dir):
    top_num_to_include_plot_data = {}
    network_size_plot_data = {}
    lr_plot_data = {}


    for label in os.listdir(result_dir):

        this_run_dir = f"{result_dir}/{label}"
        log_dir = get_log_dir(this_run_dir)
        save_dir = get_save_dir(this_run_dir)
        if not os.path.exists(log_dir) or len(os.listdir(save_dir)) == 0:
            continue


        top_num_to_include, network_size, lr = extra(label)
        if top_num_to_include not in top_num_to_include_plot_data:
            top_num_to_include_plot_data[top_num_to_include] = {"log_dirs": [log_dir], "labels": [label]}
        else:
            top_num_to_include_plot_data[top_num_to_include]["log_dirs"].append(log_dir)
            top_num_to_include_plot_data[top_num_to_include]["labels"].append(label)

        if network_size not in network_size_plot_data:
            network_size_plot_data[network_size] = {"log_dirs": [log_dir], "labels": [label]}
        else:
            network_size_plot_data[network_size]["log_dirs"].append(log_dir)
            network_size_plot_data[network_size]["labels"].append(label)

        if lr not in lr_plot_data:
            lr_plot_data[lr] = {"log_dirs": [log_dir], "labels": [label]}
        else:
            lr_plot_data[lr]["log_dirs"].append(log_dir)
            lr_plot_data[lr]["labels"].append(label)


    for network_size, data in network_size_plot_data.items():
        for lr, data_lr in lr_plot_data.items():
            title = f"fix network_size {network_size} learning_rate {lr}"

            all_data_network_size_set = list(zip(data["labels"], data["log_dirs"]))
            all_data_lr_set = list(zip(data_lr["labels"], data_lr["log_dirs"]))


            final_all_data = set(all_data_network_size_set).intersection(set(all_data_lr_set))
            final_data = {}
            print(f"network and lr num of runs: {len(final_all_data)}")

            if len(final_all_data) == 0:
                continue

            final_data["labels"], final_data["log_dirs"] = zip(*final_all_data)

            _sig_analysis(final_data["labels"], final_data["log_dirs"], aug_num_timesteps, result_dir, title)


# 60 data, shuffle 3000 times, split and lift, then
def lift_bootstrap(data):
    lift = 1.25
    results = []
    for i in range(3000):
        random.shuffle(data)
        test = data[:len(data) // 2] * lift
        ctrl = data[len(data) // 2:]
        results.append(bas.bootstrap_ab(test, ctrl, bs_stats.mean, bs_compare.percent_change))
    return results

def power_analysis_group_by_run_and_seed(dirs, num_timesteps, xaxis, task_name, labels, include_details=False):
    """
    plot the results

    :param dirs: ([str]) the save location of the results to plot
    :param num_timesteps: (int) only plot the points below this value
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param task_name: (str) the title of the task to plot
    """

    xy_dict = {}
    for i, folder in enumerate(dirs):
        label = labels[i]
        top_num_to_include, network_size, lr = extra(label)
        new_label = f"top_num_to_include{top_num_to_include}, network_size{network_size}, lr{lr}"


        timesteps = load_results(folder)
        timesteps = timesteps[timesteps.l.cumsum() <= num_timesteps]
        if new_label not in xy_dict:
            xy_dict[new_label] = [ts2xy(timesteps, xaxis)]
        else:
            xy_dict[new_label].append(ts2xy(timesteps, xaxis))



    for label, xy_sublist in xy_dict.items():
        final_average_return = np.array([window_func(xy[0], xy[1], EPISODES_WINDOW, np.mean)[1][-1] for xy in xy_sublist])[:30]
        sim = bas.bootstrap(final_average_return, stat_func=bs_stats.mean)
        print(f"bootstrap interval {label}%.2f (%.2f, %.2f)" % (sim.value, sim.lower_bound, sim.upper_bound))


        sim = lift_bootstrap(final_average_return)
        sim = bs_power.power_stats(sim)
        print(sim)
        print(f"power of {label} {sim.transpose()['Insignificant']}, {sim.transpose()['Positive Significant']},{sim.transpose()['Negative Significant']}")



def _sig_analysis(labels, total_log_dirs, aug_num_timesteps, result_dir, title):
    task_name = "augmented_input"

    power_analysis_group_by_run_and_seed(dirs=total_log_dirs, num_timesteps=aug_num_timesteps, xaxis=X_TIMESTEPS, task_name=task_name, labels=labels, include_details=False)
    # figlegend.savefig(f"{result_dir}/{title}_legend.png")




if __name__ =="__main__":

    trained_policy_env = "DartWalker2d-v1"
    trained_policy_num_timesteps = 2000000
    policy_run_nums = [0]
    policy_seeds = [0]
    eval_seed = 3
    eval_run_num = 3
    aug_num_timesteps=1500000
    additional_note = "20000000033specialtest"
    for policy_run_num in policy_run_nums:
        for policy_seed in policy_seeds:
            result_dir = get_result_dir(trained_policy_env, trained_policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num, additional_note=additional_note)

            significance_analysis(result_dir)