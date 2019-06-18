from stable_baselines.results_plotter import *
from new_neuron_analysis.experiment_augment_input import get_experiment_path_for_this_run, \
     get_log_dir, get_result_dir, AttributeDict, os, get_proj_dir, get_save_dir



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
    top_num_to_include = int(label.split("top_num_to_include_")[1].split("_")[0])
    lr = float(label.split("learning_rate_")[1].split("_")[0])
    network_size = int(label.split("network_size_")[1])
    return top_num_to_include, network_size, lr

def plot(result_dir):
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

    for top_num_to_include, data in top_num_to_include_plot_data.items():
        title = f"fix top_num_to_include {top_num_to_include}"
        _plot(data["labels"], data["log_dirs"], aug_num_timesteps, result_dir, title)

    for network_size, data in network_size_plot_data.items():
        title = f"fix network_size {network_size}"
        _plot(data["labels"], data["log_dirs"], aug_num_timesteps, result_dir, title)

    for lr, data in lr_plot_data.items():
        title = f"fix lr {lr}"
        _plot(data["labels"], data["log_dirs"], aug_num_timesteps, result_dir, title)




def plot_results_group_by_run_and_seed(dirs, num_timesteps, xaxis, task_name, labels):
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

    xy_list = []
    new_labels = []
    for label, xy_sublist in xy_dict.items():
        new_labels.append(label)
        lens = np.array([len(xy[1]) for xy in xy_sublist])
        amin = np.argmin(lens)
        min_len = np.min(lens)
        new_x = xy_sublist[amin][0]

        new_y = np.mean([xy_item[1][:min_len] for xy_item in xy_sublist], axis=0)
        xy_list.append((new_x, new_y))
    return plot_curves(xy_list, new_labels, xaxis, task_name)

def _plot(labels, total_log_dirs, aug_num_timesteps, result_dir, title):
    task_name = "augmented_input"

    fig, figlegend = plot_results_group_by_run_and_seed(dirs=total_log_dirs, num_timesteps=aug_num_timesteps, xaxis=X_TIMESTEPS, task_name=task_name, labels=labels)
    fig.savefig(f"{result_dir}/{title}.png")
    # figlegend.savefig(f"{result_dir}/{title}_legend.png")

if __name__ =="__main__":
    env = 'DartWalker2d_aug_input_current_trial-v1'
    trained_policy_env = "DartWalker2d-v1"
    trained_policy_num_timesteps = 2000000
    policy_run_num = 0
    policy_seed = 1
    eval_seed = 2
    eval_run_num = 2

    aug_num_timesteps = 800000

    result_dir = get_result_dir(trained_policy_env, trained_policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num)

    # labels, total_log_dirs = get_results(result_dir)

    plot(result_dir)