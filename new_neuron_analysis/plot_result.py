from stable_baselines.results_plotter import *
from new_neuron_analysis.experiment_augment_input import get_experiment_path_for_this_run, \
     get_log_dir, get_result_dir, AttributeDict, os, get_project_dir, get_save_dir



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



def plot(result_dir, aug_num_timesteps):
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
        print(f"top_num_to_include num of runs: {len(data['labels'])}")

        _plot(data["labels"], data["log_dirs"], aug_num_timesteps, result_dir, title)

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

            _plot(final_data["labels"], final_data["log_dirs"], aug_num_timesteps, result_dir, title)

    for lr, data in lr_plot_data.items():
        title = f"fix lr {lr}"
        print(f"lr num of runs: {len(data['labels'])}")

        _plot(data["labels"], data["log_dirs"], aug_num_timesteps, result_dir, title)



def subsample(l, target_len):
    new_inds = np.sort(np.random.choice(len(l), target_len, replace=False))
    assert (new_inds == np.sort(new_inds)).all()
    assert new_inds[0] <= new_inds[-1]

    return l[new_inds]
# def subsample(l, target_len):
#     def decide_next_skip(prob_of_down, up, down):
#         r = np.random.random_sample()
#         if r > prob_of_down:
#             current_skip_num = up
#         else:
#             current_skip_num = down
#         return current_skip_num
#
#     memory_size_threshold = 800000
#     current_non_skipped = 0
#     total_num_dumped = 0
#
#     if total_timesteps > memory_size_threshold:
#         down_sample_fraction = (total_timesteps - memory_size_threshold)/total_timesteps
#         down = int(1 / down_sample_fraction)
#         up = down + 1
#         prob_of_down = (down_sample_fraction - 1 / up) * up * down
#
#
#         current_skip_num = decide_next_skip(prob_of_down, up, down)
#
#
#     if total_timesteps > memory_size_threshold and current_non_skipped >= current_skip_num:
#         current_skip_num = decide_next_skip(prob_of_down, up, down)
#         current_non_skipped = 0


def plot_results_group_by_run_and_seed(dirs, num_timesteps, xaxis, task_name, labels, include_details=False):
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
    fill_betweens = []
    new_labels = []
    xy_list_detail = []
    for label, xy_sublist in xy_dict.items():
        new_labels.append(label)
        lens = np.array([len(xy[1]) for xy in xy_sublist])
        amin = np.argmin(lens)
        min_len = np.min(lens)
        new_x = xy_sublist[amin][0]

        # new_ys = [xy_item[1][:min_len] for xy_item in xy_sublist]
        new_ys = [subsample(xy_item[1], min_len) for xy_item in xy_sublist]
        new_y = np.mean(new_ys, axis=0)
        std = np.std(new_ys, axis=0)

        fill_between_y_up = new_y + std
        fill_between_y_down = new_y - std

        fill_betweens.append((new_x, fill_between_y_up, fill_between_y_down))
        xy_list.append((new_x, new_y))

        assert len(fill_between_y_down) == len(fill_between_y_up) == len(new_y) == len(new_x)
        if not (len(fill_between_y_down) == len(fill_between_y_up) == len(new_y) == len(new_x)):
            print("ss")
        xy_list_detail.append(xy_sublist)
    if include_details:
        return plot_curves(xy_list, new_labels, xaxis, task_name, xy_list_detail, fill_betweens=fill_betweens)
    else:
        return plot_curves(xy_list, new_labels, xaxis, task_name, None, fill_betweens=fill_betweens)


def _plot(labels, total_log_dirs, aug_num_timesteps, result_dir, title):
    task_name = "augmented_input"

    fig, figlegend = plot_results_group_by_run_and_seed(dirs=total_log_dirs, num_timesteps=aug_num_timesteps, xaxis=X_TIMESTEPS, task_name=task_name, labels=labels, include_details=False)
    fig.savefig(f"{result_dir}/{title}.png")
    # figlegend.savefig(f"{result_dir}/{title}_legend.png")




if __name__ =="__main__":
    def run():
        trained_policy_env = "DartWalker2d-v1"
        trained_policy_num_timesteps = 5000000
        policy_run_nums = [1]
        policy_seeds = [3]
        eval_seed = 4
        eval_run_num = 4
        aug_num_timesteps=1500000
        additional_note = "sandbox"
        for policy_run_num in policy_run_nums:
            for policy_seed in policy_seeds:
                result_dir = get_result_dir(trained_policy_env, trained_policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num, additional_note=additional_note)

                plot(result_dir, aug_num_timesteps)

    run()