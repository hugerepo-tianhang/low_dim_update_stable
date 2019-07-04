import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from stable_baselines.bench.monitor import load_results
import matplotlib.cm as cm

# matplotlib.use('TkAgg')  # Can change to 'Agg' for non-interactive mode
plt.rcParams['svg.fonttype'] = 'none'

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
          'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
          'darkgreen', 'tan', 'salmon', 'gold', 'darkred', 'darkblue', 'yellow', "black"]
import matplotlib._color_data as mcd



def rolling_window(array, window):
    """
    apply a rolling window to a np.ndarray

    :param array: (np.ndarray) the input Array
    :param window: (int) length of the rolling window
    :return: (np.ndarray) rolling window on the input array
    """
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)


def window_func(var_1, var_2, window, func):
    """
    apply a function to the rolling window of 2 arrays

    :param var_1: (np.ndarray) variable 1
    :param var_2: (np.ndarray) variable 2
    :param window: (int) length of the rolling window
    :param func: (numpy function) function to apply on the rolling window on variable 2 (such as np.mean)
    :return: (np.ndarray, np.ndarray)  the rolling output with applied function
    """
    var_2_window = rolling_window(var_2, window)
    function_on_var2 = func(var_2_window, axis=-1)
    return var_1[window - 1:], function_on_var2


def ts2xy(timesteps, xaxis):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """
    if xaxis == X_TIMESTEPS:
        x_var = np.cumsum(timesteps.l.values)
        y_var = timesteps.r.values
    elif xaxis == X_EPISODES:
        x_var = np.arange(len(timesteps))
        y_var = timesteps.r.values
    elif xaxis == X_WALLTIME:
        x_var = timesteps.t.values / 3600.
        y_var = timesteps.r.values
    else:
        raise NotImplementedError
    return x_var, y_var


def plot_curves(xy_list, labels, xaxis, title, xy_list_details=None):
    """
    plot the curves

    :param xy_list: ([(np.ndarray, np.ndarray)]) the x and y coordinates to plot
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param title: (str) the title of the plot
    :param xy_list_details: [ [(x,y), (x,y)], [...] ]
    """

    num = len(labels)
    width = len(labels[0])
    fig = plt.figure(figsize=(width//3, num*1.5))
    figLegend = plt.figure(figsize=(width//6, num*2))
    ax = fig.add_subplot(111)

    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0

    # colors = iter(cm.rainbow(np.linspace(0, 1, len(xy_list))))
    # names = list(mcd.CSS4_COLORS.keys())

    cm = plt.get_cmap('CMRmap')


    for (i, (x, y)) in enumerate(xy_list):
        # color = COLORS[i]
        color = cm(1.*i/len(xy_list))


        # ax.scatter(x, y, s=2, color=color)
        x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)  # So returns average of last EPISODE_WINDOW episodes
        ax.plot(x, y_mean, color=color, label=labels[i])
        if xy_list_details is not None:
            for (detail_x, detail_y) in xy_list_details[i]:
                detail_x, detail_y_mean = window_func(detail_x, detail_y, EPISODES_WINDOW,
                                        np.mean)  # So returns average of last EPISODE_WINDOW episodes

                ax.plot(detail_x, detail_y_mean, color=color, label=labels[i], alpha=0.5)

        ax.legend(loc="upper left")

    plt.xlim(minx, maxx)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel("Episode Rewards")
    plt.tight_layout()


    figLegend.legend(*ax.get_legend_handles_labels(), "center")
    return fig, figLegend

def plot_results(dirs, num_timesteps, xaxis, task_name, labels):
    """
    plot the results

    :param dirs: ([str]) the save location of the results to plot
    :param num_timesteps: (int) only plot the points below this value
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param task_name: (str) the title of the task to plot
    """

    tslist = []
    for folder in dirs:
        timesteps = load_results(folder)
        timesteps = timesteps[timesteps.l.cumsum() <= num_timesteps]
        tslist.append(timesteps)
    xy_list = [ts2xy(timesteps_item, xaxis) for timesteps_item in tslist]
    return plot_curves(xy_list, labels, xaxis, task_name)


def main():
    """
    Example usage in jupyter-notebook
    
    .. code-block:: python

        from stable_baselines import log_viewer
        %matplotlib inline
        log_viewer.plot_results(["./log"], 10e6, log_viewer.X_TIMESTEPS, "Breakout")

    Here ./log is a directory containing the monitor.csv files
    """
    import argparse
    import os
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dirs', help='List of log directories', nargs='*', default=['./log'])
    parser.add_argument('--num_timesteps', type=int, default=int(10e6))
    parser.add_argument('--xaxis', help='Varible on X-axis', default=X_TIMESTEPS)
    parser.add_argument('--task_name', help='Title of plot', default='Breakout')
    args = parser.parse_args()
    args.dirs = [os.path.abspath(folder) for folder in args.dirs]
    plot_results(args.dirs, args.num_timesteps, args.xaxis, args.task_name)
    plt.show()


if __name__ == '__main__':
    # main()
    # plot_results(["/home/panda-linux/PycharmProjects/low_dim_update_dart/low_dim_update_stable/stable_baselines/ppo2/optimizer_adam_env_DartHopper-v1_time_step_1000000_normalize_True_n_steps_4096_nminibatches_64_run_0/the_log_dir",
    #               "/home/panda-linux/PycharmProjects/low_dim_update_dart/low_dim_update_stable/stable_baselines/ppo2/optimizer_adam_env_DartHopper-v1_time_step_3000000_normalize_True_n_steps_4096_nminibatches_64_run_0/the_log_dir",
    #               "/home/panda-linux/PycharmProjects/low_dim_update_dart/low_dim_update_stable/stable_baselines/ppo2/optimizer_adam_env_DartHopper-v1_time_step_1000000_normalize_True_n_steps_4096_nminibatches_64_run_0_additional_notes_M_input/the_log_dir",
    #                 "/home/panda-linux/PycharmProjects/low_dim_update_dart/low_dim_update_stable/stable_baselines/ppo2/optimizer_adam_env_DartHopper-v1_time_step_1000000_normalize_True_n_steps_4096_nminibatches_64_run_1_additional_notes_M_input/the_log_dir",
    #                 "/home/panda-linux/PycharmProjects/low_dim_update_dart/low_dim_update_stable/stable_baselines/ppo2/optimizer_adam_env_DartHopper-v1_time_step_1000000_normalize_True_n_steps_4096_nminibatches_64_run_2_additional_notes_M_input/the_log_dir"]
    #              ,1000000, X_TIMESTEPS, "SSS", ["normal_run_0", "normal_run_1", "M_input_run_0", "M_input_run_1", "M_input_run_2"])
    # plt.show()

    plot_results([
        # "/home/panda-linux/PycharmProjects/low_dim_update_dart/low_dim_update_stable/stable_baselines/ppo2/optimizer_adam_env_DartWalker2d-v1_time_step_1000000_normalize_True_n_steps_4096_nminibatches_64_run_0_additional_notes_M_input/the_log_dir",
        #         "/home/panda-linux/PycharmProjects/low_dim_update_dart/low_dim_update_stable/stable_baselines/ppo2/optimizer_adam_env_DartWalker2d-v1_time_step_1000000_normalize_True_n_steps_4096_nminibatches_64_run_1_additional_notes_M_input/the_log_dir",
        #         "/home/panda-linux/PycharmProjects/low_dim_update_dart/low_dim_update_stable/stable_baselines/ppo2/optimizer_adam_env_DartWalker2d-v1_time_step_1000000_normalize_True_n_steps_4096_nminibatches_64_run_2_additional_notes_M_input/the_log_dir",
                "/home/panda-linux/PycharmProjects/low_dim_update_dart/low_dim_update_stable/stable_baselines/ppo2/optimizer_adam_env_DartWalker2d-v1_time_step_3000000_normalize_True_n_steps_4096_nminibatches_64_run_0/the_log_dir",
                "/home/panda-linux/PycharmProjects/low_dim_update_dart/low_dim_update_stable/stable_baselines/ppo2/optimizer_adam_env_DartWalker2d-v1_time_step_3000000_normalize_True_n_steps_4096_nminibatches_64_run_1/the_log_dir",
                "/home/panda-linux/PycharmProjects/low_dim_update_dart/low_dim_update_stable/stable_baselines/ppo2/optimizer_adam_env_DartWalker2d-v1_time_step_3000000_normalize_True_n_steps_4096_nminibatches_64_run_2/the_log_dir",
                  # "/home/panda-linux/PycharmProjects/low_dim_update_dart/low_dim_update_stable/stable_baselines/ppo2/optimizer_adam_env_DartWalker2dM_input-v1_time_step_1000000_normalize_True_n_steps_4096_nminibatches_64_run_0_additional_notes_TrianUpper_M_input/the_log_dir",
                  # "/home/panda-linux/PycharmProjects/low_dim_update_dart/low_dim_update_stable/stable_baselines/ppo2/optimizer_adam_env_DartWalker2dM_input-v1_time_step_1000000_normalize_True_n_steps_4096_nminibatches_64_run_1_additional_notes_TrianUpper_M_input/the_log_dir",
                  # "/home/panda-linux/PycharmProjects/low_dim_update_dart/low_dim_update_stable/stable_baselines/ppo2/optimizer_adam_env_DartWalker2dM_input-v1_time_step_1000000_normalize_True_n_steps_4096_nminibatches_64_run_2_additional_notes_TrianUpper_M_input/the_log_dir"
        ],
                1000000, X_TIMESTEPS, "SSS", ["normal_run_0", "normal_run_1", "norm 2"])

    plt.show()
