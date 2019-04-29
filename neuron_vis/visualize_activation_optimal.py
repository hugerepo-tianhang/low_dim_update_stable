from stable_baselines.ppo2.run_mujoco import eval_return
import cma

import numpy as np
from stable_baselines.low_dim_analysis.eval_util import *
from stable_baselines.low_dim_analysis.common import *

from stable_baselines import logger
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
from sklearn.decomposition import PCA

from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import time
import os
from stable_baselines.common.cmd_util import mujoco_arg_parser
from stable_baselines.low_dim_analysis.common_parser import get_common_parser
from numpy import linalg as LA
import numpy as np
import gym

from stable_baselines.common.cmd_util import mujoco_arg_parser
from stable_baselines import bench, logger
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import csv
import os
from stable_baselines.low_dim_analysis.eval_util import get_full_param_traj_file_path, get_full_params_dir, get_dir_path_for_this_run, get_log_dir, get_save_dir
def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return nan. It is used for logging only.

    :param arr: (np.ndarray)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)


def neuron_values_generator(args, save_dir, pi_theta, eval_timesteps):
    # logger.log(f"#######EVAL: {args}")


    neuron_values_list = []
    def make_env():
        env_out = gym.make(args.env)

        env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
        return env_out
    env = DummyVecEnv([make_env])

    if args.normalize:
        env = VecNormalize(env)

    model = PPO2.load(f"{save_dir}/ppo2") # this also loads V function
    if pi_theta is not None:
        model.set_pi_from_flat(pi_theta)

    if args.normalize:
        env.load_running_average(save_dir)



    obs = np.zeros((env.num_envs,) + env.observation_space.shape)
    obs[:] = env.reset()
    env.render()
    ep_infos = []
    for _ in range(eval_timesteps):
        actions = model.step(obs)[0]
        neuron_values = model.give_neuron_values(obs)

        # neuron_values_list.append( neuron_values )
        yield neuron_values
        obs, rew, done, infos = env.step(actions)
        env.render()

        # time.sleep(1)
        for info in infos:
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                ep_infos.append(maybe_ep_info)

        # env.render()
        done = done.any()
        if done:

            episode_rew = safe_mean([ep_info['r'] for ep_info in ep_infos])
            print(f'episode_rew={episode_rew}')
            obs = env.reset()
    # return neuron_values_list
    # return safe_mean([ep_info['r'] for ep_info in ep_infos])

def preload_neurons(args, save_dir, pi_theta, eval_timesteps):
    # logger.log(f"#######EVAL: {args}")


    neuron_values_list = []
    def make_env():
        env_out = gym.make(args.env)

        env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
        return env_out
    env = DummyVecEnv([make_env])

    if args.normalize:
        env = VecNormalize(env)

    model = PPO2.load(f"{save_dir}/ppo2") # this also loads V function
    if pi_theta is not None:
        model.set_pi_from_flat(pi_theta)

    if args.normalize:
        env.load_running_average(save_dir)



    obs = np.zeros((env.num_envs,) + env.observation_space.shape)
    obs[:] = env.reset()
    env.render()
    ep_infos = []
    for _ in range(eval_timesteps):
        actions = model.step(obs)[0]
        neuron_values = model.give_neuron_values(obs)

        neuron_values_list.append( neuron_values )
        # yield neuron_values
        obs, rew, done, infos = env.step(actions)
        env.render()

        # time.sleep(1)
        for info in infos:
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                ep_infos.append(maybe_ep_info)

        # env.render()
        done = done.any()
        if done:
            episode_rew = safe_mean([ep_info['r'] for ep_info in ep_infos])
            print(f'episode_rew={episode_rew}')
            obs = env.reset()
    return neuron_values_list
    # return safe_mean([ep_info['r'] for ep_info in ep_infos])

def give_normalizers(preload_neuron_values_list):


    def process_values(neuron_values):
        latents = neuron_values[:-2]
        latents = np.concatenate(latents, axis=1)

        dist = neuron_values[-2:]
        dist = np.concatenate(dist, axis=1)

        return [np.max(latents), np.min(latents), np.max(dist), np.min(dist)]

    result = np.array([process_values(neuron_values) for neuron_values in preload_neuron_values_list])


    latent_max, latent_min, dist_max, dist_min = np.max(result[:,0]),\
                                                 np.min(result[:,1]), \
                                                 np.max(result[:,2]), \
                                                 np.min(result[:,3])

    #since it's tanh, HARDCODE -1, 1
    latent_norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)


    dist_norm = matplotlib.colors.Normalize(vmin=dist_min, vmax=dist_max)
    return latent_norm, dist_norm

def main():


    import sys
    logger.log(sys.argv)
    common_arg_parser = get_common_parser()
    args, cma_unknown_args = common_arg_parser.parse_known_args()

    this_run_dir = get_dir_path_for_this_run(args)
    plot_dir_alg = get_plot_dir(args)

    traj_params_dir_name = get_full_params_dir(this_run_dir)
    intermediate_data_dir = get_intermediate_data_dir(this_run_dir, params_scope="pi")
    save_dir = get_save_dir( this_run_dir)


    if not os.path.exists(intermediate_data_dir):
        os.makedirs(intermediate_data_dir)
    if not os.path.exists(plot_dir_alg):
        os.makedirs(plot_dir_alg)


    final_file = get_full_param_traj_file_path(traj_params_dir_name, "pi_final")
    final_params = pd.read_csv(final_file, header=None).values[0]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')

    preload_neuron_values_list = preload_neurons(args, save_dir, final_params, args.eval_num_timesteps)
    latent_norm, dist_norm = give_normalizers(preload_neuron_values_list)
    latent_cmap = plt.get_cmap("Oranges")
    dist_cmap = plt.get_cmap("Greys")

    neuron_values_gen = neuron_values_generator(args, save_dir, final_params, args.eval_num_timesteps)


    left, right, bottom, top = 0.1, 0.9, 0.1, 0.9
    result_artists = []

    def init():
        try:
            first_neurons = neuron_values_gen.__next__()
        except StopIteration:
            return
        layer_sizes = [layer.shape[1] for layer in first_neurons]
        first_neurons = np.array([neuron_value.reshape(-1) for neuron_value in first_neurons])

        v_spacing = (top - bottom) / float(max(layer_sizes))
        h_spacing = (right - left) / float(len(layer_sizes) - 1)
        # Nodes

        for n, neuron_layer_value in enumerate(first_neurons):
            neuron_layer_value = neuron_layer_value.reshape(-1)
            layer_size = len(neuron_layer_value)
            layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
            for m, neuron_value in enumerate(neuron_layer_value):
                if n >= len(first_neurons)-2:
                    # dist
                    circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                        color=dist_cmap(dist_norm(neuron_value)), ec='k', zorder=4)
                else:
                    #latent
                    circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                        color=latent_cmap(latent_norm(neuron_value)), ec='k', zorder=4)
                ax.add_artist(circle)
                result_artists.append(circle)
        return result_artists

    def update_neuron(neuron_values):
        num_of_dist_neurons = np.concatenate(neuron_values[-2:], axis=1).shape[1]


        neuron_values = np.concatenate(neuron_values, axis=1).reshape(-1).ravel()
        for i, neuron_value in enumerate(neuron_values):
            if i >= neuron_values.shape[0] - num_of_dist_neurons:
                # dist
                result_artists[i].set_color(dist_cmap(dist_norm(neuron_value)))
            else:
                #latent
                result_artists[i].set_color(latent_cmap(latent_norm(neuron_value)))

        # plt.draw()
        return result_artists




    rot_animation = FuncAnimation(fig, update_neuron, frames=neuron_values_gen, init_func=init, interval=100)
    plt.show()

    print(f"~~~~~~~~~~~~~~~~~~~~~~saving to {plot_dir_alg}/neuron_vis.pdf")
    file_path = f"{plot_dir_alg}/neuron_firing.gif"
    if os.path.isfile(file_path):
        os.remove(file_path)
    rot_animation.save(file_path, dpi=80, writer='imagemagick')


if __name__ == '__main__':

    main()

#TODO Give filenames more info to identify which hyperparameter is the data for

