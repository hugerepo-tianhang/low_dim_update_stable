#!/usr/bin/env python3
import numpy as np
import gym
from low_dim_update_stable.new_neuron_analysis.dir_tree_util import *
import sys
from stable_baselines.common.cmd_util import mujoco_arg_parser
from stable_baselines import bench, logger
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import csv
import os
from stable_baselines.low_dim_analysis.eval_util import *
import pandas as pd
from gym.envs.registration import register
import json
from new_neuron_analysis.analyse_data import read_data
from new_neuron_analysis.util import comp_dict
import multiprocessing
import time
from stable_baselines.common.vec_env.vec_video_recorder import VecVideoRecorder

import logging
import tensorflow as tf


def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return nan. It is used for logging only.

    :param arr: (np.ndarray)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)


def get_run_name_experiment(env, num_timesteps,run_num,seed, learning_rate, top_num_to_include, network_size):

    return f'env_{env}_time_step_{num_timesteps}_' \
           f'run_{run_num}_seed_{seed}_learning_rate_{learning_rate}_top_num_to_include_{top_num_to_include.start}:{top_num_to_include.stop}_network_size_{network_size}'


def get_experiment_path_for_this_run(env, num_timesteps,run_num,seed, learning_rate, top_num_to_include, result_dir, network_size):

    return f'{result_dir}/{get_run_name_experiment(env, num_timesteps,run_num,seed, learning_rate, top_num_to_include, network_size)}'


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

def read_linear_top_var(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed,
                                           eval_run_num, additional_note, metric_param):
    trained_policy_data_dir = get_result_dir(policy_env, policy_num_timesteps, policy_run_num,
                                policy_seed, eval_seed, eval_run_num, additional_note, metric_param)

    linear_top_vars_list_path = f"{trained_policy_data_dir}/linear_top_vars_list"
    if metric_param is not None:
        linear_top_vars_list_path += f"_metric_param_{metric_param}"
    linear_top_vars_list_path += ".json"

    linear_correlation_neuron_list_path = f"{trained_policy_data_dir}/linear_correlation_neuron_list"
    if metric_param is not None:
        linear_correlation_neuron_list_path += f"_metric_param_{metric_param}"
    linear_correlation_neuron_list_path += ".json"


    with open(linear_top_vars_list_path, 'r') as fp:
        linear_top_vars_list = json.load(fp)

    with open(linear_correlation_neuron_list_path, 'r') as fp:
        linear_correlation_neuron_list = json.load(fp)
    return linear_top_vars_list, linear_correlation_neuron_list

def read_all_data(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num, additional_note,
                  num_layers=2):
    lagrangian_values, input_values, layers_values, all_weights = read_data(policy_env, policy_num_timesteps,
                                                                            policy_run_num, policy_seed, eval_seed,
                                                                            eval_run_num, additional_note=additional_note,
                                                                            num_layers=num_layers)

    trained_policy_data_dir = get_data_dir(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed,
                                           eval_run_num, additional_note=additional_note)

    linear_global_dict_path = f"{trained_policy_data_dir}/linear_global_dict.json"


    non_linear_global_dict_path = f"{trained_policy_data_dir}/non_linear_global_dict.json"


    with open(linear_global_dict_path, 'r') as fp:
        linear_global_dict = json.load(fp)

    if os.path.exists(non_linear_global_dict_path):
        with open(non_linear_global_dict_path, 'r') as fp:
            non_linear_global_dict = json.load(fp)
    else:
        non_linear_global_dict = None
    # non_linear_global_dict
    return linear_global_dict , non_linear_global_dict, lagrangian_values, input_values, layers_values, all_weights

def get_wanted_lagrangians_and_neurons(keys_to_include, linear_top_vars_list, linear_correlation_neuron_list, linear_co_threshold):
    linear_top_vars_list_wanted = []
    linear_top_vars_list_wanted_to_print = []
    linear_top_neurons_list_wanted = []
    # for i, (key, ind, linear_co, normalized_SSE) in enumerate(linear_top_vars_list):
    #     if key in keys_to_include:
    #         linear_top_vars_list_wanted.append((key, ind))
    #         linear_top_neurons_list_wanted.append(linear_correlation_neuron_list[i])
    #         linear_top_vars_list_wanted_to_print.append(linear_top_vars_list[i])
    #
    # # linear_co_threshold is the slice that I'll pass in slice(0,20), this name is just for temp use
    # return linear_top_vars_list_wanted[linear_co_threshold], linear_top_neurons_list_wanted[linear_co_threshold], \
    #        linear_top_vars_list_wanted_to_print[linear_co_threshold]
    for i, neuron_coord in enumerate(linear_correlation_neuron_list):
        if neuron_coord in linear_top_neurons_list_wanted:
            continue
        else:
            key, ind, linear_co, new_metric = linear_top_vars_list[i]

            if linear_co > linear_co_threshold.start and key in keys_to_include:
                linear_top_neurons_list_wanted.append(neuron_coord)
                linear_top_vars_list_wanted.append((key, ind))
                linear_top_vars_list_wanted_to_print.append(linear_top_vars_list[i])
    return linear_top_vars_list_wanted, linear_top_neurons_list_wanted, linear_top_vars_list_wanted_to_print



#
# def run_experiment(augment_num_timesteps, top_num_to_include_slice, augment_seed, augment_run_num, network_size,
#                policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num, learning_rate,
#                additional_note, result_dir, lagrangian_inds_to_include=None):
#
#
#     time.sleep(2)
#     exception_logger = logging.getLogger()
#     handler = logging.FileHandler(f'{result_dir}/error.log')
#     handler.setLevel(logging.ERROR)
#     # # create a logging format
#     # formatter = logging.Formatter(f'top_num_to_include_slice_{top_num_to_include_slice}'
#     #                               f'_augment_seed_{augment_seed}_augment_run_num_{augment_run_num}'
#     #                               f'_network_size_{network_size}_learning_rate_{learning_rate}_{additional_note}\n%(message)s')
#     # handler.setFormatter(formatter)
#     #
#     # # add the handlers to the logger
#     exception_logger.addHandler(handler)
#
#     try:
#         _run_experiment(augment_num_timesteps, top_num_to_include_slice, augment_seed, augment_run_num, network_size,
#                    policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num,
#                    learning_rate, additional_note, result_dir, lagrangian_inds_to_include=lagrangian_inds_to_include)
#     except Exception as e:
#         exception_logger.error(f'########top_num_to_include_slice_{top_num_to_include_slice}'
#                                   f'_augment_seed_{augment_seed}_augment_run_num_{augment_run_num}'
#                                   f'_network_size_{network_size}_learning_rate_{learning_rate}_{additional_note}'
#                       , exc_info=e)


def run_model(model, env, vedio_dir):

    obs = env.reset()
    env = VecVideoRecorder(env, vedio_dir, record_video_trigger=lambda x: x == 0, video_length=3000, name_prefix="vis_this_policy")

    env.render()
    ep_infos = []


    for _ in range(2000):
        actions = model.step(obs)[0]

        obs, rew, done, infos = env.step(actions)


        for info in infos:
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                ep_infos.append(maybe_ep_info)

        env.render()
        done = done.any()
        if done:
            episode_rew = safe_mean([ep_info['r'] for ep_info in ep_infos])
            print(f'episode_rew={episode_rew}')
            obs = env.reset()

from matplotlib import pyplot as plt
def show_M_matrix(num_dof, lagrangian_inds_to_include, top_num_to_include_slice, save_dir):
    assert num_dof == int(num_dof)
    num_dof = int(num_dof)
    M = np.zeros((num_dof, num_dof))
    num_of_M = 0
    to_include = lagrangian_inds_to_include
    for (key, ind) in to_include:
        if key == "M":
            num_of_M += 1
            row = ind // num_dof
            col = ind % num_dof
            M[row, col] = 1

    fig, ax = plt.subplots()
    im = ax.imshow(M)

    fig_name = f"included M matrix, top_num_to_include_slice={top_num_to_include_slice}, num of M={num_of_M}"
    ax.set_title(fig_name)
    fig.tight_layout()
    plt.savefig(f"{save_dir}/{fig_name}")

from stable_baselines.low_dim_analysis.common_parser import get_common_parser

def run_experiment_with_trained(augment_num_timesteps, linear_co_threshold, augment_seed, augment_run_num, network_size,
                                policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num, learning_rate,
                                additional_note, result_dir, keys_to_include, metric_param, linear_top_vars_list=None, linear_correlation_neuron_list=None, visualize=False, ):
    with tf.variable_scope("trained_model"):
        common_arg_parser = get_common_parser()
        trained_args, cma_unknown_args = common_arg_parser.parse_known_args()
        trained_args.env = policy_env
        trained_args.seed = policy_seed
        trained_args.num_timesteps = policy_num_timesteps
        trained_args.run_num = policy_run_num
        trained_this_run_dir = get_dir_path_for_this_run(trained_args)
        trained_traj_params_dir_name = get_full_params_dir(trained_this_run_dir)
        trained_save_dir = get_save_dir(trained_this_run_dir)

        trained_final_file = get_full_param_traj_file_path(trained_traj_params_dir_name, "pi_final")
        trained_final_params = pd.read_csv(trained_final_file, header=None).values[0]

        trained_model = PPO2.load(f"{trained_save_dir}/ppo2", seed=augment_seed)
        trained_model.set_pi_from_flat(trained_final_params)

    args = AttributeDict()

    args.normalize = True
    args.num_timesteps = augment_num_timesteps
    args.run_num = augment_run_num
    args.alg = "ppo2"
    args.seed = augment_seed

    logger.log(f"#######TRAIN: {args}")
    # non_linear_global_dict
    timestamp = get_time_stamp('%Y_%m_%d_%H_%M_%S')
    experiment_label = f"learning_rate_{learning_rate}timestamp_{timestamp}_augment_num_timesteps{augment_num_timesteps}" \
                       f"_top_num_to_include{linear_co_threshold.start}_{linear_co_threshold.stop}" \
                       f"_augment_seed{augment_seed}_augment_run_num{augment_run_num}_network_size{network_size}" \
                       f"_policy_num_timesteps{policy_num_timesteps}_policy_run_num{policy_run_num}_policy_seed{policy_seed}" \
                       f"_eval_seed{eval_seed}_eval_run_num{eval_run_num}_additional_note_{additional_note}"

    if policy_env == "DartWalker2d-v1":
        entry_point = 'gym.envs.dart:DartWalker2dEnv_aug_input'
    elif policy_env == "DartHopper-v1":
        entry_point = 'gym.envs.dart:DartHopperEnv_aug_input'
    elif policy_env == "DartHalfCheetah-v1":
        entry_point = 'gym.envs.dart:DartHalfCheetahEnv_aug_input'
    elif policy_env == "DartSnake7Link-v1":
        entry_point = 'gym.envs.dart:DartSnake7LinkEnv_aug_input'
    else:
        raise NotImplemented()


    this_run_dir = get_experiment_path_for_this_run(entry_point, args.num_timesteps, args.run_num,
                                                    args.seed, learning_rate=learning_rate, top_num_to_include=linear_co_threshold,
                                                    result_dir=result_dir, network_size=network_size)
    full_param_traj_dir_path = get_full_params_dir(this_run_dir)
    log_dir = get_log_dir(this_run_dir)
    save_dir = get_save_dir(this_run_dir)


    create_dir_remove(this_run_dir)
    create_dir_remove(full_param_traj_dir_path)
    create_dir_remove(save_dir)
    create_dir_remove(log_dir)
    logger.configure(log_dir)

    # note this is only linear
    if linear_top_vars_list is None or linear_correlation_neuron_list is None:

        linear_top_vars_list, linear_correlation_neuron_list = read_linear_top_var(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed,
                                           eval_run_num, additional_note, metric_param=metric_param)

    lagrangian_inds_to_include, neurons_inds_to_include, linear_top_vars_list_wanted_to_print = \
        get_wanted_lagrangians_and_neurons(keys_to_include, linear_top_vars_list, linear_correlation_neuron_list, linear_co_threshold)


    with open(f"{log_dir}/lagrangian_inds_to_include.json", 'w') as fp:
        json.dump(lagrangian_inds_to_include, fp)
    with open(f"{log_dir}/linear_top_vars_list_wanted_to_print.json", 'w') as fp:
        json.dump(linear_top_vars_list_wanted_to_print, fp)
    with open(f"{log_dir}/neurons_inds_to_include.json", 'w') as fp:
        json.dump(neurons_inds_to_include, fp)


    args.env = f'{experiment_label}_{entry_point}-v1'
    register(
        id=args.env,
        entry_point=entry_point,
        max_episode_steps=1000,
        kwargs={"lagrangian_inds_to_include": None, "trained_model": trained_model,
                "neurons_inds_to_include": neurons_inds_to_include}
    )


    def make_env():
        env_out = gym.make(args.env)
        env_out.env.visualize = visualize
        env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
        return env_out

    env = DummyVecEnv([make_env])
    walker_env = env.envs[0].env.env
    walker_env.disableViewer = not visualize


    if args.normalize:
        env = VecNormalize(env)
    policy = MlpPolicy



    set_global_seeds(args.seed)
    walker_env.seed(args.seed)

    num_dof = walker_env.robot_skeleton.ndofs
    show_M_matrix(num_dof, lagrangian_inds_to_include, linear_co_threshold, log_dir)




    # extra run info I added for my purposes
    run_info = {"run_num": args.run_num,
                "env_id": args.env,
                "full_param_traj_dir_path": full_param_traj_dir_path}

    layers = [network_size, network_size]
    policy_kwargs = {"net_arch" : [dict(vf=layers, pi=layers)]}
    model = PPO2(policy=policy, env=env, n_steps=4096, nminibatches=64, lam=0.95, gamma=0.99,
                 noptepochs=10,
                 ent_coef=0.0, learning_rate=learning_rate, cliprange=0.2, optimizer='adam', policy_kwargs=policy_kwargs,
                 seed=args.seed)
    model.tell_run_info(run_info)
    model.learn(total_timesteps=args.num_timesteps, seed=args.seed)

    model.save(f"{save_dir}/ppo2")

    if args.normalize:
        env.save_running_average(save_dir)

    return log_dir
#
# def run_check_experiment(augment_num_timesteps, augment_seed, augment_run_num, network_size,
#                          policy_env, learning_rate):
#
#     args = AttributeDict()
#
#     args.normalize = True
#     args.num_timesteps = augment_num_timesteps
#     args.run_num = augment_run_num
#     args.alg = "ppo2"
#     args.seed = augment_seed
#     args.env = 'DartWalker2d-v1'
#
#     logger.log(f"#######TRAIN: {args}")
#
#     result_dir = get_original_env_test_dir(policy_env, augment_seed)
#
#
#     this_run_dir = get_experiment_path_for_this_run(args.env, args.num_timesteps, args.run_num,
#                                                     args.seed, learning_rate=learning_rate, top_num_to_include=0,
#                                                     result_dir=result_dir, network_size=network_size)
#     full_param_traj_dir_path = get_full_params_dir(this_run_dir)
#     log_dir = get_log_dir(this_run_dir)
#     save_dir = get_save_dir(this_run_dir)
#
#     current_process_id = multiprocessing.current_process()._identity
#     if current_process_id == (1,):
#         create_dir_if_not(result_dir)
#     create_dir_remove(this_run_dir)
#     create_dir_remove(full_param_traj_dir_path)
#     create_dir_remove(save_dir)
#     create_dir_remove(log_dir)
#     logger.configure(log_dir)
#
#
#
#
#     def make_env():
#         env_out = gym.make(args.env)
#         env_out.env.visualize = False
#         env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
#         return env_out
#
#     env = DummyVecEnv([make_env])
#     walker_env = env.envs[0].env.env
#     walker_env.disableViewer = True
#
#
#
#     if args.normalize:
#         env = VecNormalize(env)
#
#     policy = MlpPolicy
#
#     # extra run info I added for my purposes
#
#
#     run_info = {"run_num": args.run_num,
#                 "env_id": args.env,
#                 "full_param_traj_dir_path": full_param_traj_dir_path,
#                 "this_run_dir": this_run_dir}
#
#     layers = [network_size, network_size]
#
#     set_global_seeds(args.seed)
#     walker_env.seed(args.seed)
#
#     policy_kwargs = {"net_arch" : [dict(vf=layers, pi=layers)]}
#     model = PPO2(policy=policy, env=env, n_steps=4096, nminibatches=64, lam=0.95, gamma=0.99,
#                  noptepochs=10,
#                  ent_coef=0.0, learning_rate=learning_rate, cliprange=0.2, optimizer='adam', policy_kwargs=policy_kwargs,
#                  seed=args.seed)
#     model.tell_run_info(run_info)
#
#
#     # model.learn(total_timesteps=args.num_timesteps, seed=args.seed)
#     model.learn(total_timesteps=args.num_timesteps)
#
#     model.save(f"{save_dir}/ppo2")
#
#     if args.normalize:
#         env.save_running_average(save_dir)
#
#     return log_dir

if __name__ == "__main__":

    # num_timesteps = 1000000
    # total_num_to_includes = [10, 30, 60]
    # trained_policy_seeds = [0,1,2]
    # trained_policy_run_nums = [0,1,2]
    # network_sizes = [16, 32, 64, 128]
    # policy_seeds = [3]
    # policy_run_nums = [0]
    # policy_num_timesteps = 5000
    # policy_env = "DartWalker2d-v1"
    #
    # eval_seeds = [4]
    # eval_run_nums = [4]
    #
    # augment_seeds = [0]
    # augment_run_nums = range(2)
    # augment_num_timesteps = 5000
    # top_num_to_includes = [slice(0,0)]
    # network_sizes = [16]
    #
    #
    # augment_num_timesteps = 5000
    # additional_note = ""
    keys_to_include = ["COM", "M", "Coriolis", "total_contact_forces_contact_bodynode",
                       "com_jacobian", "contact_bodynode_jacobian"]
    # keys_to_include = ["COM", "M", "Coriolis", "com_jacobian"]

    # policy_num_timesteps = 5000000
    # policy_env = "DartSnake7Link-v1"
    # policy_run_nums = [1]
    # policy_seeds = [3]
    #
    # eval_seeds = [4]
    # eval_run_nums = [4]
    #
    # augment_seeds = [1]
    # augment_run_nums = [0]
    # augment_num_timesteps = 1500000
    # top_num_to_includes = [slice(0,20)]
    # network_sizes = [64]
    # additional_note = "lower_fluid_force_see_if_there_is_more_bug_in_simulator"
    # metric_params = [None]

    policy_num_timesteps = 2000000
    policy_env = "DartWalker2d-v1"
    policy_seeds = [0]
    policy_run_nums = [0]

    eval_seeds = [3]
    eval_run_nums = [3]

    augment_seeds = range(1)
    augment_run_nums = [0]
    augment_num_timesteps = 5000
    top_num_to_includes = [slice(0, 10), slice(0,0)]
    network_sizes = [64]
    additional_note = "sandbox"
    metric_params = [0.5]


    # policy_seeds = [0]
    # policy_run_nums = [0]
    # policy_num_timesteps = 5000
    # policy_envs = ["DartHalfCheetah-v1"]
    # # policy_envs = ["DartHalfCheetah-v1"]
    #
    # eval_seeds = [0]
    # eval_run_nums = [0]
    #
    # augment_seeds = [0]
    # augment_run_nums = range(2)
    # augment_num_timesteps = 5000
    # top_num_to_includes = [slice(0,10)]
    # network_sizes = [64]
    # additional_note = "sandbox"
    #     for total_num_to_include in total_num_to_includes:
    #     for trained_policy_run_num in trained_policy_run_nums:
    #         for trained_policy_seed in trained_policy_seeds:
    for policy_seed in policy_seeds:
        for policy_run_num in policy_run_nums:
            for eval_seed in eval_seeds:
                for eval_run_num in eval_run_nums:
                   for augment_seed in augment_seeds:
                       for augment_run_num in augment_run_nums:
                           for top_num_to_include in top_num_to_includes:
                                for network_size in network_sizes:
                                    for metric_param in metric_params:
                                        for learning_rate in [64 / network_size * 3e-4]:
                                            result_dir = get_result_dir(policy_env, policy_num_timesteps, policy_run_num,
                                                                        policy_seed, eval_seed, eval_run_num,
                                                                        additional_note)

                                            create_dir_if_not(result_dir)

                                            run_experiment_with_trained(augment_num_timesteps, linear_co_threshold=top_num_to_include, augment_seed=augment_seed,
                                                                        augment_run_num=augment_run_num, network_size=network_size,
                                                                        policy_env=policy_env, policy_num_timesteps=policy_num_timesteps,
                                                                        policy_run_num=policy_run_num, policy_seed=policy_seed, eval_seed=eval_seed,
                                                                        eval_run_num=eval_run_num, learning_rate=learning_rate,
                                                                        additional_note=additional_note, result_dir=result_dir, keys_to_include=keys_to_include,
                                                                        metric_param=metric_param, visualize=False)
    # run_check_experiment(augment_num_timesteps, augment_seed=0,
    #                      augment_run_num=0, network_size=64,
    #                      policy_env=policy_env, learning_rate=0.0001)    # from joblib import Parallel, delayed
    # run_check_experiment(augment_num_timesteps, augment_seed=0,
    #                      augment_run_num=1, network_size=64,
    #                      policy_env=policy_env, learning_rate=0.0001)    # from joblib import Parallel, delayed

    # results = Parallel(n_jobs=-1)(delayed(run_experiment)(env=env, num_timesteps=num_timesteps,
    #                                                        trained_policy_env="DartWalker2d-v1",
    #                        trained_policy_num_timesteps=3000000, trained_policy_seed=trained_policy_seed,
    #                        trained_policy_run_num=trained_policy_run_num, top_num_to_include=total_num_to_include,
    #                                             network_size=network_size)
    #                               for trained_policy_run_num in trained_policy_run_nums
    #                               for trained_policy_seed in trained_policy_seeds
    #                               for total_num_to_include in top_num_to_includes
    #                               for network_size in network_sizes)
    #
    # from new_neuron_analysis.plot_result import plot
    # labels, log_dirs = tuple(zip(*results))
    #
    # plot(labels, log_dirs)