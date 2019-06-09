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
from stable_baselines.low_dim_analysis.eval_util import get_full_param_traj_file_path, get_full_params_dir, get_dir_path_for_this_run, get_log_dir, get_save_dir

import pandas as pd
from gym.envs.registration import register
import json

def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return nan. It is used for logging only.

    :param arr: (np.ndarray)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)


def get_run_name_experiment(args, top_num_to_include, network_size):

    return f'env_{args.env}_time_step_{args.num_timesteps}_' \
           f'run_{args.run_num}_seed_{args.seed}_top_num_to_include_{top_num_to_include}_network_size_{network_size}'


def get_experiment_path_for_this_run(args, top_num_to_include, result_dir, network_size):

    return f'{result_dir}/{get_run_name_experiment(args, top_num_to_include, network_size)}'

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
def train_augmented_input(augment_env, augment_num_timesteps, top_num_to_include, augment_seed, augment_run_num, network_size,
                          policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num):

    args = AttributeDict()

    args.normalize = True
    args.env = augment_env
    args.num_timesteps = augment_num_timesteps
    args.run_num = augment_run_num
    args.alg = "ppo2"
    args.seed = augment_seed

    logger.log(f"#######TRAIN: {args}")

    trained_policy_data_dir = get_data_dir(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num)

    linear_global_dict_path = f"{trained_policy_data_dir}/linear_global_dict.json"
    non_linear_global_dict_path = f"{trained_policy_data_dir}/non_linear_global_dict.json"

    with open(linear_global_dict_path, 'r') as fp:
        linear_global_dict = json.load(fp)

    with open(non_linear_global_dict_path, 'r') as fp:
        non_linear_global_dict = json.load(fp)

    register(
        id='DartWalker2d_aug_input_current_trial-v1',
        entry_point='gym.envs.dart:DartWalker2dEnv_aug_input',
        max_episode_steps=1000,
        kwargs={'linear_global_dict':linear_global_dict,
                'non_linear_global_dict':non_linear_global_dict,
                'top_to_include':top_num_to_include}

    )
    result_dir = get_result_dir(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num)

    this_run_dir = get_experiment_path_for_this_run(args, top_num_to_include=top_num_to_include,
                                                    result_dir=result_dir, network_size=network_size)
    full_param_traj_dir_path = get_full_params_dir(this_run_dir)
    log_dir = get_log_dir(this_run_dir)
    save_dir = get_save_dir(this_run_dir)


    create_dir_if_not(result_dir)
    create_dir_remove(this_run_dir)
    create_dir_remove(full_param_traj_dir_path)
    create_dir_remove(save_dir)
    create_dir_remove(log_dir)

    logger.configure(log_dir)






    def make_env():
        env_out = gym.make(args.env)
        env_out.env.visualize = False
        env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
        return env_out

    env = DummyVecEnv([make_env])
    env.envs[0].env.env.disableViewer = True

    if args.normalize:
        env = VecNormalize(env)

    set_global_seeds(args.seed)
    policy = MlpPolicy

    # extra run info I added for my purposes





    run_info = {"run_num": args.run_num,
                "env_id": args.env,
                "full_param_traj_dir_path": full_param_traj_dir_path}

    layers = [network_size, network_size]
    policy_kwargs = {"net_arch" : [dict(vf=layers, pi=layers)]}
    model = PPO2(policy=policy, env=env, n_steps=4096, nminibatches=64, lam=0.95, gamma=0.99,
                 noptepochs=10,
                 ent_coef=0.0, learning_rate=3e-4, cliprange=0.2, optimizer='adam', policy_kwargs=policy_kwargs)
    model.tell_run_info(run_info)

    model.learn(total_timesteps=args.num_timesteps)

    model.save(f"{save_dir}/ppo2")

    if args.normalize:
        env.save_running_average(save_dir)

    return log_dir

def run_experiment(augment_env, augment_num_timesteps, top_num_to_include, augment_seed, augment_run_num, network_size,
                          policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num):
    """
    Runs the test
    """

    log_dir = train_augmented_input(augment_env, augment_num_timesteps, top_num_to_include, augment_seed, augment_run_num, network_size,
                          policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num)

    experiment_label = f"augment_env{augment_env}, augment_num_timesteps{augment_num_timesteps}, top_num_to_include{top_num_to_include}" \
                       f", augment_seed{augment_seed}, augment_run_num{augment_run_num}, network_size{network_size}, policy_env{policy_env}" \
                       f", policy_num_timesteps{policy_num_timesteps}, policy_run_num{policy_run_num}, policy_seed{policy_seed}" \
                       f", eval_seed{eval_seed}, eval_run_num{eval_run_num}"

    return [experiment_label, log_dir]

if __name__ == "__main__":
    augment_env = 'DartWalker2d_aug_input_current_trial-v1'

    # num_timesteps = 1000000
    # total_num_to_includes = [10, 30, 60]
    # trained_policy_seeds = [0,1,2]
    # trained_policy_run_nums = [0,1,2]
    # network_sizes = [16, 32, 64, 128]

    augment_num_timesteps = 5000
    top_num_to_includes = [10]
    trained_policy_seeds = [0, 1]
    trained_policy_run_nums = [0, 1]
    network_sizes = [16]
    #     for total_num_to_include in total_num_to_includes:
    #     for trained_policy_run_num in trained_policy_run_nums:
    #         for trained_policy_seed in trained_policy_seeds:
    augment_env, augment_num_timesteps, top_num_to_include, augment_seed, augment_run_num, network_size,
    policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num
    run_experiment(    augment_env, augment_num_timesteps, top_num_to_include, augment_seed, augment_run_num, network_size,
    policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num)

    #
    # from joblib import Parallel, delayed
    #
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