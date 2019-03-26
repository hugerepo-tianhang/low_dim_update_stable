#!/usr/bin/env python3
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



def train(args):
    """
    Runs the test
    """
    args, argv = mujoco_arg_parser().parse_known_args(args)
    logger.log(f"#######TRAIN: {args}")

    this_run_dir = get_dir_path_for_this_run("ppo2", args.num_timesteps, args.env, args.normalize, args.run_num)
    if os.path.exists(this_run_dir):
        import shutil
        shutil.rmtree(this_run_dir)
    os.makedirs(this_run_dir)

    log_dir = get_log_dir( this_run_dir)
    save_dir = get_save_dir( this_run_dir)
    logger.configure(log_dir)


    def make_env():
        env_out = gym.make(args.env)
        env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
        return env_out

    env = DummyVecEnv([make_env])
    if args.normalize:
        env = VecNormalize(env)

    set_global_seeds(args.seed)
    policy = MlpPolicy

    # extra run info I added for my purposes


    full_param_traj_dir_path = get_full_params_dir( this_run_dir)

    if os.path.exists(full_param_traj_dir_path):
        import shutil
        shutil.rmtree(full_param_traj_dir_path)
    os.makedirs(full_param_traj_dir_path)


    if os.path.exists(save_dir):
        import shutil
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    run_info = {"run_num": args.run_num,
                "env_id": args.env,
                "full_param_traj_dir_path": full_param_traj_dir_path}


    model = PPO2(policy=policy, env=env, n_steps=2048, nminibatches=32, lam=0.95, gamma=0.99, noptepochs=10,
                 ent_coef=0.0, learning_rate=3e-4, cliprange=0.2)
    model.tell_run_info(run_info)

    model.learn(total_timesteps=args.num_timesteps)

    model.save(f"{save_dir}/ppo2")


    if args.normalize:
        env.save_running_average(save_dir)



def eval_return(args, save_dir, theta,  eval_timesteps, i):
    logger.log(f"#######EVAL: {args}")

    def make_env():
        env_out = gym.make(args.env)
        env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
        return env_out
    env = DummyVecEnv([make_env])
    if args.normalize:
        env = VecNormalize(env)

    model = PPO2.load(f"{save_dir}/ppo2")
    if theta is not None:
        model.set_from_flat(theta)

    if args.normalize:
        env.load_running_average(save_dir)



    obs = np.zeros((env.num_envs,) + env.observation_space.shape)
    obs[:] = env.reset()
    ep_infos = []
    for _ in range(eval_timesteps):
        actions = model.step(obs)[0]
        obs, rew, done, infos = env.step(actions)

        for info in infos:
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                ep_infos.append(maybe_ep_info)

        # env.render()
        done = done.any()
        if done:
            if theta is None:
                episode_rew = safe_mean([ep_info['r'] for ep_info in ep_infos])
                print(f'episode_rew={episode_rew}')
            obs = env.reset()

    return safe_mean([ep_info['r'] for ep_info in ep_infos])


if __name__ == '__main__':
    import sys
    # import pandas as pd
    train(sys.argv)
    # args = mujoco_arg_parser().parse_args()
    # this_run_dir = get_dir_path_for_this_run("ppo2", args.num_timesteps,
    #                                          args.env, args.normalize, 0)
    # save_dir = get_save_dir(this_run_dir)
    #
    # logger.log("grab final params")
    # traj_params_dir_name = get_full_params_dir(this_run_dir)
    #
    # final_file = get_full_param_traj_file_path(traj_params_dir_name, "final")
    # final_concat_params = pd.read_csv(final_file, header=None).values[0]
    #
    # logger.log(eval_return(args, save_dir, final_concat_params,  5000, 0))