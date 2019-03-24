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
from stable_baselines.low_dim_analysis.eval_util import get_full_params_dir, get_dir_path_for_this_run, get_log_dir, get_save_dir


def get_run_num(alg, env_id, total_timesteps):

    run_num = 0
    while os.path.exists(get_dir_path_for_this_run(alg, total_timesteps, env_id, run_num)):
        run_num += 1
    return run_num



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

    run_num = get_run_num( "ppo2", args.env, args.num_timesteps)
    log_dir = get_log_dir( "ppo2", args.num_timesteps, args.env, run_num)
    save_dir = get_save_dir( "ppo2", args.num_timesteps, args.env, run_num)
    logger.configure(log_dir)


    def make_env():
        env_out = gym.make(args.env)
        env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
        return env_out

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(args.seed)
    policy = MlpPolicy

    # extra run info I added for my purposes


    full_param_traj_dir_path = get_full_params_dir( "ppo2", args.num_timesteps,
                                                   args.env, run_num)
    if os.path.exists(full_param_traj_dir_path):
        import shutil
        shutil.rmtree(full_param_traj_dir_path)
    os.makedirs(full_param_traj_dir_path)


    if os.path.exists(save_dir):
        import shutil
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    run_info = {"run_num": run_num,
                "env_id": args.env,
                "full_param_traj_dir_path": full_param_traj_dir_path}


    model = PPO2(policy=policy, env=env, n_steps=2048, nminibatches=32, lam=0.95, gamma=0.99, noptepochs=10,
                 ent_coef=0.0, learning_rate=3e-4, cliprange=0.2)
    model.tell_run_info(run_info)

    model.learn(total_timesteps=args.num_timesteps)

    model.save(f"{save_dir}/ppo2")
    env.save_running_average(save_dir)



def eval_return(args, save_dir, theta,  eval_timesteps, i):
    # save_dir = get_save_dir( "ppo2", 50000, "Hopper-v2", 0)

    def make_env():
        env_out = gym.make(args.env)
        env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
        return env_out
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    model = PPO2.load(f"{save_dir}/ppo2")
    # flat_params = model.get_flat()

    model.set_from_flat(theta)

    env.load_running_average(save_dir)



    logger.log("Running trained model")
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
            # episode_rew = safe_mean([ep_info['r'] for ep_info in ep_infos])
            # print(f'episode_rew={episode_rew}')
            obs = env.reset()

    return safe_mean([ep_info['r'] for ep_info in ep_infos])


if __name__ == '__main__':
    import sys
    train(sys.argv)
    # args = mujoco_arg_parser().parse_args()
    #
    # eval_return(args, None, None,  5000, 0)