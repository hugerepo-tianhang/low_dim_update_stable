#!/usr/bin/env python3

import numpy as np
from stable_baselines.low_dim_analysis.eval_util import *

from stable_baselines import logger

import pandas as pd
from sklearn.decomposition import PCA

from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import gym

from stable_baselines.common.cmd_util import mujoco_arg_parser
from stable_baselines import bench, logger
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.ppo_subspace.ppo2_model import PPO2
from stable_baselines.common.policies import MlpMultPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from stable_baselines.low_dim_analysis.eval_util import get_full_param_traj_file_path, get_full_params_dir, get_dir_path_for_this_run, get_log_dir, get_save_dir
import time
import os
from stable_baselines.low_dim_analysis.common_parser import get_common_parser




def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return nan. It is used for logging only.

    :param arr: (np.ndarray)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)



def parse_unknown_args(args):
    """
    Parse arguments not consumed by arg parser into a dicitonary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval
def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''

    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k, v in parse_unknown_args(args).items()}

def plot_ppos_returns(plot_dir_alg, name, eprews, show):

    X = np.arange(len(eprews))
    fig, ax = plt.subplots()
    plt.xlabel('num of episodes')
    plt.ylabel('epreturns')

    ax.plot(X, eprews)
    fig.savefig(f"{plot_dir_alg}/{name}.pdf", dpi=300,
                bbox_inches='tight', format='pdf')
    if show: plt.show()


def get_ppos_run_num(intermediate_data_dir, n_comp):
    run_num = 0
    while os.path.exists(get_ppos_returns_dirname(intermediate_data_dir, n_comp=n_comp, run_num=run_num)):
        run_num += 1
    return run_num

def get_moving_aves(mylist, N):
    cumsum, moving_aves = [0], []

    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)

    return moving_aves


def do_ppos(ppos_args, result, intermediate_data_dir, origin_param):

    ppos_args.alg = "ppo_subspace"

    logger.log(f"#######TRAIN: {ppos_args}")
    this_run_dir = get_dir_path_for_this_run(ppos_args)
    if os.path.exists(this_run_dir):
        import shutil
        shutil.rmtree(this_run_dir)
    os.makedirs(this_run_dir)

    log_dir = get_log_dir( this_run_dir)
    save_dir = get_save_dir( this_run_dir)
    full_param_traj_dir_path = get_full_params_dir(this_run_dir)
    if os.path.exists(full_param_traj_dir_path):
        import shutil
        shutil.rmtree(full_param_traj_dir_path)
    os.makedirs(full_param_traj_dir_path)


    if os.path.exists(save_dir):
        import shutil
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    run_info = {"full_param_traj_dir_path": full_param_traj_dir_path}


    logger.configure(log_dir)


    tic = time.time()


    def make_env():
        env_out = gym.make(ppos_args.env)
        env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
        return env_out

    env = DummyVecEnv([make_env])
    if ppos_args.normalize:
        env = VecNormalize(env)

    set_global_seeds(ppos_args.seed)
    policy = MlpMultPolicy



    model = PPO2(policy=policy, env=env, n_steps=ppos_args.n_steps, nminibatches=ppos_args.nminibatches,
                 lam=0.95, gamma=0.99, noptepochs=10,
                 ent_coef=0.0, learning_rate=3e-4, cliprange=0.2,
                 policy_kwargs={"num_comp":len(result["first_n_pcs"])}, pcs=result["first_n_pcs"], origin_theta=origin_param)
    model.tell_run_info(run_info)

    eprews, optimization_path = model.learn(total_timesteps=ppos_args.ppos_num_timesteps)


    toc = time.time()
    logger.log(f"####################################CMA took {toc-tic} seconds")


    moving_ave_rewards = get_moving_aves(eprews, 100)

    return eprews, moving_ave_rewards, optimization_path



def main(origin='final_param'):


    ppos_arg_parser = get_common_parser()

    ppos_args, ppos_unknown_args = ppos_arg_parser.parse_known_args()


    this_run_dir = get_dir_path_for_this_run(ppos_args)

    traj_params_dir_name = get_full_params_dir(this_run_dir)
    intermediate_data_dir = get_intermediate_data_dir(this_run_dir)
    save_dir = get_save_dir( this_run_dir)


    if not os.path.exists(intermediate_data_dir):
        os.makedirs(intermediate_data_dir)


    '''
    ==========================================================================================
    get the pc vectors
    ==========================================================================================
    '''
    from stable_baselines.low_dim_analysis.common import do_pca
    result = do_pca(ppos_args.n_components, ppos_args.n_comp_to_use, traj_params_dir_name, intermediate_data_dir, proj=True, origin=origin)

    '''
    ==========================================================================================
    eval all xy coords
    ==========================================================================================
    '''
    from stable_baselines.low_dim_analysis.common import plot_3d_trajectory, plot_contour_trajectory, gen_subspace_coords,do_eval_returns
    proj_coord = result["proj_coords"]
    proj_xcoord = proj_coord[0]
    proj_ycoord = proj_coord[1]

    starting_coord = (np.random.uniform(np.min(proj_xcoord), np.max(proj_xcoord)),
                    np.random.uniform(np.min(proj_ycoord), np.max(proj_ycoord)))
    logger.log(f"CMA STASRTING CORRD: {starting_coord}")

    if origin=="final_param":
        origin_param = result["final_concat_params"]
    else:
        origin_param = result["mean_param"]

    eprews, moving_ave_rewards, optimization_path = do_ppos(ppos_args, result, intermediate_data_dir, origin_param)


    ppos_run_num = get_ppos_run_num(intermediate_data_dir, n_comp=ppos_args.n_comp_to_use)
    ppos_intermediate_data_dir = get_ppos_returns_dirname(intermediate_data_dir, ppos_args.n_comp_to_use, ppos_run_num)
    if not os.path.exists(ppos_intermediate_data_dir):
        os.makedirs(ppos_intermediate_data_dir)

    np.savetxt(f"{ppos_intermediate_data_dir}/eprews.txt", eprews, delimiter=',')
    plot_dir = get_plot_dir(ppos_args)
    ppos_plot_dir = get_ppos_plot_dir(plot_dir, ppos_args.n_comp_to_use, ppos_run_num)
    if not os.path.exists(ppos_plot_dir):
        os.makedirs(ppos_plot_dir)

    ret_plot_name = f"100 moving ave ppos episode rewards on {ppos_args.n_comp_to_use} dim space of real pca plane"
    plot_ppos_returns(ppos_plot_dir, ret_plot_name, moving_ave_rewards, show=False)



    xcoordinates_to_eval, ycoordinates_to_eval = gen_subspace_coords(ppos_args, np.hstack((proj_coord, optimization_path)))

    eval_returns = do_eval_returns(ppos_args, intermediate_data_dir, result["first_n_pcs"], origin_param,
                    xcoordinates_to_eval, ycoordinates_to_eval, save_dir, pca_center=origin)

    plot_contour_trajectory(ppos_plot_dir, "end_point_origin_eval_return_contour_plot with ppos path", xcoordinates_to_eval, ycoordinates_to_eval, eval_returns, proj_xcoord, proj_ycoord,
                            result["explained_variance_ratio"][:2],
                            num_levels=15, show=False, cma_path=optimization_path)


if __name__ == '__main__':

    main()
