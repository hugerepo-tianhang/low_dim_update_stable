#!/usr/bin/env python3

import numpy as np
from stable_baselines.low_dim_analysis.eval_util import *
from numpy import linalg as LA

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
from stable_baselines.low_dim_analysis.common import do_pca, plot_2d, dump_rows_write_csv, generate_run_dir

from stable_baselines.low_dim_analysis.common import plot_contour_trajectory, gen_subspace_coords, do_eval_returns, \
    get_allinone_concat_df, do_proj_on_first_n, low_dim_to_old_basis


def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return nan. It is used for logging only.

    :param arr: (np.ndarray)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)



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
        env_out.env.disableViewer = True
        env_out.env.visualize = False

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

    eprews, optimization_path = model.learn(total_timesteps=ppos_args.ppos_num_timesteps, give_optimization_path=True)


    toc = time.time()
    logger.log(f"####################################PPOS took {toc-tic} seconds")


    moving_ave_rewards = get_moving_aves(eprews, 100)

    return eprews, moving_ave_rewards, optimization_path


def main():


    import sys
    logger.log(sys.argv)
    ppos_arg_parser = get_common_parser()


    ppos_args, ppos_unknown_args = ppos_arg_parser.parse_known_args()
    full_space_alg = ppos_args.alg

    # origin = "final_param"
    origin = ppos_args.origin

    this_run_dir = get_dir_path_for_this_run(ppos_args)

    traj_params_dir_name = get_full_params_dir(this_run_dir)
    intermediate_data_dir = get_intermediate_data_dir(this_run_dir)
    save_dir = get_save_dir( this_run_dir)


    if not os.path.exists(intermediate_data_dir):
        os.makedirs(intermediate_data_dir)

    ppos_run_num, ppos_intermediate_data_dir = generate_run_dir(get_ppos_returns_dirname, intermediate_dir=intermediate_data_dir, n_comp=ppos_args.n_comp_to_use)
    '''
    ==========================================================================================
    get the pc vectors
    ==========================================================================================
    '''
    proj_or_not = (ppos_args.n_comp_to_use == 2)
    result = do_pca(ppos_args.n_components, ppos_args.n_comp_to_use, traj_params_dir_name, intermediate_data_dir,
                    proj=proj_or_not,
                    origin=origin, use_IPCA=ppos_args.use_IPCA, chunk_size=ppos_args.chunk_size)
    '''
    ==========================================================================================
    eval all xy coords
    ==========================================================================================
    '''



    if origin=="final_param":
        origin_param = result["final_concat_params"]
    else:
        origin_param = result["mean_param"]

    final_param = result["final_concat_params"]
    last_proj_coord = do_proj_on_first_n(final_param, result["first_n_pcs"], origin_param)


    if origin=="final_param":
        back_final_param = low_dim_to_old_basis(last_proj_coord, result["first_n_pcs"], origin_param)
        assert np.testing.assert_almost_equal(back_final_param, final_param)

    starting_coord = last_proj_coord
    logger.log(f"PPOS STASRTING CORRD: {starting_coord}")



    # starting_coord = (1/2*np.max(xcoordinates_to_eval), 1/2*np.max(ycoordinates_to_eval)) # use mean
    assert result["first_n_pcs"].shape[0] == ppos_args.n_comp_to_use

    eprews, moving_ave_rewards, optimization_path = do_ppos(ppos_args, result, intermediate_data_dir, origin_param)


    ppos_args.alg = full_space_alg
    plot_dir = get_plot_dir(ppos_args)
    ppos_plot_dir = get_ppos_plot_dir(plot_dir, ppos_args.n_comp_to_use, ppos_run_num)
    if not os.path.exists(ppos_plot_dir):
        os.makedirs(ppos_plot_dir)

    ret_plot_name = f"cma return on {ppos_args.n_comp_to_use} dim space of real pca plane, " \
                    f"explained {np.sum(result['explained_variance_ratio'][:ppos_args.n_comp_to_use])}"
    plot_ppos_returns(ppos_plot_dir, ret_plot_name, moving_ave_rewards, show=False)

    if ppos_args.n_comp_to_use == 2:
        proj_coords = result["proj_coords"]
        assert proj_coords.shape[1] == 2

        xcoordinates_to_eval, ycoordinates_to_eval = gen_subspace_coords(ppos_args, np.vstack((proj_coords, optimization_path)).T)

        eval_returns = do_eval_returns(ppos_args, intermediate_data_dir, result["first_n_pcs"], origin_param,
                        xcoordinates_to_eval, ycoordinates_to_eval, save_dir, pca_center=origin)

        plot_contour_trajectory(ppos_plot_dir, "end_point_origin_eval_return_contour_plot", xcoordinates_to_eval,
                                ycoordinates_to_eval, eval_returns, proj_coords[:, 0], proj_coords[:, 1],
                                result["explained_variance_ratio"][:2],
                                num_levels=25, show=False, sub_alg_path=optimization_path)



    opt_mean_path_in_old_basis = [low_dim_to_old_basis(projected_opt_params, result["first_n_pcs"], origin_param) for projected_opt_params in optimization_path]
    distance_to_final = [LA.norm(opt_mean - final_param, ord=2) for opt_mean in opt_mean_path_in_old_basis]
    distance_to_final_plot_name = f"distance_to_final over generations "
    plot_2d(ppos_plot_dir, distance_to_final_plot_name, np.arange(len(distance_to_final)), distance_to_final, "num generation", "distance_to_final", False)

    # plot_3d_trajectory(cma_plot_dir, "end_point_origin_eval_return_3d_plot", xcoordinates_to_eval, ycoordinates_to_eval,
    #                         eval_returns, proj_xcoord, proj_ycoord,
    #                         result["explained_variance_ratio"][:2],
    #                         num_levels=15, show=False)


if __name__ == '__main__':

    main()
