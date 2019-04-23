from stable_baselines.ppo2.run_mujoco import eval_return
import cma

import numpy as np
from stable_baselines.low_dim_analysis.eval_util import *
from stable_baselines.low_dim_analysis.common import *

from stable_baselines import logger

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




def main():


    import sys
    logger.log(sys.argv)
    common_arg_parser = get_common_parser()
    cma_args, cma_unknown_args = common_arg_parser.parse_known_args()

    origin_name = cma_args.origin

    this_run_dir = get_dir_path_for_this_run(cma_args)
    plot_dir_alg = get_plot_dir(cma_args)

    traj_params_dir_name = get_full_params_dir(this_run_dir)
    intermediate_data_dir = get_intermediate_data_dir(this_run_dir, params_scope="pi")
    save_dir = get_save_dir( this_run_dir)


    if not os.path.exists(intermediate_data_dir):
        os.makedirs(intermediate_data_dir)
    if not os.path.exists(plot_dir_alg):
        os.makedirs(plot_dir_alg)


    start_file = get_full_param_traj_file_path(traj_params_dir_name, "pi_start")
    start_params = pd.read_csv(start_file, header=None).values[0]




    '''
    ==========================================================================================
    get the pc vectors
    ==========================================================================================
    '''
    pca_indexes = cma_args.other_pca_index
    pca_indexes = [int(pca_index) for pca_index in pca_indexes.split(":")]

    n_comp_to_project_on = pca_indexes
    result = do_pca(n_components=cma_args.n_components, traj_params_dir_name=traj_params_dir_name,
                    intermediate_data_dir=intermediate_data_dir, use_IPCA=cma_args.use_IPCA,
                    chunk_size=cma_args.chunk_size, reuse=True)
    logger.debug("after pca")

    if origin_name =="final_param":
        origin_param = result["final_params"]
    elif origin_name =="start_param":
        origin_param = start_params
    else:
        origin_param = result["mean_param"]


    pcs_to_project_on = result["pcs_components"][n_comp_to_project_on]

    proj_coords = project(result["pcs_components"], pcs_slice=n_comp_to_project_on, origin_name=origin_name,
                          origin_param=origin_param, IPCA_chunk_size=cma_args.chunk_size,
                          traj_params_dir_name=traj_params_dir_name, intermediate_data_dir=intermediate_data_dir,
                            n_components=cma_args.n_components, reuse=True)

    '''
    ==========================================================================================
    eval all xy coords
    ==========================================================================================
    '''
    other_pcs_plot_dir = get_other_pcs_plane_plot_dir(plot_dir_alg, pca_indexes)

    if not os.path.exists(other_pcs_plot_dir):
        os.makedirs(other_pcs_plot_dir)


    if proj_coords.shape[1] == 2:

        xcoordinates_to_eval, ycoordinates_to_eval = gen_subspace_coords(cma_args, proj_coords, center_length=5)

        eval_returns = do_eval_returns(cma_args, intermediate_data_dir, pcs_to_project_on,
                                       origin_param,
                        xcoordinates_to_eval, ycoordinates_to_eval, save_dir, pca_center=origin_name, reuse=False)

        plot_contour_trajectory(other_pcs_plot_dir, f"{pca_indexes}_final_origin_eval_return_contour_plot", xcoordinates_to_eval,
                                ycoordinates_to_eval, eval_returns, proj_coords[:, 0], proj_coords[:, 1],
                                result["explained_variance_ratio"][pca_indexes],
                                num_levels=25, show=False)
    elif proj_coords.shape[1] == 3:
        plot_3d_trajectory_path_only(other_pcs_plot_dir, f"{pca_indexes}_final_origin_3d_path_plot", proj_coords,
                                     explained_ratio=result["explained_variance_ratio"][pca_indexes])


if __name__ == '__main__':

    main()

#TODO Give filenames more info to identify which hyperparameter is the data for

