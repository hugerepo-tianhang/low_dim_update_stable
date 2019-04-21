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

import os
from stable_baselines.low_dim_analysis.eval_util import get_full_param_traj_file_path, get_full_params_dir, get_dir_path_for_this_run, get_log_dir, get_save_dir


def plot_3d_trajectory_path_only(dir_path, file_name, projected_path, explained_ratio, show=False):
    assert projected_path.shape[1] == 3
    assert len(explained_ratio) == 3
    """3d + trajectory"""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = projected_path.T[0]
    ys = projected_path.T[1]
    zs = projected_path.T[2]

    ax.plot(xs, ys, zs)

    ax.set_xlabel(f'explained:{explained_ratio[0]}')
    ax.set_ylabel(f'explained:{explained_ratio[1]}')
    ax.set_zlabel(f'explained:{explained_ratio[2]}')

    print(f"~~~~~~~~~~~~~~~~~~~~~~saving to {dir_path}/{file_name}.pdf")
    file_path = f"{dir_path}/{file_name}.pdf"
    if os.path.isfile(file_path):
        os.remove(file_path)
    fig.savefig(file_path, dpi=300,
                bbox_inches='tight', format='pdf')
    if show: plt.show()



def main():


    import sys
    logger.log(sys.argv)
    common_arg_parser = get_common_parser()
    cma_args, cma_unknown_args = common_arg_parser.parse_known_args()

    origin_name = "final_param"

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

    plot_3d_trajectory_path_only(other_pcs_plot_dir, f"{pca_indexes}_final_origin_3d_path_plot", proj_coords,
                                 explained_ratio=result["explained_variance_ratio"][pca_indexes])




if __name__ == '__main__':

    main()

#TODO Give filenames more info to identify which hyperparameter is the data for

