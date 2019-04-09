from stable_baselines.ppo2.run_mujoco import eval_return
import cma

import numpy as np
from stable_baselines.low_dim_analysis.eval_util import *

from stable_baselines import logger

import pandas as pd
from sklearn.decomposition import PCA

from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import time
import os
from stable_baselines.common.cmd_util import mujoco_arg_parser
from stable_baselines.low_dim_analysis.common_parser import get_common_parser

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def cal_angle(v1, v2):

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def main():


    import sys
    logger.log(sys.argv)
    common_arg_parser = get_common_parser()
    cma_args, cma_unknown_args = common_arg_parser.parse_known_args()


    this_run_dir = get_dir_path_for_this_run(cma_args)

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
    from stable_baselines.low_dim_analysis.common import do_pca, plot_2d

    origin = "mean_param"
    result = do_pca(cma_args.n_components, cma_args.n_comp_to_use, traj_params_dir_name, intermediate_data_dir, proj=False,
                    origin=origin, use_IPCA=cma_args.use_IPCA, chunk_size=cma_args.chunk_size)

    final_params = result["final_concat_params"]
    all_pcs = result["pcs_components"]

    logger.log("grab start params")
    start_file = get_full_param_traj_file_path(traj_params_dir_name, "start")
    start_params = pd.read_csv(start_file, header=None).values[0]

    angles = []
    for pc in all_pcs:
        angles.append(cal_angle(pc, final_params - start_params))

    plot_dir = get_plot_dir(cma_args)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    angles_plot_name = f"angles with final - start start n_comp:{all_pcs.shape[0]} dim space of mean pca plane, "
    plot_2d(plot_dir, angles_plot_name, np.arange(all_pcs.shape[0]), angles, "num of pcs", "angle with diff", False)


if __name__ == '__main__':

    main()

#TODO Give filenames more info to identify which hyperparameter is the data for

