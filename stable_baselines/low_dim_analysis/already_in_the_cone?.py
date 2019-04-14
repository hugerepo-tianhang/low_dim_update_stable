from stable_baselines.ppo2.run_mujoco import eval_return
import cma
from stable_baselines.low_dim_analysis.common import do_pca, plot_2d, get_allinone_concat_df, \
    get_projected_vector_in_old_basis, cal_angle ,unit_vector
import math
import numpy as np
from stable_baselines.low_dim_analysis.eval_util import *

from stable_baselines import logger

import pandas as pd
from sklearn.decomposition import IncrementalPCA

from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import time
import os
from stable_baselines.common.cmd_util import mujoco_arg_parser
from stable_baselines.low_dim_analysis.common_parser import get_common_parser
from stable_baselines.low_dim_analysis.common import cal_angle_plane,postize_angle


def main():

    # requires  n_comp_to_use, pc1_chunk_size
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

    logger.log("grab final params")
    final_file = get_full_param_traj_file_path(traj_params_dir_name, "final")
    final_params = pd.read_csv(final_file, header=None).values[0]

    logger.log("grab start params")
    start_file = get_full_param_traj_file_path(traj_params_dir_name, "start")
    start_params = pd.read_csv(start_file, header=None).values[0]

    V = final_params - start_params

    pcs_components = np.loadtxt(
        get_pcs_filename(intermediate_dir=intermediate_data_dir, n_comp=cma_args.num_comp_to_load), delimiter=',')

    smallest_error_angle = postize_angle(cal_angle(V, pcs_components[0]))
    logger.log(f"@@@@@@@@@@@@ {smallest_error_angle}")

    curr_angles = []

    all_param_iterator = get_allinone_concat_df(dir_name=traj_params_dir_name, use_IPCA=True, chunk_size=cma_args.pc1_chunk_size)
    ipca = IncrementalPCA(n_components=1)  # for sparse PCA to speed up

    inside_final_cone = []
    for chunk in all_param_iterator:
        # for param in chunk.values:
        logger.log(f"currently at {all_param_iterator._currow}")
        ipca.partial_fit(chunk.values)

        angle = postize_angle(cal_angle(V, ipca.components_[0]))

        param = chunk.values[-1]



        curr_angle = cal_angle(param - start_params, ipca.components_[0])
        curr_angle = postize_angle(curr_angle)

        curr_angle_final = cal_angle(param - start_params, pcs_components[0])

        inside_final_cone.append(curr_angle_final - smallest_error_angle)
        curr_angles.append(curr_angle - angle)


    plot_dir = get_plot_dir(cma_args)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    angles_plot_name = f"$$$curr_angles$$$"
    plot_2d(plot_dir, angles_plot_name, np.arange(len(curr_angles)), curr_angles, "num of chunks", "angle with diff in degrees", False)
    angles_plot_name = f"inside final cone?"
    plot_2d(plot_dir, angles_plot_name, np.arange(len(inside_final_cone)), inside_final_cone, "num of chunks", "angle with diff in degrees", False)

if __name__ == '__main__':

    main()

#TODO Give filenames more info to identify which hyperparameter is the data for

