from stable_baselines.ppo2.run_mujoco import eval_return
import cma
from stable_baselines.low_dim_analysis.common import do_pca, plot_2d, get_allinone_concat_df, \
    get_projected_vector_in_old_basis, cal_angle
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
from stable_baselines.low_dim_analysis.common import cal_angle_plane


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

    all_param_iterator = get_allinone_concat_df(dir_name=traj_params_dir_name, use_IPCA=True, chunk_size=cma_args.pc1_chunk_size)
    all_grads_iterator = get_allinone_concat_df(dir_name=traj_params_dir_name, use_IPCA=True, chunk_size=cma_args.pc1_chunk_size, index="grads")


    angles_with_pc1_along_the_way = []
    grad_vs_final_min_current_param = []
    ipca = IncrementalPCA(1)  # for sparse PCA to speed up
    for chunk in all_param_iterator:

        logger.log(f"currently at {all_param_iterator._currow}")

        target_direction = final_params - chunk.values[-1]

        ipca.partial_fit(chunk.values)
        angle_with_pc1 = cal_angle(target_direction, ipca.components_[0])


        angles_with_pc1_along_the_way.append(angle_with_pc1)


        grads = all_grads_iterator.__next__().values
        for i, grad in enumerate(grads):

            grad_angle = cal_angle(grad, final_params - chunk.values[i])
            grad_vs_final_min_current_param.append(grad_angle)


    plot_dir = get_plot_dir(cma_args)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    angles_plot_name = f"final - current VS so far pc1" \
                       f"cma_args.pc1_chunk_size: {cma_args.pc1_chunk_size}"
    plot_2d(plot_dir, angles_plot_name, np.arange(len(angles_with_pc1_along_the_way)), angles_with_pc1_along_the_way, "num of chunks", "angle with diff in degrees", False)
    grad_vs_current_plot_name = f"##final - current param VS current grad" \
                       f"cma_args.pc1_chunk_size: {cma_args.pc1_chunk_size}"
    plot_2d(plot_dir, grad_vs_current_plot_name, np.arange(len(grad_vs_final_min_current_param)), grad_vs_final_min_current_param, "num of chunks", "angle with diff in degrees", False)


if __name__ == '__main__':

    main()

#TODO Give filenames more info to identify which hyperparameter is the data for

