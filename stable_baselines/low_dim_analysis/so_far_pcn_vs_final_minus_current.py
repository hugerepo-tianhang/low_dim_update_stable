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
    angles_with_pc1_along_the_way = []
    angles_with_weighted_along_the_way = []
    angle_diff_pc1_vs_weighted = []
    ipca = IncrementalPCA(n_components=cma_args.n_comp_to_use)  # for sparse PCA to speed up
    for chunk in all_param_iterator:
        if chunk.shape[0] < cma_args.n_comp_to_use:
            logger.log("skipping too few data")
            continue
        logger.log(f"currently at {all_param_iterator._currow}")

        target_direction = final_params - chunk.values[-1]

        ipca.partial_fit(chunk.values)
        pcs = ipca.components_[:cma_args.n_comp_to_use]
        angle_with_pc1 = cal_angle(target_direction, pcs[0])

        pcs_weighted_direction = np.matmul(ipca.explained_variance_ratio_[:cma_args.n_comp_to_use], ipca.components_[:cma_args.n_comp_to_use])

        angle_with_weighted = cal_angle(target_direction, pcs_weighted_direction)

        #TODO ignore 90 or 180 for now
        if angle_with_pc1 > 90:
            angle_with_pc1 = 180 - angle_with_pc1
        if angle_with_weighted > 90:
            angle_with_weighted = 180 - angle_with_weighted

        angles_with_pc1_along_the_way.append(angle_with_pc1)
        angles_with_weighted_along_the_way.append(angle_with_weighted)
        angle_diff_pc1_vs_weighted.append(angle_with_pc1 - angle_with_weighted)

    plot_dir = get_plot_dir(cma_args)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    angles_plot_name = f"angles algone the way start start n_comp_used :{cma_args.n_comp_to_use} dim space of mean pca plane, " \
                       f"cma_args.pc1_chunk_size: {cma_args.pc1_chunk_size} "
    plot_2d(plot_dir, angles_plot_name, np.arange(len(angles_with_pc1_along_the_way)), angles_with_pc1_along_the_way, "num of chunks", "angle with diff in degrees", False)
    angles_plot_name = f"weighted angles algone the way start start n_comp_used :{cma_args.n_comp_to_use} dim space of mean pca plane, " \
                       f"cma_args.pc1_chunk_size: {cma_args.pc1_chunk_size} "
    plot_2d(plot_dir, angles_plot_name, np.arange(len(angles_with_weighted_along_the_way)), angles_with_weighted_along_the_way, "num of chunks", "angle with diff in degrees", False)
    angles_plot_name = f"angle_diff_pc1_vs_weighted n_comp_used :{cma_args.n_comp_to_use} dim space of mean pca plane, " \
                       f"cma_args.pc1_chunk_size: {cma_args.pc1_chunk_size} "
    plot_2d(plot_dir, angles_plot_name, np.arange(len(angle_diff_pc1_vs_weighted)), angle_diff_pc1_vs_weighted, "num of chunks", "angle with diff in degrees", False)


if __name__ == '__main__':

    main()

#TODO Give filenames more info to identify which hyperparameter is the data for

