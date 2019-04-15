from stable_baselines.ppo2.run_mujoco import eval_return
import cma
from stable_baselines.low_dim_analysis.common import do_pca, plot_2d, get_allinone_concat_df, \
    get_projected_vector_in_old_basis, cal_angle ,unit_vector, cal_angle_between_nd_planes, cal_angle_between_2_2d_planes
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
    result = do_pca(cma_args.n_components, cma_args.n_comp_to_use, traj_params_dir_name, intermediate_data_dir,
                    proj=False,
                    origin="mean_param", use_IPCA=cma_args.use_IPCA, chunk_size=cma_args.chunk_size, reuse=True)
    logger.debug("after pca")

    final_pcs = result["first_n_pcs"]

    all_param_iterator = get_allinone_concat_df(dir_name=traj_params_dir_name, use_IPCA=True, chunk_size=cma_args.pc1_chunk_size)
    plane_angles_vs_final_plane_along_the_way = []
    ipca = IncrementalPCA(n_components=cma_args.n_comp_to_use)  # for sparse PCA to speed up
    for chunk in all_param_iterator:
        if all_param_iterator._currow <= cma_args.pc1_chunk_size * cma_args.skipped_chunks:
            logger.log(f"skipping: currow: {all_param_iterator._currow} skip threshold {cma_args.pc1_chunk_size * cma_args.skipped_chunks}")
            continue

        logger.log(f"currently at {all_param_iterator._currow}")
        ipca.partial_fit(chunk.values)

        first_n_pcs = ipca.components_[:cma_args.n_comp_to_use]
        assert final_pcs.shape[0] == first_n_pcs.shape[0]


        plane_angle = cal_angle_between_nd_planes(first_n_pcs, final_pcs)
        plane_angles_vs_final_plane_along_the_way.append(plane_angle)


    plot_dir = get_plot_dir(cma_args)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plane_angles_vs_final_plane_plot_dir = get_plane_angles_vs_final_plane_along_the_way_plot_dir(plot_dir, cma_args.n_comp_to_use)
    if not os.path.exists(plane_angles_vs_final_plane_plot_dir):
        os.makedirs(plane_angles_vs_final_plane_plot_dir)




    angles_plot_name = f"skiped {cma_args.skipped_chunks} plane_angles_vs_final_plane_plot_dir "
    plot_2d(plane_angles_vs_final_plane_plot_dir, angles_plot_name, np.arange(len(plane_angles_vs_final_plane_along_the_way)), plane_angles_vs_final_plane_along_the_way, "num of chunks", "angle with diff in degrees", False)



if __name__ == '__main__':

    main()

#TODO Give filenames more info to identify which hyperparameter is the data for

