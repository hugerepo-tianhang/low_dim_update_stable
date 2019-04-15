from stable_baselines.ppo2.run_mujoco import eval_return
import cma
from stable_baselines.low_dim_analysis.common import do_pca, plot_2d, get_allinone_concat_df, \
    get_projected_vector_in_old_basis, cal_angle
import math
import numpy as np
from stable_baselines.low_dim_analysis.eval_util import *

from stable_baselines import logger

import pandas as pd
from sklearn.decomposition import IncrementalPCA, PCA

from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import time
import os
from stable_baselines.common.cmd_util import mujoco_arg_parser
from wpca import WPCA, EMPCA
from stable_baselines.low_dim_analysis.common_parser import get_common_parser
from stable_baselines.low_dim_analysis.common import *

def lin(a):
    return a

def lin1(a):
    return a*0.5

def lin2(a):
    return a*0.1

def f2(a):
    return np.exp(a)

def f3(a):
    return np.exp(0.5*a)

def f4(a):
    return np.exp(0.1*a)

Funcs = [lin, lin1, lin2, f2, f3, f4]
def gen_weights(mat, f):
    return [[f(a) for i in range(len(mat[0]))] for a in range(len(mat))]



def main():

    # requires  n_comp_to_use, pc1_chunk_size
    import sys
    logger.log(sys.argv)
    common_arg_parser = get_common_parser()
    cma_args, cma_unknown_args = common_arg_parser.parse_known_args()


    this_run_dir = get_dir_path_for_this_run(cma_args)

    traj_params_dir_name = get_full_params_dir(this_run_dir)
    intermediate_data_dir = get_intermediate_data_dir(this_run_dir)


    if not os.path.exists(intermediate_data_dir):
        os.makedirs(intermediate_data_dir)

    logger.log("grab final params")
    final_file = get_full_param_traj_file_path(traj_params_dir_name, "final")
    final_params = pd.read_csv(final_file, header=None).values[0]

    logger.log("grab start params")
    start_file = get_full_param_traj_file_path(traj_params_dir_name, "start")
    start_params = pd.read_csv(start_file, header=None).values[0]

    V = final_params - start_params

    '''
    ==========================================================================================
    get the pc vectors
    ==========================================================================================
    '''

    result = do_pca(cma_args.n_components, cma_args.n_comp_to_use, traj_params_dir_name, intermediate_data_dir,
                    proj=False,
                    origin="mean_param", use_IPCA=cma_args.use_IPCA, chunk_size=cma_args.chunk_size, reuse=True)
    logger.debug("after pca")

    final_plane = result["first_n_pcs"]

    count_file = get_full_param_traj_file_path(traj_params_dir_name, "total_num_dumped")
    total_num = pd.read_csv(count_file, header=None).values[0]


    all_param_iterator = get_allinone_concat_df(dir_name=traj_params_dir_name, use_IPCA=True, chunk_size=cma_args.pc1_chunk_size)
    unduped_angles_along_the_way = []
    duped_angles_along_the_way = []
    diff_along = []

    unweighted_pc1_vs_V_angles = []
    duped_pc1_vs_V_angles = []
    pc1_vs_V_diffs = []

    unweighted_ipca = IncrementalPCA(n_components=cma_args.n_comp_to_use)  # for sparse PCA to speed up

    all_matrix_buffer = []


    try:
        for chunk in all_param_iterator:
            chunk = chunk.values
            unweighted_ipca.partial_fit(chunk)
            unweighted_angle = cal_angle_between_nd_planes(final_plane,
                                                           unweighted_ipca.components_[:cma_args.n_comp_to_use])
            unweighted_pc1_vs_V_angle = postize_angle(cal_angle_between_nd_planes(V,
                                                           unweighted_ipca.components_[0]))

            unweighted_pc1_vs_V_angles.append(unweighted_pc1_vs_V_angle)



            #TODO ignore 90 or 180 for now
            if unweighted_angle > 90:
                unweighted_angle = 180 - unweighted_angle
            unduped_angles_along_the_way.append(unweighted_angle)


            np.testing.assert_almost_equal(cal_angle_between_nd_planes(unweighted_ipca.components_[:cma_args.n_comp_to_use][0], final_plane[0]),
                                        cal_angle(unweighted_ipca.components_[:cma_args.n_comp_to_use][0], final_plane[0]))




            all_matrix_buffer.extend(chunk)

            weights = gen_weights(all_matrix_buffer, Funcs[cma_args.func_index_to_use])
            logger.log(f"currently at {all_param_iterator._currow}")
            # ipca = PCA(n_components=1)  # for sparse PCA to speed up
            # ipca.fit(duped_in_so_far)
            wpca = WPCA(n_components=cma_args.n_comp_to_use)  # for sparse PCA to speed up
            tic = time.time()
            wpca.fit(all_matrix_buffer, weights=weights)
            toc = time.time()


            logger.debug(f"WPCA of {len(all_matrix_buffer)} data took {toc - tic} secs ")
            duped_angle = cal_angle_between_nd_planes(final_plane, wpca.components_[:cma_args.n_comp_to_use])


            duped_pc1_vs_V_angle = postize_angle(cal_angle_between_nd_planes(V, wpca.components_[0]))
            duped_pc1_vs_V_angles.append(duped_pc1_vs_V_angle)
            pc1_vs_V_diffs.append(duped_pc1_vs_V_angle - unweighted_pc1_vs_V_angle)


            #TODO ignore 90 or 180 for now
            if duped_angle > 90:
                duped_angle = 180 - duped_angle
            duped_angles_along_the_way.append(duped_angle)
            diff_along.append(unweighted_angle - duped_angle)
    finally:
        plot_dir = get_plot_dir(cma_args)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        angles_plot_name = f"WPCA" \
                           f"cma_args.pc1_chunk_size: {cma_args.pc1_chunk_size} "
        plot_2d(plot_dir, angles_plot_name, np.arange(len(duped_angles_along_the_way)), duped_angles_along_the_way, "num of chunks", "angle with diff in degrees", False)

        angles_plot_name = f"Not WPCA exponential 2" \
                           f"cma_args.pc1_chunk_size: {cma_args.pc1_chunk_size} "
        plot_2d(plot_dir, angles_plot_name, np.arange(len(unduped_angles_along_the_way)), unduped_angles_along_the_way, "num of chunks", "angle with diff in degrees", False)


        angles_plot_name = f"Not WPCA - WPCA diff_along exponential 2," \
                           f"cma_args.pc1_chunk_size: {cma_args.pc1_chunk_size} "
        plot_2d(plot_dir, angles_plot_name, np.arange(len(diff_along)), diff_along, "num of chunks", "angle with diff in degrees", False)




        angles_plot_name = f"PC1 VS VWPCA PC1 VS V" \
                           f"cma_args.pc1_chunk_size: {cma_args.pc1_chunk_size} "
        plot_2d(plot_dir, angles_plot_name, np.arange(len(duped_pc1_vs_V_angles)), duped_pc1_vs_V_angles, "num of chunks", "angle with diff in degrees", False)

        angles_plot_name = f"PC1 VS VNot WPCA PC1 VS V" \
                           f"cma_args.pc1_chunk_size: {cma_args.pc1_chunk_size} "
        plot_2d(plot_dir, angles_plot_name, np.arange(len(unweighted_pc1_vs_V_angles)), unweighted_pc1_vs_V_angles, "num of chunks", "angle with diff in degrees", False)


        angles_plot_name = f"PC1 VS VNot WPCA - WPCA diff PC1 VS V" \
                           f"cma_args.pc1_chunk_size: {cma_args.pc1_chunk_size} "
        plot_2d(plot_dir, angles_plot_name, np.arange(len(pc1_vs_V_diffs)), pc1_vs_V_diffs, "num of chunks", "angle with diff in degrees", False)


        del all_matrix_buffer
        import gc
        gc.collect()

if __name__ == '__main__':

    main()

#TODO Give filenames more info to identify which hyperparameter is the data for

