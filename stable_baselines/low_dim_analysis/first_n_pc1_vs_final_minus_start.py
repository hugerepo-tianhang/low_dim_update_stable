from stable_baselines.ppo2.run_mujoco import eval_return
import cma
from stable_baselines.low_dim_analysis.common import do_pca, plot_2d, get_allinone_concat_df, \
    get_projected_vector_in_old_basis, cal_angle ,unit_vector
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

    logger.log("grab start params")
    start_file = get_full_param_traj_file_path(traj_params_dir_name, "start")
    start_params = pd.read_csv(start_file, header=None).values[0]

    V = final_params - start_params
    all_grads_iterator = get_allinone_concat_df(dir_name=traj_params_dir_name, use_IPCA=True, chunk_size=cma_args.pc1_chunk_size, index="grads")
    all_param_iterator = get_allinone_concat_df(dir_name=traj_params_dir_name, use_IPCA=True, chunk_size=cma_args.pc1_chunk_size)
    angles_along_the_way = []
    grad_vs_pull = []
    pc1s = []
    ipca = IncrementalPCA(n_components=1)  # for sparse PCA to speed up

    i = 1
    for chunk in all_param_iterator:

        logger.log(f"currently at {all_param_iterator._currow}")
        ipca.partial_fit(chunk.values)
        pc1 = ipca.components_[0]
        if i % 2 == 0:
            pc1 = -pc1
        angle = cal_angle(V,pc1)
        angles_along_the_way.append(angle)
        pc1s.append(pc1)

        current_grad = all_grads_iterator.__next__().values[-1]
        current_param = chunk.values[-1]
        delta = unit_vector(current_param - start_params)
        pull_dir = V - delta
        pull_dir_vs_grad = cal_angle(pull_dir, current_grad)
        grad_vs_pull.append(pull_dir_vs_grad)
        i += 1


    plot_dir = get_plot_dir(cma_args)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    first_n_pc1_vs_V_plot_dir = get_first_n_pc1_vs_V_plot_dir(plot_dir, cma_args.pc1_chunk_size)
    if not os.path.exists(first_n_pc1_vs_V_plot_dir):
        os.makedirs(first_n_pc1_vs_V_plot_dir)




    angles_plot_name = f"angles algone the way dim space of mean pca plane "
    plot_2d(first_n_pc1_vs_V_plot_dir, angles_plot_name, np.arange(len(angles_along_the_way)), angles_along_the_way, "num of chunks", "angle with diff in degrees", False)

    grad_vs_pull_plot_name = f"grad vs V - delta_theta"
    plot_2d(first_n_pc1_vs_V_plot_dir, grad_vs_pull_plot_name, np.arange(len(grad_vs_pull)), grad_vs_pull, "num of chunks", "angle in degrees", False)

    pcpca = PCA(n_components=min(len(pc1s), 100))
    pcpca.fit(pc1s)
    logger.log(pcpca.explained_variance_ratio_)
    logger.log(cal_angle_plane(V, pcpca.components_[:2]))
    np.savetxt(f"{first_n_pc1_vs_V_plot_dir}/pcs_pcs.txt", pcpca.explained_variance_ratio_, delimiter=',')
    np.savetxt(f"{first_n_pc1_vs_V_plot_dir}/pcs_V_vs_pcapca_first_2_comp_plane.txt", np.array([cal_angle_plane(V, pcpca.components_[:2])]), delimiter=',')

    i = 0
    for angle in angles_along_the_way:
        if angle > 90:
            i += 1

    np.savetxt(f"{first_n_pc1_vs_V_plot_dir}/num of angles bigger than 90.txt",
                       np.array([i]), delimiter=',')


if __name__ == '__main__':

    main()

#TODO Give filenames more info to identify which hyperparameter is the data for

