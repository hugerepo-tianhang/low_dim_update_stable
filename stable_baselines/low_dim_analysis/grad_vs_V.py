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


def plot_2d_2(plot_dir_alg, name, X, grad_vs_v, pc1_vs_V, xlabel, ylabel, show):
    fig, ax = plt.subplots()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    ax.plot(X, grad_vs_v)
    ax.plot(X, pc1_vs_V)

    plt.legend(['in so far grad_vs_v', 'in so far pc1_vs_V'], loc='upper left')

    file_path = f"{plot_dir_alg}/{name}.pdf"
    if os.path.isfile(file_path):
        os.remove(file_path)
    logger.log(f"####saving to {file_path}")
    fig.savefig(file_path, dpi=300,
                bbox_inches='tight', format='pdf')
    if show: plt.show()


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

    all_param_iterator = get_allinone_concat_df(dir_name=traj_params_dir_name, use_IPCA=True, chunk_size=cma_args.pc1_chunk_size)
    angles_along_the_way = []
    grad_vs_Vs = []


    ipca = IncrementalPCA(n_components=cma_args.n_comp_to_use)  # for sparse PCA to speed up
    last_chunk_last = start_params
    for chunk in all_param_iterator:
        chunk = chunk.values
        if chunk.shape[0] < cma_args.n_comp_to_use:
            logger.log("skipping too few data")
            continue

        for i in range(chunk.shape[0]):
            if i == 0:
                grad = chunk[i] - last_chunk_last
            else:
                grad = chunk[i] - chunk[i-1]

            grad_angle = cal_angle(grad, V)
            grad_vs_Vs.append(grad_angle)

        last_chunk_last = chunk[-1]

        logger.log(f"currently at {all_param_iterator._currow}")
        ipca.partial_fit(chunk)

        angle = cal_angle(V, ipca.components_[0])
        angles_along_the_way.extend([angle]*chunk.shape[0])


    plot_dir = get_plot_dir(cma_args)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)


    #TODO ignore negative for now
    angles_along_the_way = np.array(angles_along_the_way)
    if angles_along_the_way[-1] > 90:
        angles_along_the_way = 180 - angles_along_the_way


    assert len(angles_along_the_way) == len(grad_vs_Vs)

    angles_plot_name = f"in so far grad and pc1 vs final - start " \
                       f"cma_args.pc1_chunk_size: {cma_args.pc1_chunk_size} "
    plot_2d_2(plot_dir, angles_plot_name, np.arange(len(grad_vs_Vs)),grad_vs_v=grad_vs_Vs, pc1_vs_V=angles_along_the_way,
              xlabel="num of chunks", ylabel="angle with diff in degrees", show=False)


if __name__ == '__main__':

    main()

#TODO Give filenames more info to identify which hyperparameter is the data for

