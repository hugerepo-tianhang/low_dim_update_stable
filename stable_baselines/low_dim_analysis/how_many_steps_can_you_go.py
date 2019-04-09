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



def plot_final_project_returns_returns(plot_dir_alg, name, projected_returns, start, end, show):

    X = np.arange(start, end + 1)

    assert len(X) == len(projected_returns)

    fig, ax = plt.subplots()
    plt.xlabel('num of comp used')
    plt.ylabel('eval returns for projected returns')

    ax.plot(X, projected_returns)
    fig.savefig(f"{plot_dir_alg}/{name}.pdf", dpi=300,
                bbox_inches='tight', format='pdf')
    if show: plt.show()


def main(n_comp_start=2, do_eval=True):


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
    from stable_baselines.low_dim_analysis.common import \
        calculate_projection_errors, plot_2d, get_allinone_concat_df, calculate_num_axis_to_explain

    origin = "mean_param"
    ratio_threshold = 0.99
    consec_threshold = 5
    error_threshold = 0.05

    tic = time.time()
    all_param_matrix = get_allinone_concat_df(dir_name=traj_params_dir_name).values
    toc = time.time()
    print('\nElapsed time getting the chunk concat diff took {:.2f} s\n'
          .format(toc - tic))

    n_comps = min(cma_args.n_comp_to_use, cma_args.chunk_size)

    num_to_explains = []

    deviates = []
    for i in range(0, len(all_param_matrix), cma_args.chunk_size):
        if i + cma_args.chunk_size >= len(all_param_matrix):
            break
        chunk = all_param_matrix[i:i + cma_args.chunk_size]

        pca = PCA(n_components=n_comps) # for sparse PCA to speed up
        pca.fit(chunk)

        num, explained = calculate_num_axis_to_explain(pca, ratio_threshold)
        num_to_explains.append(num)


        pcs_components = pca.components_

        num_to_deviate = 0
        consec = 0

        for j in range(i + cma_args.chunk_size, len(all_param_matrix)):

            errors = calculate_projection_errors(pca.mean_, pcs_components, all_param_matrix[j], num)
            if errors[0] >= error_threshold:

                consec += 1
                if consec >= consec_threshold:
                    break

            num_to_deviate += 1

        deviates.append(num_to_deviate)

    plot_dir = get_plot_dir(cma_args)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    deviate_plot_name = f"num of steps to deviates from this plane chunk_size: {cma_args.chunk_size} ratio_threshold: {ratio_threshold} consec_threshold: {consec_threshold}error_threshold: {error_threshold}, "
    plot_2d(plot_dir, deviate_plot_name, np.arange(len(deviates)), deviates, "num of chunks", "num of steps to deviates from this plane", False)

    num_to_explain_plot_name = f"num to explain chunk_size: {cma_args.chunk_size} "
    plot_2d(plot_dir, num_to_explain_plot_name, np.arange(len(num_to_explains)), num_to_explains, "num of chunks", "num_to_explains", False)


if __name__ == '__main__':

    main(do_eval=True)

#TODO Give filenames more info to identify which hyperparameter is the data for

