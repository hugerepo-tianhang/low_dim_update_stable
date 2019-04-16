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
    intermediate_data_dir = get_intermediate_data_dir(this_run_dir, params_scope="pi")
    save_dir = get_save_dir( this_run_dir)


    if not os.path.exists(intermediate_data_dir):
        os.makedirs(intermediate_data_dir)


    '''
    ==========================================================================================
    get the pc vectors
    ==========================================================================================
    '''
    from stable_baselines.low_dim_analysis.common import do_pca, get_projected_data_in_old_basis, \
        calculate_projection_errors, plot_2d

    origin = "mean_param"
    result = do_pca(cma_args.n_components, cma_args.n_comp_to_use, traj_params_dir_name, intermediate_data_dir, proj=False,
                    origin=origin, use_IPCA=cma_args.use_IPCA, chunk_size=cma_args.chunk_size)

    final_params = result["final_concat_params"]
    all_pcs = result["pcs_components"]
    mean_param = result["mean_param"]
    projected = []
    projection_errors = []

    for num_pcs in range(n_comp_start, all_pcs.shape[0]+1):
        projected.append( get_projected_data_in_old_basis(mean_param, all_pcs, final_params, num_pcs) )
        proj_to_n_pcs_error = calculate_projection_errors(mean_param, all_pcs, final_params, num_pcs)
        assert len(proj_to_n_pcs_error) == 1
        projection_errors.extend(proj_to_n_pcs_error)

    plot_dir = get_plot_dir(cma_args)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if do_eval:


        from stable_baselines.ppo2.run_mujoco import eval_return
        thetas_to_eval = projected

        tic = time.time()

        eval_returns = Parallel(n_jobs=cma_args.cores_to_use, max_nbytes='100M') \
            (delayed(eval_return)(cma_args, save_dir, theta, cma_args.eval_num_timesteps, i) for (i, theta) in
             enumerate(thetas_to_eval))
        toc = time.time()
        logger.log(f"####################################1st version took {toc-tic} seconds")

        np.savetxt(get_projected_finals_eval_returns_filename(intermediate_dir=intermediate_data_dir,
                                                                     n_comp_start=n_comp_start,
                                                                     np_comp_end=all_pcs.shape[0],
                                                                     pca_center=origin),
                   eval_returns, delimiter=',')

        ret_plot_name = f"final project performances on start: {n_comp_start} end:{all_pcs.shape[0]} dim space of mean pca plane, "
        plot_final_project_returns_returns(plot_dir, ret_plot_name, eval_returns, n_comp_start, all_pcs.shape[0], show=False)



    error_plot_name = f"final project errors on start: {n_comp_start} end:{all_pcs.shape[0]} dim space of mean pca plane, "
    plot_2d(plot_dir, error_plot_name, np.arange(n_comp_start, all_pcs.shape[0]+1), projection_errors, "num of pcs", "projection error", False)



if __name__ == '__main__':

    main(do_eval=True)

#TODO Give filenames more info to identify which hyperparameter is the data for

