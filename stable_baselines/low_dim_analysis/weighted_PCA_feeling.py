from stable_baselines.ppo2.run_mujoco import eval_return
import cma
from stable_baselines.low_dim_analysis.common import do_pca, plot_2d, get_allinone_concat_df, \
    get_projected_vector_in_old_basis, cal_angle ,unit_vector, cal_angle_between_nd_planes, cal_angle_between_2_2d_planes
import math
import numpy as np
from stable_baselines.low_dim_analysis.eval_util import *
from wpca import WPCA
from stable_baselines import logger

import pandas as pd
from sklearn.decomposition import IncrementalPCA, PCA

from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import time
import os
from stable_baselines.common.cmd_util import mujoco_arg_parser
from stable_baselines.low_dim_analysis.common_parser import get_common_parser
from stable_baselines.low_dim_analysis.common import cal_angle_plane, postize_angle

def f(a):
    return a

def gen_weights(mat):
    return [f(a) for a in range(mat.shape[0])]

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
    pcs_components = np.loadtxt(
        get_pcs_filename(intermediate_dir=intermediate_data_dir, n_comp=cma_args.n_components), delimiter=',')


    final_pcs = pcs_components

    all_thetas_downsampled = get_allinone_concat_df(dir_name=traj_params_dir_name).values
    
    all_thetas_downsampled = all_thetas_downsampled[:5000]
    wpca = WPCA(n_components=1)  # for sparse PCA to speed up

    weights = gen_weights(all_thetas_downsampled)
    wpca.fit(all_thetas_downsampled, weights=weights)

    first_n_pcs = wpca.components_[:cma_args.n_comp_to_use]
    assert final_pcs.shape[0] == first_n_pcs.shape[0]


    logger.log("grab final params")
    final_file = get_full_param_traj_file_path(traj_params_dir_name, "final")
    final_params = pd.read_csv(final_file, header=None).values[0]

    logger.log("grab start params")
    start_file = get_full_param_traj_file_path(traj_params_dir_name, "start")
    start_params = pd.read_csv(start_file, header=None).values[0]

    V = final_params - start_params

    PCA_error_angle = postize_angle(cal_angle(final_pcs[0], V))
    WPCA_error_angle = postize_angle(cal_angle(first_n_pcs[0], V))


    np.test.assert_almost_equal(cal_angle_between_nd_planes(first_n_pcs[0], final_pcs[0]), cal_angle(final_pcs[0], first_n_pcs[0]))

    print(f"plane angel: {PCA_error_angle - WPCA_error_angle}")


if __name__ == '__main__':

    main()

#TODO Give filenames more info to identify which hyperparameter is the data for

