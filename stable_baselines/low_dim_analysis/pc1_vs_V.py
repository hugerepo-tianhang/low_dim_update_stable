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

    pcs_components = np.loadtxt(
        get_pcs_filename(intermediate_dir=intermediate_data_dir, n_comp=cma_args.num_comp_to_load), delimiter=',')

    angle = cal_angle(V, pcs_components[0])
    logger.log(f"@@@@@@@@@@@@ {angle}")
if __name__ == '__main__':

    main()

#TODO Give filenames more info to identify which hyperparameter is the data for

