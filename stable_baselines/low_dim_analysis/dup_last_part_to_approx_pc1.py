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
from stable_baselines.low_dim_analysis.common_parser import get_common_parser
from stable_baselines.low_dim_analysis.common import cal_angle_plane

def dup_so_far_buffer(all_params_so_far, last_percentage, num):
    total = len(all_params_so_far)
    last_start_index = total - int(total * last_percentage)
    repeats = np.zeros(total, dtype=int)
    repeats[last_start_index:] = num
    dup = np.repeat(all_params_so_far, repeats, axis=0)
    return dup

def gen_last_percentage(currow, total_row):
    a = currow/total_row
    return 0.5 * np.exp(-2*a)


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

    count_file = get_full_param_traj_file_path(traj_params_dir_name, "total_num_dumped")
    total_num = pd.read_csv(count_file, header=None).values[0]

    V = final_params - start_params

    all_param_iterator = get_allinone_concat_df(dir_name=traj_params_dir_name, use_IPCA=True, chunk_size=cma_args.pc1_chunk_size)
    unduped_angles_along_the_way = []
    duped_angles_along_the_way = []
    diff_along = []
    num = 2 #TODO hardcode!
    undup_ipca = IncrementalPCA(n_components=1)  # for sparse PCA to speed up

    all_matrix_buffer = []
    aaa = -1
    for chunk in all_param_iterator:
        aaa+=1
        if aaa >= 10:
            break
        chunk = chunk.values
        undup_ipca.partial_fit(chunk)
        unduped_angle = cal_angle(V, undup_ipca.components_[0])

        #TODO ignore 90 or 180 for now
        if unduped_angle > 90:
            unduped_angle = 180 - unduped_angle
        unduped_angles_along_the_way.append(unduped_angle)







        all_matrix_buffer.extend(chunk)

        last_percentage = gen_last_percentage(all_param_iterator._currow, total_num)
        duped_in_so_far = dup_so_far_buffer(all_matrix_buffer, last_percentage, num)

        logger.log(f"currently at {all_param_iterator._currow}, last_pecentage: {last_percentage}")
        # ipca = PCA(n_components=1)  # for sparse PCA to speed up
        # ipca.fit(duped_in_so_far)
        ipca = IncrementalPCA(n_components=1)  # for sparse PCA to speed up
        for i in range(0, len(duped_in_so_far), cma_args.chunk_size):
            logger.log(f"partial fitting: i : {i} len(duped_in_so_far): {len(duped_in_so_far)}")
            if i + cma_args.chunk_size > len(duped_in_so_far):
                ipca.partial_fit(duped_in_so_far[i:])
            else:
                ipca.partial_fit(duped_in_so_far[i: i + cma_args.chunk_size])

        duped_angle = cal_angle(V, ipca.components_[0])


        #TODO ignore 90 or 180 for now
        if duped_angle > 90:
            duped_angle = 180 - duped_angle
        duped_angles_along_the_way.append(duped_angle)
        diff_along.append(unduped_angle - duped_angle)

    plot_dir = get_plot_dir(cma_args)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    angles_plot_name = f"duped exponential 2, num dup: {num}" \
                       f"cma_args.pc1_chunk_size: {cma_args.pc1_chunk_size} "
    plot_2d(plot_dir, angles_plot_name, np.arange(len(duped_angles_along_the_way)), duped_angles_along_the_way, "num of chunks", "angle with diff in degrees", False)

    angles_plot_name = f"unduped exponential 2, num dup: {num}" \
                       f"cma_args.pc1_chunk_size: {cma_args.pc1_chunk_size} "
    plot_2d(plot_dir, angles_plot_name, np.arange(len(unduped_angles_along_the_way)), unduped_angles_along_the_way, "num of chunks", "angle with diff in degrees", False)


    angles_plot_name = f"undup - dup diff_along exponential 2, num dup: {num}" \
                       f"cma_args.pc1_chunk_size: {cma_args.pc1_chunk_size} "
    plot_2d(plot_dir, angles_plot_name, np.arange(len(diff_along)), diff_along, "num of chunks", "angle with diff in degrees", False)


    del all_matrix_buffer
    import gc
    gc.collect()

if __name__ == '__main__':

    main()

#TODO Give filenames more info to identify which hyperparameter is the data for

