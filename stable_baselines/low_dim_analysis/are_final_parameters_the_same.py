from stable_baselines.ppo2.run_mujoco import eval_return
import cma

import numpy as np
from stable_baselines.low_dim_analysis.eval_util import *
from stable_baselines.low_dim_analysis.common import *

from stable_baselines import logger

import pandas as pd
from sklearn.decomposition import PCA

from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import time
import os
from stable_baselines.common.cmd_util import mujoco_arg_parser
from stable_baselines.low_dim_analysis.common_parser import get_common_parser
from numpy import linalg as LA
import numpy as np
import gym

from stable_baselines.common.cmd_util import mujoco_arg_parser
from stable_baselines import bench, logger
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import csv
import os
from stable_baselines.low_dim_analysis.eval_util import get_full_param_traj_file_path, get_full_params_dir, get_dir_path_for_this_run, get_log_dir, get_save_dir
import itertools



def main():


    import sys
    logger.log(sys.argv)
    common_arg_parser = get_common_parser()
    cma_args, cma_unknown_args = common_arg_parser.parse_known_args()
    run_nums = cma_args.run_nums_to_check
    run_nums = [int(run_num) for run_num in run_nums.split(":")]


    final_params_list = []
    start_params_list = []

    for run_num in run_nums:
        cma_args.run_num = run_num
        if os.path.exists(get_dir_path_for_this_run(cma_args)):

            this_run_dir = get_dir_path_for_this_run(cma_args)
            plot_dir_alg = get_plot_dir(cma_args)

            traj_params_dir_name = get_full_params_dir(this_run_dir)
            intermediate_data_dir = get_intermediate_data_dir(this_run_dir, params_scope="pi")
            save_dir = get_save_dir( this_run_dir)


            if not os.path.exists(intermediate_data_dir):
                os.makedirs(intermediate_data_dir)
            if not os.path.exists(plot_dir_alg):
                os.makedirs(plot_dir_alg)

            start_file = get_full_param_traj_file_path(traj_params_dir_name, "pi_start")
            start_params = pd.read_csv(start_file, header=None).values[0]

            final_file = get_full_param_traj_file_path(traj_params_dir_name, "pi_final")
            final_params = pd.read_csv(final_file, header=None).values[0]

            final_params_list.append(final_params)
            start_params_list.append(start_params)

            cma_args.run_num += 1

    final_params_distances = []
    for i in range(len(final_params_list)):
        for j in range(i+1, len(final_params_list)):
            final_params_distances.append(LA.norm(final_params_list[i] - final_params_list[j], ord=2))


    plot_dir = get_plot_dir(cma_args)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    np.savetxt(f"{plot_dir}/final_params_distances.txt", final_params_distances, delimiter=",")

if __name__ == '__main__':

    main()

#TODO Give filenames more info to identify which hyperparameter is the data for

