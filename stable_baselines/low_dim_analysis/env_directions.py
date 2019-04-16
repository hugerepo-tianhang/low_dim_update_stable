'''
1.pick two random bad components cma contour plot. ppo


1. is it true that if you follow pc1 you always get better. Just plot Y = start + pc1 * N * alpha. That means that there
exist a general direction. this direction means make some feature more important and some feature less important will
always make the agent perform better. no
2. what is the explained raiios of pcs along the way. is the pc1 evolving in 2d or what? yes
3. later part pc1 vs final - start, is everything later already identify pc1?
4. weight later part heavier make it drag angle down further.

5. #TODO it has to be incrementally weighted
6. plot Y = grad vs V - (current - start)
7. plot Y = grad vs PC1 - (current - start)
8. Plot T = current param - start vs pc1

1. adjust weight first n samples to better approx final - start
1. weighted by varaince explained is bad idea


1. Subspace relay network. intuition is that training might goes in stages(is it??????) like hopper, you might put your
com lean forward first and then work on stablizing.

1. is the direction of biggest axis roughly same direction for each env?

2. what's the cos(pc1_last_5000, pc1_final_5000)
2. what's the cos(pc1_first_n, pc1_final)

2. what's the smallest N such that if you follow this hyperplane for 5000 steps, it's 99% explained
'''
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


def main(origin="final_param"):


    import sys
    logger.log(sys.argv)
    common_arg_parser = get_common_parser()
    cma_args, cma_unknown_args = common_arg_parser.parse_known_args()


    cma_args = {
        "alg": 'ppo2',
        "env": "DartHopper-v1",
        "num_timesteps": 5000,
        "normalize": True,
        "n_steps": 2048,
        "nminibatches": 32,
        "run_num": 0
    }
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
    from stable_baselines.low_dim_analysis.common import do_pca
    result = do_pca(cma_args.n_components, cma_args.n_comp_to_use, traj_params_dir_name, intermediate_data_dir, proj=True,
                    origin=origin, use_IPCA=cma_args.use_IPCA, chunk_size=cma_args.chunk_size)


