
from stable_baselines.low_dim_analysis.common import *
import tensorflow as tf

import matplotlib.colors
import pandas as pd

from matplotlib import pyplot as plt

from stable_baselines.low_dim_analysis.common_parser import get_common_parser
import numpy as np
import gym

from stable_baselines import bench, logger
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import os
from stable_baselines.low_dim_analysis.eval_util import get_full_param_traj_file_path, get_full_params_dir, get_dir_path_for_this_run, get_log_dir, get_save_dir




def main():


    import sys
    logger.log(sys.argv)
    common_arg_parser = get_common_parser()
    args, cma_unknown_args = common_arg_parser.parse_known_args()

    this_run_dir = get_dir_path_for_this_run(args)
    plot_dir_alg = get_plot_dir(args)

    traj_params_dir_name = get_full_params_dir(this_run_dir)
    intermediate_data_dir = get_intermediate_data_dir(this_run_dir, params_scope="pi")
    save_dir = get_save_dir( this_run_dir)


    if not os.path.exists(intermediate_data_dir):
        os.makedirs(intermediate_data_dir)
    if not os.path.exists(plot_dir_alg):
        os.makedirs(plot_dir_alg)


    final_file = get_full_param_traj_file_path(traj_params_dir_name, "pi_final")
    final_params = pd.read_csv(final_file, header=None).values[0]


    def make_env():
        env_out = gym.make(args.env)

        env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
        return env_out
    env = DummyVecEnv([make_env])

    if args.normalize:
        env = VecNormalize(env)

    model = PPO2.load(f"{save_dir}/ppo2") # this also loads V function
    model.set_pi_from_flat(final_params)

    if args.normalize:
        env.load_running_average(save_dir)

    obz_tensor = model.act_model.fake_input_tensor

    some_neuron = model.act_model.policy_neurons[2][-1]

    grads = tf.gradients(tf.math.negative(some_neuron), obz_tensor)

    grads = list(zip(grads, obz_tensor))

    trainer = tf.train.AdamOptimizer(learning_rate=0.01, epsilon=1e-5)

    train_op = trainer.apply_gradients(grads)
    for i in range(10000):
        obz, _ = model.sess.run([obz_tensor, train_op])

if __name__ == "__main__":
    main()