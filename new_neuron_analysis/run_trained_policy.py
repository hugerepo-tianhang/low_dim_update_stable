from low_dim_update_stable.new_neuron_analysis.dir_tree_util import *
import pandas as pd
from sklearn import linear_model

from stable_baselines.common.vec_env.vec_video_recorder import VecVideoRecorder

from stable_baselines.low_dim_analysis.common_parser import get_common_parser
import numpy as np
import gym

from stable_baselines import bench, logger
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.low_dim_analysis.eval_util import get_full_param_traj_file_path, get_full_params_dir, get_dir_path_for_this_run, get_log_dir, get_save_dir
import shutil
import os

from matplotlib import pyplot as plt
import sys

import minepy
import pickle
from matplotlib import pyplot as plt
from stable_baselines.common import set_global_seeds

lagrangian_keys = ["M", "q", "dq", "COM", "Coriolis", "total_contact_forces_contact_bodynode", "com_jacobian", "contact_bodynode_jacobian"]


def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return nan. It is used for logging only.

    :param arr: (np.ndarray)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)



# def fill_contacts_jac_dict(contacts, contact_dict, neuron_values):
#     for contact in contacts:
#         J_contact = contact.bodynode1.linear_jacobian(contact.bodynode1.to_local(contact.point))
#         if contact.bodynode1.name in contact_dict:
#             contact_dict[contact.bodynode1.name][contact.bodynode1.name].append(J_contact.reshape((-1, 1)))
#             for i, layer in enumerate(neuron_values[1:-2]):
#                 contact_dict[contact.bodynode1.name][i].append(layer.reshape((-1, 1)))
#         else:
#             contact_dict[contact.bodynode1.name] = {}
#             contact_dict[contact.bodynode1.name][contact.bodynode1.name] = [J_contact.reshape((-1, 1))]
#             for i, layer in enumerate(neuron_values[1:-2]):
#                 contact_dict[contact.bodynode1.name][i] = [layer.reshape((-1, 1))]



def compute_alpha(npoints):
    NPOINTS_BINS = [1, 25, 50, 250, 500, 1000, 2500, 5000, 10000, 40000]
    ALPHAS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4]

    if npoints < 1:
        raise ValueError("the number of points must be >=1")

    return ALPHAS[np.digitize([npoints], NPOINTS_BINS)[0] - 1]


regr = linear_model.LinearRegression()
def get_normalized_SSE(lagrange_l, neuron_l, regr):

    range_neuron = max(neuron_l) - min(neuron_l)
    neuron_l = neuron_l/range_neuron
    X = lagrange_l.reshape(-1,1)
    y = neuron_l.reshape(-1,1)
    regr.fit(X, y)
    y_pred = regr.predict(X)
    SSE = ((y - y_pred) ** 2).sum()
    TV = ((y - np.average(y)) ** 2).sum()
    return SSE,TV

def get_Reflective_correlation_coefficient(lagrange_l, neuron_l):

    prod = np.dot(lagrange_l, neuron_l)
    A = np.sum(lagrange_l**2)
    B = np.sum(neuron_l**2)
    return prod/np.sqrt(A*B)

def scatter_the_nonlinear_significant_but_not_linear_ones(lagrangian_values,
                                                          layer_values_list, linear_threshold, nonlinear_threshold, out_dir):
    for key, nda in lagrangian_values.items():
        for ind, lagrange_l in enumerate(nda):
            for layer_ind, layer in enumerate(layer_values_list):
                for neuron_ind, neuron_l in enumerate(layer):
                    linear_co = np.corrcoef(lagrange_l, neuron_l)[1, 0]
                    alpha_cl = compute_alpha(lagrange_l.shape[0])
                    mine = minepy.MINE(alpha=alpha_cl, c=5, est="mic_e")
                    mine.compute_score(lagrange_l, neuron_l)
                    mic = mine.mic()

                    if abs(linear_co) < linear_threshold and mic > nonlinear_threshold:
                        name = f"{out_dir}/{key}_index_{ind}_VS_layer_{layer_ind}_neuron_{neuron_ind}_" \
                               f"linear_correlation{linear_co}_nonlinear_correlation{mic}.jpg"
                        plt.figure()
                        plt.scatter(lagrange_l, neuron_l)
                        plt.xlabel("lagrange")
                        plt.ylabel("neuron")

                        plt.savefig(name)
                        plt.close()



def scatter_the_linear_significant_ones(lagrangian_values, layer_values_list, threshold, out_dir):
    for key, nda in lagrangian_values.items():
        for ind, lagrange_l in enumerate(nda):
            for layer_ind, layer in enumerate(layer_values_list):
                for neuron_ind, neuron_l in enumerate(layer):
                    co = np.corrcoef(lagrange_l, neuron_l)[1,0]
                    normalized_SSE, TV = get_normalized_SSE(lagrange_l, neuron_l, regr)
                    Reflective_correlation_coefficient = get_Reflective_correlation_coefficient(lagrange_l, neuron_l)
                    # if abs(co) > threshold:
                    if (abs(co) > threshold and normalized_SSE < 200):
                        name = f"{out_dir}/{key}_index_{ind}_VS_layer_{layer_ind}_neuron_{neuron_ind}_" \
                               f"linear_correlation{co}_normalized_SSE_{normalized_SSE}_Syy_{TV}" \
                               f"_Reflective_correlation_coefficient_{Reflective_correlation_coefficient}.jpg"
                        plt.figure()

                        plt.scatter(lagrange_l,neuron_l)
                        plt.xlabel("lagrange")
                        plt.ylabel("neuron")

                        plt.savefig(name)
                        plt.close()


def plot_everything(lagrangian_values, layer_values_list, out_dir, PLOT_CUTOFF):
    for key, nda in lagrangian_values.items():
        for ind, l in enumerate(nda):
            name = f"{out_dir}/{key}_index_{ind}.jpg"
            plt.figure()

            plt.plot(l[:PLOT_CUTOFF])
            plt.savefig(name)
            plt.close()
    for layer_ind, layer in enumerate(layer_values_list):
        for neuron_ind, l in enumerate(layer):
            name = f"{out_dir}/layer_{layer_ind}_neuron_{neuron_ind}.jpg"
            plt.figure()

            plt.plot(l[:PLOT_CUTOFF])
            plt.savefig(name)
            plt.close()







def eval_trained_policy_and_collect_data(eval_seed, eval_run_num, policy_env, policy_num_timesteps, policy_seed, policy_run_num, additional_note):


    logger.log(sys.argv)
    common_arg_parser = get_common_parser()
    args, cma_unknown_args = common_arg_parser.parse_known_args()
    args.env = policy_env
    args.seed = policy_seed
    args.num_timesteps = policy_num_timesteps
    args.run_num = policy_run_num
    this_run_dir = get_dir_path_for_this_run(args)
    traj_params_dir_name = get_full_params_dir(this_run_dir)
    save_dir = get_save_dir( this_run_dir)



    final_file = get_full_param_traj_file_path(traj_params_dir_name, "pi_final")
    final_params = pd.read_csv(final_file, header=None).values[0]


    def make_env():
        env_out = gym.make(args.env)
        env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
        env_out.seed(eval_seed)
        return env_out
    env = DummyVecEnv([make_env])
    running_env = env.envs[0].env.env


    set_global_seeds(eval_seed)
    running_env.seed(eval_seed)

    if args.normalize:
        env = VecNormalize(env)

    model = PPO2.load(f"{save_dir}/ppo2", seed=eval_seed)
    model.set_pi_from_flat(final_params)
    if args.normalize:
        env.load_running_average(save_dir)

    # is it necessary?
    running_env = env.venv.envs[0].env.env


    lagrangian_values = {}

    obs = np.zeros((env.num_envs,) + env.observation_space.shape)

    obs[:] = env.reset()

    # env = VecVideoRecorder(env, "./",
    #                            record_video_trigger=lambda x: x == 0, video_length=3000,
    #                            name_prefix="3000000agent-{}".format(args.env))

    #init lagrangian values
    for lagrangian_key in lagrangian_keys:
        flat_array = running_env.get_lagrangian_flat_array(lagrangian_key)
        lagrangian_values[lagrangian_key] = [flat_array]


    neuron_values = model.give_neuron_values(obs)
    raw_layer_values_list = [[neuron_value.reshape((-1,1))] for neuron_value in neuron_values]

    # env.render()
    ep_infos = []
    steps_to_first_done = 0
    first_done = False
    for _ in range(30000):
        actions = model.step(obs)[0]

        # yield neuron_values
        obs, rew, done, infos = env.step(actions)
        if done and not first_done:
            first_done = True

        if not first_done:
            steps_to_first_done += 1


        neuron_values = model.give_neuron_values(obs)


        for i, layer in enumerate(neuron_values):
            raw_layer_values_list[i].append(layer.reshape((-1,1)))

        # fill_contacts_jac_dict(infos[0]["contacts"], contact_dict=contact_values, neuron_values=neuron_values)

        # filling lagrangian values
        for lagrangian_key in lagrangian_keys:
            flat_array = running_env.get_lagrangian_flat_array(lagrangian_key)
            lagrangian_values[lagrangian_key].append(flat_array)

        # env.render()

        # time.sleep(1)
        for info in infos:
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                ep_infos.append(maybe_ep_info)

        # env.render()
        done = done.any()
        if done:
            episode_rew = safe_mean([ep_info['r'] for ep_info in ep_infos])
            print(f'episode_rew={episode_rew}')
            obs = env.reset()


    #Hstack into a big matrix
    for lagrangian_key in lagrangian_keys:
        lagrangian_values[lagrangian_key] = np.hstack(lagrangian_values[lagrangian_key])

    # for contact_body_name, l in contact_values.items():
    #     body_contact_dict = contact_values[contact_body_name]
    #     for name, l in body_contact_dict.items():
    #         body_contact_dict[name] = np.hstack(body_contact_dict[name])
    input_values = np.hstack(raw_layer_values_list[0])

    layers_values = [np.hstack(layer_list) for layer_list in raw_layer_values_list][1:-2]# drop variance and inputs


    data_dir = get_data_dir(policy_env=args.env, policy_num_timesteps=policy_num_timesteps, policy_run_num=policy_run_num
                            , policy_seed=policy_seed, eval_seed=eval_seed, eval_run_num=eval_run_num, additional_note=additional_note)
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)


    lagrangian_values_fn = f"{data_dir}/lagrangian.pickle"

    with open(lagrangian_values_fn, 'wb') as handle:
        pickle.dump(lagrangian_values, handle, protocol=pickle.HIGHEST_PROTOCOL)

    input_values_fn = f"{data_dir}/input_values.npy"
    layers_values_fn = f"{data_dir}/layer_values.npy"

    np.save(input_values_fn, input_values)
    np.save(layers_values_fn, layers_values)


    all_weights = model.get_all_weight_values()

    for ind, weights in enumerate(all_weights):
        fname = f"{data_dir}/weights_layer_{ind}.txt"
        np.savetxt(fname, weights)



if __name__ == '__main__':
    # visualize_policy_and_collect_COM(seed=0, run_num=0, policy_num_timesteps=3000000, policy_run_num=0, policy_seed=0)

    policy_env = "DartWalker2d-v1"

    eval_trained_policy_and_collect_data(eval_seed=4, eval_run_num=4, policy_env=policy_env,
                                         policy_num_timesteps=5000000, policy_seed=3,
                                         policy_run_num=1, additional_note="sandbox")
    # visualize_policy_and_collect_COM(seed=3, run_num=3, policy_env=policy_env, policy_num_timesteps=2000000,
    #                              policy_seed=1, policy_run_num=0)
# seeds = [0, 1, 2]
    # run_nums = [0, 1, 2]
    # for seed in seeds:
    #     for run_num in run_nums:
    #         run_trained_policy(seed=seed, run_num=run_num)
    #
    # visualize_trained_policy(seed=3, run_num, policy_env, policy_num_timesteps, policy_seed, policy_run_num)
#TODO Give filenames more info to identify which hyperparameter is the data for

