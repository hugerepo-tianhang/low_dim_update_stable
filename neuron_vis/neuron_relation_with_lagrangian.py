
from stable_baselines.low_dim_analysis.common import *


import matplotlib.colors
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from matplotlib import pyplot as plt

from stable_baselines.low_dim_analysis.common_parser import get_common_parser
import numpy as np
import gym
from stable_baselines.common.policies import MlpPolicy

from stable_baselines import bench, logger
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import os
from stable_baselines.low_dim_analysis.eval_util import get_full_param_traj_file_path, get_full_params_dir, get_dir_path_for_this_run, get_log_dir, get_save_dir
import shutil

import minepy

def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return nan. It is used for logging only.

    :param arr: (np.ndarray)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)


def neuron_values_generator(args, save_dir, pi_theta, eval_timesteps):
    # logger.log(f"#######EVAL: {args}")


    neuron_values_list = []
    def make_env():
        env_out = gym.make(args.env)

        env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
        return env_out
    env = DummyVecEnv([make_env])

    if args.normalize:
        env = VecNormalize(env)


    # policy = MlpPolicy
    # # model = PPO2.load(f"{save_dir}/ppo2") # this also loads V function
    # model = PPO2(policy=policy, env=env, n_steps=args.n_steps, nminibatches=args.nminibatches, lam=0.95, gamma=0.99, noptepochs=10,
    #              ent_coef=0.0, learning_rate=3e-4, cliprange=0.2, optimizer=args.optimizer)
    model = PPO2.load(f"{save_dir}/ppo2") # this also loads V function
    if pi_theta is not None:
        model.set_pi_from_flat(pi_theta)

    if args.normalize:
        env.load_running_average(save_dir)



    obs = np.zeros((env.num_envs,) + env.observation_space.shape)
    obs[:] = env.reset()
    env.render()
    ep_infos = []
    while 1:
        neuron_values, actions, _, _, _ = model.step_with_neurons(obs)
        # neuron_values = model.give_neuron_values(obs)

        # neuron_values_list.append( neuron_values )
        yield neuron_values
        obs, rew, done, infos = env.step(actions)
        env.render()

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
    # return neuron_values_list
    # return safe_mean([ep_info['r'] for ep_info in ep_infos])

def preload_neurons(args, save_dir, pi_theta, eval_timesteps):
    # logger.log(f"#######EVAL: {args}")


    neuron_values_list = []
    def make_env():
        env_out = gym.make(args.env)

        env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
        return env_out
    env = DummyVecEnv([make_env])

    if args.normalize:
        env = VecNormalize(env)
    # policy = MlpPolicy
    model = PPO2.load(f"{save_dir}/ppo2") # this also loads V function
    # model = PPO2(policy=policy, env=env, n_steps=args.n_steps, nminibatches=args.nminibatches, lam=0.95, gamma=0.99, noptepochs=10,
    #              ent_coef=0.0, learning_rate=3e-4, cliprange=0.2, optimizer=args.optimizer)
    if pi_theta is not None:
        model.set_pi_from_flat(pi_theta)

    if args.normalize:
        env.load_running_average(save_dir)



    obs = np.zeros((env.num_envs,) + env.observation_space.shape)
    obs[:] = env.reset()
    env.render()
    ep_infos = []
    for _ in range(1024):
        actions = model.step(obs)[0]
        neuron_values = model.give_neuron_values(obs)

        neuron_values_list.append( neuron_values )
        # yield neuron_values
        obs, rew, done, infos = env.step(actions)
        env.render()

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
    return neuron_values_list
    # return safe_mean([ep_info['r'] for ep_info in ep_infos])

def give_normalizers(preload_neuron_values_list):


    def process_values(neuron_values):
        obz = neuron_values[0]

        dist = neuron_values[-2:]
        dist = np.concatenate(dist, axis=1)

        return [np.max(obz), np.min(obz), np.max(dist), np.min(dist)]

    result = np.array([process_values(neuron_values) for neuron_values in preload_neuron_values_list])

    obz_max, obz_min, dist_max, dist_min = np.max(result[:,0]),\
                                                 np.min(result[:,1]), \
                                                 np.max(result[:,2]), \
                                                 np.min(result[:,3])

    #since it's tanh, HARDCODE -1, 1
    latent_norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    obz_norm = matplotlib.colors.Normalize(vmin=obz_min, vmax=obz_max)


    dist_norm = matplotlib.colors.Normalize(vmin=dist_min, vmax=dist_max)
    return obz_norm, latent_norm, dist_norm

def get_neuron_values_matrix(preload):
    result = preload[0]
    for neuron_values in preload[1:]:
        for i, layer_neurons in enumerate(neuron_values):
            result[i] = np.hstack(result[i], layer_neurons)

    result = [re.T for re in result]
    return result

def get_correlations(lagrangian_values, layer_values_list):
    result = []
    for lag_v in lagrangian_values:
        result_v = []
        for layer in layer_values_list:
            layer_result = []
            for v in layer:
                co = np.corrcoef(lag_v, v)[1,0]
                if np.isnan(co):
                    co = 0
                layer_result.append(co)
            result_v.append(layer_result)
        result.append(result_v)
    return np.array(result)


def get_normalized_correlations(lagrangian_values, layer_values_list):
    result = []
    for lag_v in lagrangian_values:
        result_v = []
        scale_lag_v = np.max(lag_v) - np.min(lag_v)
        lag_v = lag_v/scale_lag_v

        for layer in layer_values_list:
            layer_result = []
            for v in layer:

                scale_v = np.max(v) - np.min(v)
                v = v / scale_v
                co = np.corrcoef(lag_v, v)[1,0]
                if np.isnan(co):
                    co = 0
                layer_result.append(co)
            result_v.append(layer_result)
        result.append(result_v)
    return np.array(result)


from itertools import permutations

def get_linear_regressions_2_perm(lagrangian_values, layer_values_list):
    result = []
    regr = linear_model.LinearRegression()

    for lag_v in lagrangian_values:
        if np.allclose(lag_v, np.zeros(lag_v.shape)):
            result_v = []
        else:
            result_v = []
            concat = np.concatenate(layer_values_list)

            perm_2 = permutations(range(len(concat)), 2)
            for i, j in list(perm_2):

                X = concat[(i,j),:].T
                # Train the model using the training sets
                regr.fit(X, lag_v)

                # Make predictions using the testing set
                y_pred = regr.predict(X)
                ab_error = np.abs(y_pred - lag_v)
                mean_ab_error_percent = np.mean(np.abs(np.divide(ab_error, lag_v)))
                result_v.append(np.concatenate(([mean_ab_error_percent, np.mean(ab_error), i,j ], regr.coef_)))

            result_v = np.vstack(result_v)
        result.append(result_v)
    return np.array(result)


def get_linear_regressions_1_perm(lagrangian_values, layer_values_list):
    result = []
    regr = linear_model.LinearRegression()

    for lag_v in lagrangian_values:
        # if np.allclose(lag_v, np.zeros(lag_v.shape), atol=1e-5):
        #     result_v = []
        # else:
        result_v = []
        concat = np.concatenate(layer_values_list)

        for i in range(len(concat)):

            X = concat[i,:].reshape(-1,1)
            y = lag_v.reshape(-1,1)
            # Train the model using the training sets
            regr.fit(X, y)

            # Make predictions using the testing set
            y_pred = regr.predict(X)
            ab_error = np.abs(y_pred - y)
            mean_ab_error_percent = np.mean(np.abs(np.divide(ab_error, y)))
            result_v.append(np.concatenate(([mean_ab_error_percent,  np.mean(ab_error), i], regr.coef_.reshape(-1))))

        result_v = np.vstack(result_v)
        result.append(result_v)
    return np.array(result)


#TODO Not well trained doesn't have correlation? YES?
# COM vs Q q(0) == com(x) q(1) correlate to com(x) ow no,
# COM VS M. NO.
# distribution correlation of rv, MIC*
# big weight from strong variable to strong variable, not much so doesn't first layer correlation
# why is the (5,5) constant
#


#TODO: what is q(3) since q(3) has a neuron firing with it?
# Q vs M plot, I think M
def get_results(name, lagrangian_values, layer_values_list, perm_num):

    if perm_num == 1:
        lin_reg = get_linear_regressions_1_perm(lagrangian_values[name], layer_values_list)
    else:
        lin_reg = get_linear_regressions_2_perm(lagrangian_values[name], layer_values_list)

    best_lin_reg = []
    for lin_l in lin_reg:
        if lin_l == []:
            best_lin_reg.append([])
        else:
            best_lin_reg.append(lin_l[np.argmin(lin_l[:, 0])])

    best_lin_reg = np.array(best_lin_reg)

    logger.info(f"dumping {perm_num} and {name}")
    lin_reg.dump(f"lin_reg_{perm_num}_{name}.txt")
    best_lin_reg.dump(f"best_lin_reg_{perm_num}_{name}.txt")
    return lin_reg, best_lin_reg

def fill_contacts_jac_dict(contacts, contact_dict, neuron_values):
    for contact in contacts:
        J_contact = contact.bodynode1.linear_jacobian(contact.bodynode1.to_local(contact.point))
        if contact.bodynode1.name in contact_dict:
            contact_dict[contact.bodynode1.name][contact.bodynode1.name].append(J_contact.reshape((-1, 1)))
            for i, layer in enumerate(neuron_values[1:-2]):
                contact_dict[contact.bodynode1.name][i].append(layer.reshape((-1, 1)))
        else:
            contact_dict[contact.bodynode1.name] = {}
            contact_dict[contact.bodynode1.name][contact.bodynode1.name] = [J_contact.reshape((-1, 1))]
            for i, layer in enumerate(neuron_values[1:-2]):
                contact_dict[contact.bodynode1.name][i] = [layer.reshape((-1, 1))]

from matplotlib import pyplot as plt


def compute_alpha(npoints):
    NPOINTS_BINS = [1, 25, 50, 250, 500, 1000, 2500, 5000, 10000, 40000]
    ALPHAS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4]

    if npoints < 1:
        raise ValueError("the number of points must be >=1")

    return ALPHAS[np.digitize([npoints], NPOINTS_BINS)[0] - 1]

PLOT_CUTOFF = 300

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
                    if abs(co) > threshold:
                        name = f"{out_dir}/{key}_index_{ind}_VS_layer_{layer_ind}_neuron_{neuron_ind}_" \
                               f"linear_correlation{co}.jpg"
                        plt.figure()

                        plt.scatter(lagrange_l,neuron_l)
                        plt.xlabel("lagrange")
                        plt.ylabel("neuron")

                        plt.savefig(name)
                        plt.close()


def plot_everything(lagrangian_values, layer_values_list, out_dir):
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


#TODO make sure that there is a linear relationship. plot them out
# what is the nonlinear relationship that's not linear. plot them out
# X is sin, Y is sin with phase shift. Do they have functional relation?
# does MIC capture relationship of 2 sin wave?


#TODO Mddq + C = t M is general for all motor skill(task independent),
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


    layer_values_list = []
    def make_env():
        env_out = gym.make(args.env)

        env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
        return env_out
    env = DummyVecEnv([make_env])

    if args.normalize:
        env = VecNormalize(env)
    # policy = MlpPolicy
    model = PPO2.load(f"{save_dir}/ppo2") # this also loads V function
    # model = PPO2(policy=policy, env=env, n_steps=args.n_steps, nminibatches=args.nminibatches, lam=0.95, gamma=0.99, noptepochs=10,
    #              ent_coef=0.0, learning_rate=3e-4, cliprange=0.2, optimizer=args.optimizer)
    model.set_pi_from_flat(final_params)

    if args.normalize:
        env.load_running_average(save_dir)

    sk = env.venv.envs[0].env.env.robot_skeleton
    lagrangian_values = {}

    obs = np.zeros((env.num_envs,) + env.observation_space.shape)
    obs[:] = env.reset()

    lagrangian_values["M"] = [sk.M.reshape((-1,1))]
    lagrangian_values["COM"] = [sk.C.reshape((-1,1))]
    lagrangian_values["Coriolis"] = [sk.c.reshape((-1,1))]
    lagrangian_values["q"] = [sk.q.reshape((-1, 1))]
    lagrangian_values["dq"] = [sk.dq.reshape((-1, 1))]


    contact_values = {}


    neuron_values = model.give_neuron_values(obs)
    layer_values_list = [[neuron_value.reshape((-1,1))] for neuron_value in neuron_values]

    env.render()
    ep_infos = []
    for _ in range(3000):
        actions = model.step(obs)[0]

        # yield neuron_values
        obs, rew, done, infos = env.step(actions)

        neuron_values = model.give_neuron_values(obs)


        for i, layer in enumerate(neuron_values):
            layer_values_list[i].append(layer.reshape((-1,1)))

        fill_contacts_jac_dict(infos[0]["contacts"], contact_dict=contact_values, neuron_values=neuron_values)



        lagrangian_values["M"].append(sk.M.reshape((-1, 1)))
        lagrangian_values["q"].append(sk.q.reshape((-1, 1)))
        lagrangian_values["dq"].append(sk.dq.reshape((-1, 1)))
        lagrangian_values["COM"].append(sk.C.reshape((-1, 1)))
        lagrangian_values["Coriolis"].append(sk.c.reshape((-1, 1)))

        env.render()

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
    lagrangian_values["M"] = np.hstack(lagrangian_values["M"])
    lagrangian_values["COM"] = np.hstack(lagrangian_values["COM"])
    lagrangian_values["Coriolis"] = np.hstack(lagrangian_values["Coriolis"])
    lagrangian_values["q"] = np.hstack(lagrangian_values["q"])
    lagrangian_values["dq"] = np.hstack(lagrangian_values["dq"])

    for contact_body_name, l in contact_values.items():
        body_contact_dict = contact_values[contact_body_name]
        for name, l in body_contact_dict.items():
            body_contact_dict[name] = np.hstack(body_contact_dict[name])

    layer_values_list = [np.hstack(layer_list) for layer_list in layer_values_list][1:-2]# drop variance



    # plt.scatter(lagrangian_values["M"][15], layer_values_list[1][2])
    # plt.scatter(lagrangian_values["M"][11], layer_values_list[0][63])
    out_dir = f"/home/panda-linux/PycharmProjects/low_dim_update_dart/low_dim_update_stable/neuron_vis/plots_{args.env}_{args.num_timesteps}"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)


    plot_everything(lagrangian_values, layer_values_list, out_dir)
    scatter_the_linear_significant_ones(lagrangian_values, layer_values_list, threshold=0.6, out_dir=out_dir)
    scatter_the_nonlinear_significant_but_not_linear_ones(lagrangian_values,
                                                          layer_values_list,
                                                          linear_threshold=0.3,
                                                          nonlinear_threshold=0.6, out_dir=out_dir)
    #
    # contact_dicts = {}
    # for contact_body_name, l in contact_values.items():
    #     body_contact_dict = contact_values[contact_body_name]
    #
    #
    #     contact_dicts[contact_body_name] = {}
    #
    #     build_dict = contact_dicts[contact_body_name]
    #
    #     build_dict["body"] = {}
    #     build_dict["layer"] = {}
    #     for name, l in body_contact_dict.items():
    #         for i in range(len(l)):
    #
    #             if name == contact_body_name:
    #                 build_dict["body"][f"{contact_body_name}_{i}"] = l[i]
    #             else:
    #                 build_dict["layer"][f"layer_{name}_neuron_{i}"] = l[i]
    #
    #     body_contact_df = pd.DataFrame.from_dict(build_dict["body"], "index")
    #     layer_contact_df = pd.DataFrame.from_dict(build_dict["layer"], "index")

        # body_contact_df.to_csv(f"{data_dir}/{contact_body_name}_contact.txt", sep='\t')
        # layer_contact_df.to_csv(f"{data_dir}/{contact_body_name}_layers.txt", sep='\t')




    # #TO CSV format
    # data_dir = f"/home/panda-linux/PycharmProjects/low_dim_update_dart/mictools/examples/neuron_vis_data{args.env}_time_steps_{args.num_timesteps}"
    # if os.path.exists(data_dir):
    #     shutil.rmtree(data_dir)
    #
    # os.makedirs(data_dir)
    #
    # for contact_body_name, d in contact_dicts.items():
    #
    #     build_dict = d
    #
    #     body_contact_df = pd.DataFrame.from_dict(build_dict["body"], "index")
    #     layer_contact_df = pd.DataFrame.from_dict(build_dict["layer"], "index")
    #
    #     body_contact_df.to_csv(f"{data_dir}/{contact_body_name}_contact.txt", sep='\t')
    #     layer_contact_df.to_csv(f"{data_dir}/{contact_body_name}_layers.txt", sep='\t')
    #
    #
    #
    # neurons_dict = {}
    # for layer_index in range(len(layer_values_list)):
    #     for neuron_index in range(len(layer_values_list[layer_index])):
    #         neurons_dict[f"layer_{layer_index}_neuron_{neuron_index}"] = layer_values_list[layer_index][neuron_index]
    #
    # for i in range(len(lagrangian_values["COM"])):
    #     neurons_dict[f"COM_index_{i}"] = lagrangian_values["COM"][i]
    #
    # neuron_df = pd.DataFrame.from_dict(neurons_dict, "index")
    #
    #
    #
    # lagrangian_dict = {}
    # for k,v in lagrangian_values.items():
    #     for i in range(len(v)):
    #         lagrangian_dict[f"{k}_index_{i}"] = v[i]
    #
    # lagrangian_df = pd.DataFrame.from_dict(lagrangian_dict, "index")
    #
    #
    # neuron_df.to_csv(f"{data_dir}/neurons.txt", sep='\t')
    # lagrangian_df.to_csv(f"{data_dir}/lagrangian.txt", sep='\t')






    # cor = {}
    # best_cor = {}
    # cor["M"] = get_correlations(lagrangian_values["M"], layer_values_list)
    # best_cor["M"] = [np.max(np.abs(cor_m)) for cor_m in cor["M"]]
    #
    #
    # cor["COM"] = get_correlations(lagrangian_values["COM"], layer_values_list)
    # best_cor["COM"] = [np.max(np.abs(cor_m)) for cor_m in cor["COM"]]
    #
    # cor["Coriolis"] = get_correlations(lagrangian_values["Coriolis"], layer_values_list)
    # best_cor["Coriolis"] = [np.max(np.abs(cor_m)) for cor_m in cor["Coriolis"]]
    # best_cor["Coriolis_argmax"] = [np.argmax(np.abs(cor_m)) for cor_m in cor["Coriolis"]]
    #
    #
    #
    #
    # ncor = {}
    # nbest_cor = {}
    # ncor["M"] = get_normalized_correlations(lagrangian_values["M"], layer_values_list)
    # nbest_cor["M"] = [np.max(np.abs(cor_m)) for cor_m in ncor["M"]]
    #
    #
    # ncor["COM"] = get_normalized_correlations(lagrangian_values["COM"], layer_values_list)
    # nbest_cor["COM"] = [np.max(np.abs(cor_m)) for cor_m in ncor["COM"]]
    #
    # ncor["Coriolis"] = get_normalized_correlations(lagrangian_values["Coriolis"], layer_values_list)
    # nbest_cor["Coriolis"] = [np.max(np.abs(cor_m)) for cor_m in ncor["Coriolis"]]
    # nbest_cor["Coriolis_argmax"] = [np.argmax(np.abs(cor_m)) for cor_m in ncor["Coriolis"]]
    #
    #
    #
    #
    #
    # lin_reg = {"perm_1":{}, "perm_2":{}}
    # best_lin_reg = {"perm_1":{}, "perm_2":{}}
    # lin_reg["perm_1"]["M"], best_lin_reg["perm_1"]["M"] = get_results("M", lagrangian_values, layer_values_list, perm_num=1)
    # lin_reg["perm_2"]["M"], best_lin_reg["perm_2"]["M"] = get_results("M", lagrangian_values, layer_values_list, perm_num=2)
    # lin_reg["perm_1"]["COM"], best_lin_reg["perm_1"]["COM"] = get_results("COM", lagrangian_values, layer_values_list, perm_num=1)
    # lin_reg["perm_2"]["COM"], best_lin_reg["perm_2"]["COM"] = get_results("COM", lagrangian_values, layer_values_list, perm_num=2)

    #
    #
    # lin_reg_1["M"] = get_linear_regressions_1_perm(lagrangian_values["M"], layer_values_list)
    # lin_reg_2["M"] = get_linear_regressions_2_perm(lagrangian_values["M"], layer_values_list)
    # best_lin_reg_2["M"] = []
    # for lin_l in lin_reg_2["M"]:
    #     if lin_l == []:
    #         best_lin_reg_2["M"].append([])
    #     else:
    #         best_lin_reg_2["M"].append(lin_l[np.argmin(lin_l[:,0])])
    #
    # best_lin_reg_1["M"] = []
    # for lin_l in lin_reg_1["M"]:
    #     if lin_l == []:
    #         best_lin_reg_1["M"].append([])
    #     else:
    #         best_lin_reg_1["M"].append(lin_l[np.argmin(lin_l[:,0])])
    # best_lin_reg_1["M"] = np.array(best_lin_reg_1["M"])
    # best_lin_reg_2["M"] = np.array(best_lin_reg_2["M"])
    #
    #
    # lin_reg_1["M"].dump("lin_reg_1_M.txt")
    # lin_reg_2["M"].dump("lin_reg_2_M.txt")
    # best_lin_reg_1["M"].dump("best_lin_reg_1_M.txt")
    # best_lin_reg_2["M"].dump("best_lin_reg_2_M.txt")
    #
    # lin_reg_1["COM"] = get_linear_regressions_1_perm(lagrangian_values["COM"], layer_values_list)
    # lin_reg_2["COM"] = get_linear_regressions_2_perm(lagrangian_values["COM"], layer_values_list)
    # best_lin_reg_2["COM"] = []
    # for lin_l in lin_reg_2["COM"]:
    #     if lin_l == []:
    #         best_lin_reg_2["COM"].append([])
    #     else:
    #         best_lin_reg_2["COM"].append(lin_l[np.argmin(lin_l[:, 0])])
    #
    # best_lin_reg_1["COM"] = []
    # for lin_l in lin_reg_1["COM"]:
    #     if lin_l == []:
    #         best_lin_reg_1["COM"].append([])
    #     else:
    #         best_lin_reg_1["COM"].append(lin_l[np.argmin(lin_l[:, 0])])
    #
    #
    # best_lin_reg_1["COM"] = np.array(best_lin_reg_1["M"])
    # best_lin_reg_2["COM"] = np.array(best_lin_reg_2["M"])
    # lin_reg_1["COM"].dump("lin_reg_1_COM.txt")
    # lin_reg_2["COM"].dump("lin_reg_2_COM.txt")
    # best_lin_reg_1["COM"].dump("best_lin_reg_1_COM.txt")
    # best_lin_reg_2["COM"].dump("best_lin_reg_2_COM.txt")

    pass
    # M =
    #
    # for layer_matrix in neuron_values:
    #     for
    #     np.corrcoef()

if __name__ == '__main__':

    main()

#TODO Give filenames more info to identify which hyperparameter is the data for

