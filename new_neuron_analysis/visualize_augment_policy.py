from sklearn import linear_model

from stable_baselines.common.vec_env.vec_video_recorder import VecVideoRecorder

from stable_baselines.low_dim_analysis.eval_util import get_full_param_traj_file_path, get_aug_plot_dir, get_full_params_dir, get_save_dir


import minepy
from matplotlib import pyplot as plt
import numpy as np
import gym
from low_dim_update_stable.new_neuron_analysis.dir_tree_util import *
from stable_baselines import bench, logger
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import os

import pandas as pd
from gym.envs.registration import register

from new_neuron_analysis.experiment_augment_input import read_all_data, get_experiment_path_for_this_run, AttributeDict


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


def visualize_policy_and_collect_COM(augment_num_timesteps, top_num_to_include_slice, augment_seed, augment_run_num, network_size,
               policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num, learning_rate,
               additional_note, metric_param):
    result_dir = get_result_dir(policy_env, policy_num_timesteps, policy_run_num,
                                policy_seed, eval_seed, eval_run_num, additional_note, metric_param)
    args = AttributeDict()

    args.normalize = True
    args.num_timesteps = augment_num_timesteps
    args.run_num = augment_run_num
    args.alg = "ppo2"
    args.seed = augment_seed

    logger.log(f"#######VISUALIZE: {args}")
    # non_linear_global_dict
    linear_global_dict, non_linear_global_dict, lagrangian_values, input_values, layers_values, all_weights = read_all_data(
        policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num, additional_note=additional_note)
    timestamp = get_time_stamp('%Y_%m_%d_%H_%M_%S')
    experiment_label = f"learning_rate_{learning_rate}timestamp_{timestamp}_augment_num_timesteps{augment_num_timesteps}" \
                       f"_top_num_to_include{top_num_to_include_slice.start}_{top_num_to_include_slice.stop}" \
                       f"_augment_seed{augment_seed}_augment_run_num{augment_run_num}_network_size{network_size}" \
                       f"_policy_num_timesteps{policy_num_timesteps}_policy_run_num{policy_run_num}_policy_seed{policy_seed}" \
                       f"_eval_seed{eval_seed}_eval_run_num{eval_run_num}_additional_note_{additional_note}"

    entry_point = 'gym.envs.dart:DartWalker2dEnv_aug_input'

    this_run_dir = get_experiment_path_for_this_run(entry_point, args.num_timesteps, args.run_num,
                                                    args.seed, learning_rate=learning_rate, top_num_to_include=top_num_to_include_slice,
                                                    result_dir=result_dir, network_size=network_size, metric_param=metric_param)
    traj_params_dir_name = get_full_params_dir(this_run_dir)
    save_dir = get_save_dir( this_run_dir)

    aug_plot_dir = get_aug_plot_dir(this_run_dir) + "_vis"


    final_file = get_full_param_traj_file_path(traj_params_dir_name, "pi_final")
    final_params = pd.read_csv(final_file, header=None).values[0]

    args.env = f'{experiment_label}_{entry_point}-v1'
    register(
        id=args.env,
        entry_point=entry_point,
        max_episode_steps=1000,
        kwargs={'linear_global_dict':linear_global_dict,
                'non_linear_global_dict':non_linear_global_dict,
                'top_to_include_slice':top_num_to_include_slice,
                'aug_plot_dir': aug_plot_dir,
                "lagrangian_values":lagrangian_values,
                "layers_values":layers_values}
    )


    def make_env():
        env_out = gym.make(args.env)

        env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
        return env_out

    env = DummyVecEnv([make_env])
    walker_env = env.envs[0].env.env


    walker_env.disableViewer = False

    if args.normalize:
        env = VecNormalize(env)

    set_global_seeds(args.seed)
    walker_env.seed(args.seed)



    model = PPO2.load(f"{save_dir}/ppo2", seed=augment_seed)
    model.set_pi_from_flat(final_params)
    if args.normalize:
        env.load_running_average(save_dir)



    sk = env.venv.envs[0].env.env.robot_skeleton
    lagrangian_values = {}

    obs = np.zeros((env.num_envs,) + env.observation_space.shape)

    obs[:] = env.reset()

    env = VecVideoRecorder(env, aug_plot_dir, record_video_trigger=lambda x: x == 0, video_length=3000, name_prefix="vis_this_policy")

    lagrangian_values["M"] = [sk.M.reshape((-1,1))]
    lagrangian_values["COM"] = [sk.C.reshape((-1,1))]
    lagrangian_values["Coriolis"] = [sk.c.reshape((-1,1))]
    lagrangian_values["q"] = [sk.q.reshape((-1, 1))]
    lagrangian_values["dq"] = [sk.dq.reshape((-1, 1))]


    contact_values = {}


    neuron_values = model.give_neuron_values(obs)
    raw_layer_values_list = [[neuron_value.reshape((-1,1))] for neuron_value in neuron_values]

    env.render()
    ep_infos = []
    steps_to_first_done = 0
    first_done = False

    # epi_rew = 0
    for _ in range(3000):
        actions = model.step(obs)[0]

        # yield neuron_values
        obs, rew, done, infos = env.step(actions)
        # epi_rew+= rew[0]
        if done and not first_done:
            first_done = True

        if not first_done:
            steps_to_first_done += 1


        neuron_values = model.give_neuron_values(obs)


        for i, layer in enumerate(neuron_values):
            raw_layer_values_list[i].append(layer.reshape((-1,1)))

        # fill_contacts_jac_dict(infos[0]["contacts"], contact_dict=contact_values, neuron_values=neuron_values)



        lagrangian_values["M"].append(sk.M.reshape((-1, 1)))
        lagrangian_values["q"].append(sk.q.reshape((-1, 1)))
        lagrangian_values["dq"].append(sk.dq.reshape((-1, 1)))
        lagrangian_values["COM"].append(sk.C.reshape((-1, 1)))
        lagrangian_values["Coriolis"].append(sk.c.reshape((-1, 1)))

        # env.render()

        # time.sleep(1)
        for info in infos:
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                ep_infos.append(maybe_ep_info)

        env.render()
        done = done.any()
        if done:
            episode_rew = safe_mean([ep_info['r'] for ep_info in ep_infos])
            print(f'episode_rew={episode_rew}')
            # print(f'episode_rew={epi_rew}')
            # epi_rew = 0
            obs = env.reset()


    #Hstack into a big matrix
    lagrangian_values["M"] = np.hstack(lagrangian_values["M"])
    lagrangian_values["COM"] = np.hstack(lagrangian_values["COM"])
    lagrangian_values["Coriolis"] = np.hstack(lagrangian_values["Coriolis"])
    lagrangian_values["q"] = np.hstack(lagrangian_values["q"])
    lagrangian_values["dq"] = np.hstack(lagrangian_values["dq"])

    # for contact_body_name, l in contact_values.items():
    #     body_contact_dict = contact_values[contact_body_name]
    #     for name, l in body_contact_dict.items():
    #         body_contact_dict[name] = np.hstack(body_contact_dict[name])
    input_values = np.hstack(raw_layer_values_list[0])

    layers_values = [np.hstack(layer_list) for layer_list in raw_layer_values_list][1:-2]# drop variance and inputs



    for i, com in enumerate(lagrangian_values["COM"]):
        plt.figure()
        plt.plot(np.arange(len(com)), com)
        plt.xlabel("time")
        plt.ylabel(f"COM{i}")

        plt.savefig(f"{aug_plot_dir}/COM{i}.jpg")
        plt.close()

def visualize_augment_experiment(augment_num_timesteps, top_num_to_include_slice, augment_seed, augment_run_num, network_size,
                   policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num, learning_rate,
                    additional_note, result_dir, lagrangian_inds_to_include=None):

    args = AttributeDict()

    args.normalize = True
    args.num_timesteps = augment_num_timesteps
    args.run_num = augment_run_num
    args.alg = "ppo2"
    args.seed = augment_seed

    logger.log(f"#######TRAIN: {args}")
    # non_linear_global_dict
    timestamp = get_time_stamp('%Y_%m_%d_%H_%M_%S')
    experiment_label = f"learning_rate_{learning_rate}timestamp_{timestamp}_augment_num_timesteps{augment_num_timesteps}" \
                       f"_top_num_to_include{top_num_to_include_slice.start}_{top_num_to_include_slice.stop}" \
                       f"_augment_seed{augment_seed}_augment_run_num{augment_run_num}_network_size{network_size}" \
                       f"_policy_num_timesteps{policy_num_timesteps}_policy_run_num{policy_run_num}_policy_seed{policy_seed}" \
                       f"_eval_seed{eval_seed}_eval_run_num{eval_run_num}_additional_note_{additional_note}"

    if policy_env == "DartWalker2d-v1":
        entry_point = 'gym.envs.dart:DartWalker2dEnv_aug_input'
    elif policy_env == "DartHopper-v1":
        entry_point = 'gym.envs.dart:DartHopperEnv_aug_input'
    elif policy_env == "DartHalfCheetah-v1":
        entry_point = 'gym.envs.dart:DartHalfCheetahEnv_aug_input'
    elif policy_env == "DartSnake7Link-v1":
        entry_point = 'gym.envs.dart:DartSnake7LinkEnv_aug_input'
    else:
        raise NotImplemented()


    this_run_dir = get_experiment_path_for_this_run(entry_point, args.num_timesteps, args.run_num,
                                                    args.seed, learning_rate=learning_rate, top_num_to_include=top_num_to_include_slice,
                                                    result_dir=result_dir, network_size=network_size)
    full_param_traj_dir_path = get_full_params_dir(this_run_dir)
    log_dir = get_log_dir(this_run_dir)
    save_dir = get_save_dir(this_run_dir)


    create_dir_remove(this_run_dir)
    create_dir_remove(full_param_traj_dir_path)
    create_dir_remove(save_dir)
    create_dir_remove(log_dir)
    logger.configure(log_dir)

    # note this is only linear
    if lagrangian_inds_to_include is None:
        linear_top_vars_list = read_linear_top_var(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed,
                                           eval_run_num, additional_note)

        # keys_to_include = ["COM", "M", "Coriolis", "total_contact_forces_contact_bodynode",
        #                    "com_jacobian", "contact_bodynode_jacobian"]
        keys_to_include = ["COM", "M", "Coriolis", "com_jacobian"]
        # lagrangian_inds_to_include = linear_top_vars_list[top_num_to_include_slice]
        lagrangian_inds_to_include = get_wanted_lagrangians(keys_to_include, linear_top_vars_list, top_num_to_include_slice)


    with open(f"{log_dir}/lagrangian_inds_to_include.json", 'w') as fp:
        json.dump(lagrangian_inds_to_include, fp)


    args.env = f'{experiment_label}_{entry_point}-v1'
    register(
        id=args.env,
        entry_point=entry_point,
        max_episode_steps=1000,
        kwargs={"lagrangian_inds_to_include": lagrangian_inds_to_include}
    )


    def make_env():
        env_out = gym.make(args.env)
        env_out.env.visualize = False
        env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
        return env_out

    env = DummyVecEnv([make_env])
    walker_env = env.envs[0].env.env
    walker_env.disableViewer = True


    if args.normalize:
        env = VecNormalize(env)
    policy = MlpPolicy


    # extra run info I added for my purposes
    run_info = {"run_num": args.run_num,
                "env_id": args.env,
                "full_param_traj_dir_path": full_param_traj_dir_path}

    layers = [network_size, network_size]
    set_global_seeds(args.seed)
    walker_env.seed(args.seed)

    policy_kwargs = {"net_arch" : [dict(vf=layers, pi=layers)]}
    model = PPO2(policy=policy, env=env, n_steps=4096, nminibatches=64, lam=0.95, gamma=0.99,
                 noptepochs=10,
                 ent_coef=0.0, learning_rate=learning_rate, cliprange=0.2, optimizer='adam', policy_kwargs=policy_kwargs,
                 seed=args.seed)
    model.tell_run_info(run_info)


    model.learn(total_timesteps=args.num_timesteps, seed=args.seed)

    model.save(f"{save_dir}/ppo2")

    if args.normalize:
        env.save_running_average(save_dir)

    return log_dir

if __name__ == '__main__':
    # visualize_policy_and_collect_COM(seed=0, run_num=0, policy_num_timesteps=3000000, policy_run_num=0, policy_seed=0)

    policy_env = "DartWalker2d-v1"
    policy_seed = 3
    policy_run_num = 1
    policy_num_timesteps = 5000000
    policy_env = "DartWalker2d-v1"

    eval_seed = 4
    eval_run_num = 4

    augment_seed = 2
    augment_run_num = 0
    augment_num_timesteps = 1500000
    top_num_to_include = slice(0,0)
    network_size = 64

    additional_note = ""
    learning_rate = 64 / network_size * 3e-4
    metric_param = None
    visualize_policy_and_collect_COM(augment_num_timesteps, top_num_to_include_slice=top_num_to_include, augment_seed=augment_seed,
                                                       augment_run_num=augment_run_num, network_size=network_size,
                                                       policy_env=policy_env, policy_num_timesteps=policy_num_timesteps,
                                                       policy_run_num=policy_run_num, policy_seed=policy_seed, eval_seed=eval_seed,
                                                       eval_run_num=eval_run_num, learning_rate=learning_rate,
                                     additional_note=additional_note, metric_param=metric_param)
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

