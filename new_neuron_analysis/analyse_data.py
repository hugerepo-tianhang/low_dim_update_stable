import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import shutil
import numpy as np

import minepy
from sklearn import  linear_model
import copy
from joblib import delayed, Parallel
import pickle
from low_dim_update_stable.new_neuron_analysis.dir_tree_util import *
import json
regr = linear_model.LinearRegression()


class NonLinearGlobalDictRow(object):
    length = 3
    reg = {"mic": 0, "layer_ind": 1, "neuron_ind": 2}

    @classmethod
    def get_non_linear_global_dict_row(cls, mic, layer_ind, neuron_ind):
        result = [np.nan] * cls.length
        result[cls.reg["mic"]] = mic
        result[cls.reg["layer_ind"]] = layer_ind
        result[cls.reg["neuron_ind"]] = neuron_ind

        return result


class LinearGlobalDictRow(object):
    length = 3
    reg = {"co": 0, "layer_ind": 1, "neuron_ind": 2}

    @classmethod
    def get_linear_global_dict_row(cls, linear_co, layer_ind, neuron_ind):
        result = [np.nan] * cls.length
        result[cls.reg["co"]] = linear_co
        result[cls.reg["layer_ind"]] = layer_ind
        result[cls.reg["neuron_ind"]] = neuron_ind

        return result



def get_normalized_SSE(lagrange_l, neuron_l, regr):
    neuron_l_copy = copy.deepcopy(neuron_l)
    range_neuron = max(neuron_l_copy) - min(neuron_l_copy)
    if range_neuron != 0:
        neuron_l_copy = neuron_l_copy / range_neuron

    X = lagrange_l.reshape(-1,1)
    y = neuron_l_copy.reshape(-1, 1)
    y[np.isnan(y)] = 0
    X[np.isnan(X)] = 0
    regr.fit(X, y)
    y_pred = regr.predict(X)
    SSE = ((y - y_pred) ** 2).sum()
    TV = ((y - np.average(y)) ** 2).sum()
    return SSE,TV

# def get_Reflective_correlation_coefficient(lagrange_l, neuron_l):
#
#     prod = np.dot(lagrange_l, neuron_l)
#     A = np.sum(lagrange_l**2)
#     B = np.sum(neuron_l**2)
#     return prod/np.sqrt(A*B)




def scatter_the_non_linear_significant_ones(non_linear_global_dict, BEST_TO_TAKE,
                                        layer_values_list, lagrangian_values, plot_dir):

    non_linear_best_dir = f"{plot_dir}/non_linear_best"
    if not os.path.exists(non_linear_best_dir):
        os.makedirs(non_linear_best_dir)


    for key, nda in non_linear_global_dict.items():
        for ind, non_linear_cos in enumerate(nda):

            lagrange_l = lagrangian_values[key][ind]


            non_linear_cos = np.array(non_linear_cos)

            take = min(BEST_TO_TAKE, len(non_linear_cos))
            if take > 0:

                best_rows = non_linear_cos[np.argpartition(np.abs(non_linear_cos[:, 0]), -take)[-take:]]
                for best_row in best_rows:
                    best_layer_ind= best_row[NonLinearGlobalDictRow.reg["layer_ind"]]
                    best_neuron_ind= best_row[NonLinearGlobalDictRow.reg["neuron_ind"]]
                    best_mic = best_row[NonLinearGlobalDictRow.reg["mic"]]


                    best_neuron_l = layer_values_list[int(best_layer_ind)][int(best_neuron_ind)]

                    name = f"{non_linear_best_dir}/{key}_index_{ind}_VS_layer_{best_layer_ind}" \
                           f"_neuron_{best_neuron_ind}_nonlinear_correlation{best_mic}.jpg"


                    plt.figure()

                    plt.scatter(lagrange_l, best_neuron_l)
                    plt.xlabel("lagrange")
                    plt.ylabel("neuron")

                    plt.savefig(name)
                    plt.close()

def crunch_non_linear_correlation(lagrangian_values, layer_values_list, data_dir):
    def get_mic_co(lagrange_l, neuron_l, layer_ind, neuron_ind):
        def compute_alpha(npoints):
            NPOINTS_BINS = [1, 25, 50, 250, 500, 1000, 2500, 5000, 10000, 40000]
            ALPHAS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4]

            if npoints < 1:
                raise ValueError("the number of points must be >=1")

            return ALPHAS[np.digitize([npoints], NPOINTS_BINS)[0] - 1]

        alpha_cl = compute_alpha(lagrange_l.shape[0])
        mine = minepy.MINE(alpha=alpha_cl, c=5, est="mic_e")
        mine.compute_score(lagrange_l, neuron_l)
        mic = mine.mic()

        range_neuron = max(neuron_l) - min(neuron_l)
        if range_neuron < 1e-5:
            mic = 0
        if np.isnan(mic):
            mic = 0

        return NonLinearGlobalDictRow.get_non_linear_global_dict_row(mic, layer_ind, neuron_ind)

    non_linear_global_dict = {}
    for key, nda in lagrangian_values.items():
        non_linear_global_dict[key] = [100] * len(nda)
        for ind, lagrange_l in enumerate(nda):
            print(f"nonlinear currently {key}_index_{ind}")
            non_linear_cos = Parallel(n_jobs=-1)(delayed(get_mic_co)(lagrange_l, neuron_l, layer_ind, neuron_ind)
                                                 for layer_ind, layer in enumerate(layer_values_list)
                                                 for neuron_ind, neuron_l in enumerate(layer))

            non_linear_global_dict[key][ind] = list(non_linear_cos)

        if 100 in non_linear_global_dict[key]:
            raise Exception("not good nonlinear")

    if not os.path.exists(data_dir):
        raise Exception("where does the raw data come from then??")


    fn = f"{data_dir}/non_linear_global_dict.json"
    with open(fn, 'w') as fp:
        json.dump(non_linear_global_dict, fp, sort_keys=True, indent=4)
    return non_linear_global_dict

def scatter_the_linear_significant_ones(linear_global_dict, BEST_TO_TAKE, layer_values_list, lagrangian_values, plot_dir):
    linear_best_dir = f"{plot_dir}/linear_best"
    if not os.path.exists(linear_best_dir):
        os.makedirs(linear_best_dir)

    for key, nda in linear_global_dict.items():
        for ind, linear_cos in enumerate(nda):

            lagrange_l = lagrangian_values[key][ind]
            linear_cos = np.array(linear_cos)

            take = min(BEST_TO_TAKE, len(linear_cos))
            if take > 0:
                best_rows = linear_cos[np.argpartition(np.abs(linear_cos[:, 0]), -take)[-take:]]

                for best_row in best_rows:
                    best_layer_ind= best_row[LinearGlobalDictRow.reg["layer_ind"]]
                    best_neuron_ind= best_row[LinearGlobalDictRow.reg["neuron_ind"]]
                    best_co = best_row[LinearGlobalDictRow.reg["co"]]

                    best_neuron_l = layer_values_list[int(best_layer_ind)][int(best_neuron_ind)]

                    best_normalized_SSE, best_TV = get_normalized_SSE(lagrange_l, best_neuron_l, regr)

                    name = f"{linear_best_dir}/{key}_index_{ind}_VS_layer_{best_layer_ind}_neuron_{best_neuron_ind}_" \
                           f"linear_correlation{best_co}_normalized_SSE_{best_normalized_SSE}_Syy_{best_TV}.jpg"
                    plt.figure()

                    plt.scatter(lagrange_l, best_neuron_l)
                    plt.xlabel("lagrange")
                    plt.ylabel("neuron")

                    plt.savefig(name)
                    plt.close()



def crunch_linear_correlation(lagrangian_values, layers_values_list, data_dir):
    def get_co(lagrange_l, neuron_l, layer_ind, neuron_ind):
        linear_co = np.corrcoef(lagrange_l, neuron_l)[1, 0]
        range_neuron = max(neuron_l) - min(neuron_l)
        if range_neuron < 1e-5:
            linear_co = 0
        if np.isnan(linear_co):
            linear_co = 0

        normalized_SSE, best_TV = get_normalized_SSE(lagrange_l, neuron_l, regr)
        #TODO get a weighted metric??
        if normalized_SSE > 150:
            linear_co = 0

        linear_global_dict_row = LinearGlobalDictRow.get_linear_global_dict_row(linear_co, layer_ind, neuron_ind)
        return linear_global_dict_row

    linear_global_dict = {}
    for key, nda in lagrangian_values.items():
        linear_global_dict[key] = [100]*len(nda)
        for ind, lagrange_l in enumerate(nda):

            print(f"linear currently {key}_index_{ind}")

            linear_cos = Parallel(n_jobs=-1)(
                delayed(get_co)(lagrange_l, neuron_l, layer_ind, neuron_ind) for layer_ind, layer in
                enumerate(layers_values_list) for neuron_ind, neuron_l in enumerate(layer))
            linear_global_dict[key][ind] = linear_cos

        if 100 in linear_global_dict[key]:
            raise Exception("not good")

    if not os.path.exists(data_dir):
        raise Exception("where does the raw data come from then??")

    fn = f"{data_dir}/linear_global_dict.json"
    with open(fn, 'w') as fp:
        json.dump(linear_global_dict, fp, sort_keys=True, indent=4)

    return linear_global_dict

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

def read_data(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num, num_layers=2):
    data_dir = get_data_dir(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num)
    lagrangian_values_fn = f"{data_dir}/lagrangian.pickle"

    with open(lagrangian_values_fn, 'rb') as handle:
        lagrangian_values = pickle.load(handle)

    input_values_fn = f"{data_dir}/input_values.npy"
    layers_values_fn = f"{data_dir}/layer_values.npy"

    input_values = np.load(input_values_fn)
    layers_values = np.load(layers_values_fn)


    all_weights = [0]*num_layers
    for layer_ind in range(num_layers):
        fname = f"{data_dir}/weights_layer_{layer_ind}.txt"
        weights = np.loadtxt(fname)
        all_weights[layer_ind] = weights
    return lagrangian_values, input_values, layers_values, all_weights

def crunch_and_plot_data(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num):
    lagrangian_values, input_values, layers_values_list, all_weights = read_data(policy_env, policy_num_timesteps,
                                                                                 policy_run_num, policy_seed,
                                                                                 eval_seed, eval_run_num)

    data_dir = get_data_dir(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num)
    # plot_dir = get_plot_dir(env=env, num_timesteps=num_timesteps, seed=seed, run_num=run_num)
    #
    # if os.path.exists(plot_dir):
    #     shutil.rmtree(plot_dir)
    # os.makedirs(plot_dir)


    BEST_TO_TAKE = 5

    linear_global_dict = crunch_linear_correlation(lagrangian_values, layers_values_list, data_dir)
    # scatter_the_linear_significant_ones(linear_global_dict, BEST_TO_TAKE, layers_values_list, lagrangian_values, plot_dir)


    non_linear_global_dict = crunch_non_linear_correlation(lagrangian_values, layers_values_list, data_dir)
    # scatter_the_non_linear_significant_ones(non_linear_global_dict, BEST_TO_TAKE, layers_values_list,
    #                                         lagrangian_values, plot_dir)

if __name__ == "__main__":
    policy_num_timesteps = 3000000
    policy_env = "DartWalker2d-v1"
    eval_seeds = [0, 1, 2]
    eval_run_nums = [0, 1, 2]



    policy_run_num = 0
    policy_seed = 0

    for eval_seed in eval_seeds:
        for eval_run_num in eval_run_nums:
            crunch_and_plot_data(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num)

    # import json
    # out_dir = f"/home/panda-linux/PycharmProjects/DeepMimic/neuron_vis/plots_{env}_{num_timesteps}"
    # fn = f"{out_dir}/data/linear_global_dict.json"
    #
    # non_linear_out_dir = f"{out_dir}/non_linear"
    # non_fn = f"{non_linear_out_dir}/data/non_linear_global_dict.json"
    # with open(non_fn, 'r') as non_fp:
    #     non_linear_d = json.load(non_fp)
    #
    #     with open(fn, 'r') as fp:
    #         linear_d = json.load(fp)
    #
    #         best_linear_cos = {"M": 100*np.ones(43*43), "COM": 100*np.ones(4), "C": 100*np.ones(43)}
    #         for lagrange_name, data in linear_d.items():
    #             lagrange_key, lagrange_ind = lagrange_name.split("_")
    #             data = np.array(data)
    #             linear_cos = data[:,0]
    #
    #
    #             best_linear_cos[lagrange_key][int(lagrange_ind)] = np.max(np.abs(linear_cos))
    #
    #         for key, l in best_linear_cos.items():
    #
    #             l = np.array(l)
    #             # if key == "M":
    #             #     print("")
    #             not_zero_l = l[l>=0.0000001]
    #             r90_100 = l[l>0.9]
    #             r80_90 = l[(l>0.8) & (l<0.9)]
    #             r60_80 = l[(l>0.6) & (l<0.8)]
    #             r0_60 = l[(l>0.0000001) & (l<0.6)]
    #             print("LINEAR")
    #             print(f"for key: {key}")
    #             print(f"total: {len(l)}")
    #             print(f"total not_zero_l: {len(not_zero_l)}")
    #             print(f"total r90_100: {len(r90_100)}")
    #             print(f"total r80_90: {len(r80_90)}")
    #             print(f"total r60_80: {len(r60_80)}")
    #             print(f"total r0_60: {len(r0_60)}")
            # pass
            #
            # m = lagrangian_values["M"]
            # max_min = np.max(m,axis=1) - np.min(m,axis=1)
            # set(np.argwhere(max_min == 0).reshape(-1).tolist()) - set(np.argwhere(l == 0).reshape(-1).tolist())
            #
            # max_min = np.max(np.array(lagrangian_values["M"])[l == 0], axis=1) - np.min(np.array(lagrangian_values["M"])[l == 0],
            #                                                                   axis=1)

    #
    #
    #
    #         best_non_linear_cos = {"M": 100*np.ones(43*43), "COM": 100*np.ones(4), "C": 100*np.ones(43)}
    #         for lagrange_name, data in non_linear_d.items():
    #             lagrange_key, lagrange_ind = lagrange_name.split("_")
    #             data = np.array(data)
    #             mics = data[:,0]
    #
    #             best_non_linear_cos[lagrange_key][int(lagrange_ind)] = np.max(np.abs(mics))
    #
    #         for key, non_l in best_non_linear_cos.items():
    #             non_l = np.array(non_l)
    #             not_zero_l = non_l[non_l>=0.0000001]
    #             r90_100 = non_l[non_l>0.9]
    #             r80_90 = non_l[(non_l>0.8) & (non_l<0.9)]
    #             r60_80 = non_l[(non_l>0.6) & (non_l<0.8)]
    #             r0_60 = non_l[(non_l>0.0000001) & (non_l<0.6)]
    #             print("NON LINEAR")
    #             print(f"for key: {key}")
    #             print(f"total: {len(non_l)}")
    #             print(f"total not_zero_l: {len(not_zero_l)}")
    #             print(f"total r90_100: {len(r90_100)}")
    #             print(f"total r80_90: {len(r80_90)}")
    #             print(f"total r60_80: {len(r60_80)}")
    #             print(f"total r0_60: {len(r0_60)}")
    #
    #
    #         M_best = np.array(best_linear_cos["M"])
    #         weak_l_ind = np.argwhere((M_best<0.6)&(M_best>0.0000001)).reshape(-1)
    #         best_non_linear_cos[weak_l_ind]
    #
    #
    #         pass
