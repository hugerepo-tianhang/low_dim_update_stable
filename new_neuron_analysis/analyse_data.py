import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import shutil
import numpy as np

import minepy
from sklearn import  linear_model
from joblib import delayed, Parallel
import pickle
from new_neuron_analysis.dir_tree_util import *
import json
import copy

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
    length = 4
    reg = {"co": 0, "normalized_SSE": 1, "layer_ind": 2, "neuron_ind": 3}

    @classmethod
    def get_linear_global_dict_row(cls, linear_co, normalized_SSE, layer_ind, neuron_ind):
        result = [np.nan] * cls.length
        result[cls.reg["co"]] = linear_co
        result[cls.reg["normalized_SSE"]] = normalized_SSE
        result[cls.reg["layer_ind"]] = layer_ind
        result[cls.reg["neuron_ind"]] = neuron_ind

        return result



def get_mean_normalized_SSE(lagrange_l, neuron_l, regr):
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
    mean_SSE = np.mean((y - y_pred) ** 2)
    mean_TV = np.mean((y - np.average(y)) ** 2)
    return mean_SSE,mean_TV

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

                    best_normalized_SSE, best_TV = get_mean_normalized_SSE(lagrange_l, best_neuron_l, regr)

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

        normalized_SSE, best_TV = get_mean_normalized_SSE(lagrange_l, neuron_l, regr)
        #TODO get a weighted metric??
        # if normalized_SSE > 150:
        #     linear_co = 0

        linear_global_dict_row = LinearGlobalDictRow.get_linear_global_dict_row(linear_co, normalized_SSE, layer_ind, neuron_ind)
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

def read_data(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num, additional_note, num_layers=2):
    data_dir = get_data_dir(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num, additional_note)
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

#===================
# give best correlated variables to use
#====================
def plot_best(lagrangian_l, neuron_l, fig_name, aug_plot_dir, regr):

    plt.figure()

    plt.scatter(lagrangian_l, neuron_l)
    # plt.plot(lagrangian_l, regr.predict(lagrangian_l), color='blue', linewidth=3)

    plt.xlabel("lagrange")
    plt.ylabel("neuron")

    plt.savefig(f"{aug_plot_dir}/{fig_name}")
    plt.close()

import shutil
def create_dir_remove(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)



def get_upper_tri(linear_M_nd):
    num_M = linear_M_nd.shape[0]
    n = np.sqrt(num_M)
    upper_tri_inds = np.triu_indices(n)
    flattened_ind = [int(row * n + col) for row, col in zip(upper_tri_inds[0], upper_tri_inds[1])]
    upper_tri_linear_M_nd = linear_M_nd[flattened_ind,:]
    return upper_tri_linear_M_nd, flattened_ind

def get_key_and_ind(ind, num_ind_in_stack, M_flattened_ind):
    lagrangian_key = None
    past_num = 0
    for (key, key_num) in num_ind_in_stack:
        if ind < past_num + key_num:
            lagrangian_key = key
            key_ind = ind - past_num
            # ind is actually tri_M_index
            if key == "M":
                lagrangian_index = M_flattened_ind[key_ind]
            else:
                lagrangian_index = key_ind

            break
        else:
            past_num += key_num

    if lagrangian_key is None:
        raise Exception(f"wtf ind out of bound to lookup ind{ind}")
    return lagrangian_key, int(lagrangian_index)



from new_neuron_analysis.run_trained_policy import lagrangian_keys as lagrangian_keys_in_run_policy
def linear_lagrangian_to_include_in_state(linear_global_dict, data_dir,
                                          lagrangian_values, layers_values, metric_param):
    aug_plot_dir = f"{data_dir}/top_vars_plots"

    num_dof = np.sqrt(len(linear_global_dict["M"]))
    assert num_dof == int(num_dof)


    def get_concat_data(linear_global_dict):

        concat = None
        num_ind_in_stack = []

        # keys_to_include = ["COM", "M", "Coriolis", "total_contact_forces_contact_bodynode",
        #                    "com_jacobian", "contact_bodynode_jacobian"]



        for key in lagrangian_keys_in_run_policy:

            # if key not in lagrangian_keys_in_run_policy:
            #     raise Exception(f"this key {key} in not in lagrangian_keys")

            nd = np.array(linear_global_dict[key])
            if key == "M":
                nd, M_flattened_ind = get_upper_tri(nd)

            num = nd.shape[0]

            num_ind_in_stack.append((key, num))

            if concat is None:
                concat = nd
            else:
                concat = np.vstack((concat, nd))

        #if there is no M just raise
        return num_ind_in_stack, concat, M_flattened_ind



    linear_correlation_list = []


    num_ind_in_stack, concat, M_flattened_ind = get_concat_data(linear_global_dict)


    linear_cos = np.abs(concat[:,:,0])
    normalized_SSE = concat[:,:,1]
    max_normalized_SSE = 1/15 # 200/3000 if higher than this, start getting neg

    # linear_cos[np.where(normalized_SSE>max_normalized_SSE)] = 0 # if SSE too big directly determined that it's not linear
    new_metric_matrix = metric_param *linear_cos + (1 - normalized_SSE/max_normalized_SSE) * (1-metric_param) # (max - norm)/max
    # new_metric_matrix = normalized_SSE


    argmax_for_each = np.argmax(new_metric_matrix, axis=1)
    max_over_new_metric_for_each_lagrange = new_metric_matrix[np.arange(len(argmax_for_each)), argmax_for_each]

    max_over_neurons_concat = concat[np.arange(len(argmax_for_each)), argmax_for_each]



    # #new use new metric for lagrangian wide
    # arg_sorted = np.argsort(max_over_new_metric_for_each_lagrange)
    # arg_sorted = arg_sorted[::-1]

    # old use linear cos for lagrangian wide
    max_for_each_lagrange = np.abs(max_over_neurons_concat[:,0])


    arg_sorted = np.argsort(max_for_each_lagrange)
    arg_sorted = arg_sorted[::-1]

    #now arg_sorted has the biggest correlated var at the first element
    if aug_plot_dir is not None:
        create_dir_remove(aug_plot_dir)


    test_list = []
    linear_correlation_neuron_list = []
    for i, ind in enumerate(arg_sorted):

        neuron_coord = max_over_neurons_concat[ind][-2:]
        linear_co =  max_over_neurons_concat[ind][0]
        normalized_SSE =  max_over_neurons_concat[ind][1]
        new_metric = metric_param * np.abs(linear_co) + \
                     (1 - normalized_SSE / max_normalized_SSE) * (1 - metric_param)  # (max - norm)/max

        lagrangian_key, lagrangian_index = get_key_and_ind(ind, num_ind_in_stack, M_flattened_ind)


        linear_correlation_list.append((lagrangian_key, lagrangian_index, linear_co, new_metric))
        linear_correlation_neuron_list.append((int(neuron_coord[0]), int(neuron_coord[1])))
        #=================================
        # for debugging below

        lagrangian_l = lagrangian_values[lagrangian_key][lagrangian_index]
        neuron_l = layers_values[int(neuron_coord[0]), int(neuron_coord[1]), :]


        test_linear_co = np.abs(np.corrcoef(lagrangian_l, neuron_l)[1, 0])
        test_normalized_SSE, test_TV = get_mean_normalized_SSE(lagrangian_l, neuron_l, regr)
        if aug_plot_dir is not None:
            fig_name = f"largest_rank_{i}_{lagrangian_key}_{lagrangian_index}_VS_layer{neuron_coord[0]}" \
                       f"_neuron_{neuron_coord[1]}_linear_co_{linear_co} normalized_SSE{normalized_SSE}.jpg"
            plot_best(lagrangian_l, neuron_l, fig_name, aug_plot_dir, regr)

    #
    #     if test_normalized_SSE > max_normalized_SSE:
    #         test_linear_co = 0
    #     test_new_metric = 0.5 * test_linear_co + (1 - test_normalized_SSE / max_normalized_SSE) * 0.5
    #     test_list.append(test_new_metric)
    #     #====================================
    # assert max(np.abs(np.array(test_list) - max_over_new_metric_for_each_lagrange[arg_sorted])) < 0.0000001

    fn = f"{data_dir}/linear_top_vars_list_metric_param_{metric_param}.json"
    with open(fn, 'w') as fp:
        json.dump(linear_correlation_list, fp)

    fn = f"{data_dir}/linear_correlation_neuron_list_metric_param_{metric_param}.json"
    with open(fn, 'w') as fp:
        json.dump(linear_correlation_neuron_list, fp)


    # show_M_matrix(num_dof, result=result, top_num_to_include_slice=slice(0,10), save_dir=aug_plot_dir),
    # show_M_matrix(num_dof, result=result, top_num_to_include_slice=slice(0,20), save_dir=aug_plot_dir)
    return linear_correlation_list, linear_correlation_neuron_list


def crunch_and_plot_data(trained_policy_env, trained_policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num, additional_note, metric_param):
    lagrangian_values, input_values, layers_values_list, all_weights = read_data(trained_policy_env,
                                                                                 trained_policy_num_timesteps,
                                                                                 policy_run_num, policy_seed,
                                                                                 eval_seed, eval_run_num, additional_note=additional_note)

    data_dir = get_data_dir(trained_policy_env, trained_policy_num_timesteps, policy_run_num, policy_seed, eval_seed,
                            eval_run_num, additional_note=additional_note)


    linear_global_dict = crunch_linear_correlation(lagrangian_values, layers_values_list, data_dir)
    # BEST_TO_TAKE = 5
    # scatter_the_linear_significant_ones(linear_global_dict, BEST_TO_TAKE, layers_values_list, lagrangian_values,
    #                                     data_dir)

    # non_linear_global_dict = crunch_non_linear_correlation(lagrangian_values, layers_values_list, data_dir)
    # scatter_the_non_linear_significant_ones(non_linear_global_dict, BEST_TO_TAKE, layers_values_list,
    #                                         lagrangian_values, data_dir)


#
# def lagrangian_to_include_in_state_non_linear(linear_global_dict, non_linear_global_dict, top_to_include_slice, aug_plot_dir,
#                                    lagrangian_values, layers_values):
#
#     result = {"M": [], "Coriolis": [], "COM": []}
#     if top_to_include_slice.stop - top_to_include_slice.start == 0:
#         return result
#
#
#     non_linear_M_nd = np.array(non_linear_global_dict["M"])
#     non_linear_C_nd = np.array(non_linear_global_dict["Coriolis"])
#     non_linear_COM_nd = np.array(non_linear_global_dict["COM"])
#
#     num_C = non_linear_C_nd.shape[0]
#     num_COM = non_linear_COM_nd.shape[0]
#
#     upper_tri_linear_M_nd, flattened_ind = get_upper_tri(non_linear_M_nd)
#     num_tri_M = upper_tri_linear_M_nd.shape[0]
#
#     concat = np.vstack((upper_tri_linear_M_nd, non_linear_C_nd, non_linear_COM_nd))
#
#
#     mics = np.abs(concat[:,:,0])
#
#     new_metric_matrix = mics
#     argmax_for_each = np.argmax(new_metric_matrix, axis=1)
#
#     max_over_neurons_concat = concat[np.arange(len(argmax_for_each)), argmax_for_each]
#     max_for_each_lagrange = np.abs(max_over_neurons_concat[:,0])
#
#     end = top_to_include_slice.stop
#     start = top_to_include_slice.start
#
#     if len(max_for_each_lagrange) < end:
#         raise Exception("Not even enough lagrangian that you asked for")
#
#     arg_to_include_orignal_index = np.argpartition(max_for_each_lagrange, -end)[len(max_for_each_lagrange) - end:]
#
#     num_to_include = end - start
#     arg_to_include_top_index = np.argpartition(max_for_each_lagrange[arg_to_include_orignal_index], -num_to_include)[:num_to_include]
#     arg_to_include = arg_to_include_orignal_index[arg_to_include_top_index]
#
#     create_dir_remove(aug_plot_dir)
#
#
#     for ind in arg_to_include:
#         neuron_coord = max_over_neurons_concat[ind][-2:]
#         mic =  max_over_neurons_concat[ind][0]
#         if ind < num_tri_M:
#             lagrangian_key = "M"
#             #ind is actually tri_M_index
#             M_ind = flattened_ind[ind]
#             lagrangian_index = M_ind
#
#
#         elif ind < num_C + num_tri_M:
#             lagrangian_key = "Coriolis"
#
#             C_ind = ind - num_tri_M
#             lagrangian_index = C_ind
#
#
#         elif ind < num_tri_M + num_C + num_COM:
#             lagrangian_key = "COM"
#
#             COM_ind = ind - (num_tri_M + num_C)
#             lagrangian_index = COM_ind
#
#
#         else:
#             raise Exception(f"WHAT? ind{ ind }")
#
#         result[lagrangian_key].append(lagrangian_index)
#
#         lagrangian_l = lagrangian_values[lagrangian_key][lagrangian_index]
#         # check_linear_co = np.max(np.abs(np.array(linear_global_dict[lagrangian_key][lagrangian_index])[:,LinearGlobalDictRow.reg["co"]]))
#         # if check_linear_co != mic:
#         #     print("s")
#         # assert  check_linear_co == mic, f"check_linear_co {check_linear_co} VS mic{mic}"
#
#
#         neuron_l = layers_values[int(neuron_coord[0]), int(neuron_coord[1]), :]
#         fig_name = f"{lagrangian_key}_{lagrangian_index}_VS_layer{neuron_coord[0]}" \
#                    f"_neuron_{neuron_coord[1]}_mic_{mic}.jpg"
#
#
#         plot_best(lagrangian_l, neuron_l, fig_name, aug_plot_dir)
#
#
#     num_inds_to_add = sum(map(len, result))
#     assert num_inds_to_add == top_to_include_slice.stop - top_to_include_slice.start
#     if num_inds_to_add == 0:
#         assert result == {"M": [], "Coriolis": [], "COM": []}
#
#     return result


if __name__ == "__main__":
    policy_env = "DartWalker2d-v1"
    policy_num_timesteps = 5000000
    policy_run_nums = [1]
    policy_seeds = [3]
    eval_seed = 4
    eval_run_num = 4
    aug_num_timesteps=1500000
    additional_note="use_COM_Jac_on_hopper_and_walker"
    metric_param =0.5
    for policy_run_num in policy_run_nums:
        for policy_seed in policy_seeds:

            crunch_and_plot_data(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num, additional_note, metric_param)

            # plot_dir = get_plot_dir(env=env, num_timesteps=num_timesteps, seed=seed, run_num=run_num)
            #
            # if os.path.exists(plot_dir):
            #     shutil.rmtree(plot_dir)
            # os.makedirs(plot_dir)


            # BEST_TO_TAKE = 5
            # linear_global_dict = crunch_linear_correlation(lagrangian_values, layers_values_list, data_dir)

            # scatter_the_linear_significant_ones(linear_global_dict, BEST_TO_TAKE, layers_values_list, lagrangian_values,
            #                                     data_dir)
            #
            # non_linear_global_dict = crunch_non_linear_correlation(lagrangian_values, layers_values_list, data_dir)
            # scatter_the_non_linear_significant_ones(non_linear_global_dict, BEST_TO_TAKE, layers_values_list,
            #                                         lagrangian_values, data_dir)
            # aug_plot_dir ="/home/panda-linux/PycharmProjects/low_dim_update_dart/low_dim_update_stable/new_neuron_analysis/result/DartWalker2d-v1_policy_num_timesteps_5000000_policy_run_num_1_policy_seed_3_run_seed_4_run_run_num_4_additional_note_"
            # result_10 = [("M", 2), ("M", 3), ("M", 5), ("M", 15), ("M", 17), ("M", 40), ("M", 41), ("M", 70), ("M", 71)]
            # result_20 = [("M", 2), ("M", 3), ("M", 4), ("M", 5), ("M", 7), ("M", 12), ("M", 13), ("M", 14), ("M", 15),
            #           ("M", 16), ("M", 17), ("M", 20), ("M", 23), ("M", 26), ("M", 40), ("M", 41), ("M", 62), ("M", 70), ("M", 71)]
            # num_dof = 9
            # show_M_matrix(num_dof, result_10, top_num_to_include_slice=slice(0,len(result_10)), save_dir=aug_plot_dir)
            # show_M_matrix(num_dof, result_20, top_num_to_include_slice=slice(0,len(result_20)), save_dir=aug_plot_dir)


            # from new_neuron_analysis.experiment_augment_input import read_all_data
            #
            # linear_global_dict, non_linear_global_dict, lagrangian_values, input_values, layers_values, all_weights =\
            #     read_all_data(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num,
            #               additional_note,
            #               num_layers=2)
            #
            # data_dir = get_data_dir(policy_env, policy_num_timesteps, policy_run_num, policy_seed,
            #                         eval_seed,
            #                         eval_run_num, additional_note=additional_note)
            # linear_lagrangian_to_include_in_state(linear_global_dict, data_dir,
            #                               lagrangian_values, layers_values, metric_param=metric_param)
