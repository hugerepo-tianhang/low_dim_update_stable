'''
1. calculate insofar PCA reconstruction
2. calculate reconstruction along opt path using the final PCA
3. repeat policy learning and viz/analysis from different init theta
4. repeat everything with 2d walker
5. figure out why TRPO doesn't work
'''

'''
My note:
Goal: Give a closer look at the trajectory. at each stage what is it like.
1. early stage, is it trying to look for the plane or it's right on the plane?
2. after found the plane, does it strictly follow the plane?
3. how did you find the plane?

How much does init pos, env influence the plane? How much does the land scape effect the time to find the plane.

ABout the plane:
1. is the plane a property of the
1.

'''
import numpy as np
from baselines.low_dim_analysis.eval_util import get_full_params_dir, get_plot_dir, get_dir_path_for_this_run, \
    get_full_param_traj_file_path

from baselines import logger

import pandas as pd
from sklearn.decomposition import IncrementalPCA, PCA
from numpy import linalg as LA
from matplotlib import pyplot as plt
from baselines.low_dim_analysis.common import calculate_projection_error, \
    calculate_num_axis_to_explain, plot_2d_check_index



def plot_distances(plot_dir_alg,
                   distances,
                   show=False):

    fig = plt.figure()
    plt.plot(range(len(distances)), distances)
    plt.xlabel('update number')
    plt.ylabel('projection_errors')
    # plt.title('projection error to final PCA plane')
    # plt.clabel(CS1, inline=1, fontsize=6)
    # plt.clabel(CS2, inline=1, fontsize=6)
    fig.savefig(f"{plot_dir_alg}/full_distances.pdf", dpi=300,
                bbox_inches='tight', format='pdf')
    if show: plt.show()



def get_concat_matrix_diff(dir_name, index, final_concat_params):
    theta_file = get_full_param_traj_file_path(dir_name, index)

    concat_df = pd.read_csv(theta_file, header=None)

    concat_matrix_diff = concat_df.sub(final_concat_params, axis='columns').values
    return concat_matrix_diff


def get_allinone_concat_matrix_diff(dir_name, final_concat_params):
    index = 0
    theta_file = get_full_param_traj_file_path(dir_name, index)

    concat_df = pd.read_csv(theta_file, header=None)

    result_matrix_diff = concat_df.sub(final_concat_params, axis='columns')

    index += 1

    while os.path.exists(get_full_param_traj_file_path(traj_params_dir_name, index)):
        theta_file = get_full_param_traj_file_path(dir_name, index)

        part_concat_df = pd.read_csv(theta_file, header=None)

        part_concat_df = part_concat_df.sub(final_concat_params, axis='columns')

        result_matrix_diff = result_matrix_diff.append(part_concat_df, ignore_index=True)
        index += 1

    return result_matrix_diff.values


import csv

from collections.abc import Iterable

def dump_my(dir_name, data, file_name):
    if not isinstance(data, Iterable):
        data = [data]

    var_output_file = f"{dir_name}/{file_name}.csv"

    with open(var_output_file, 'a') as fp:
        wr = csv.writer(fp)
        wr.writerow(data)

def get_distances(concat):

    return [LA.norm(concat[i+1] - concat[i]) for i in range(len(concat)-1)]



if __name__ == '__main__':

    import time
    import os

    from baselines.low_dim_analysis.common_parser import get_common_parser
    parser = get_common_parser()
    args = parser.parse_args()


    threads_or_None = 'threads' if args.use_threads else None
    logger.log(f"THREADS OR NOT: {threads_or_None}")

    plot_dir_alg = get_plot_dir(args.machine, args.alg, args.total_timesteps, args.env_id, args.run_num)
    traj_params_dir_name = get_full_params_dir(args.machine, args.alg, args.total_timesteps, args.env_id, args.run_num)

    if os.path.exists(plot_dir_alg):
        import shutil
        shutil.rmtree(plot_dir_alg)
    os.makedirs(plot_dir_alg)

    logger.log("grab final params")
    final_file = get_full_param_traj_file_path(traj_params_dir_name, "final")
    final_concat_params = pd.read_csv(final_file, header=None).values[0]


    tic = time.time()
    concat_matrix_diff = get_allinone_concat_matrix_diff(dir_name=traj_params_dir_name,
                                                         final_concat_params=final_concat_params)
    toc = time.time()
    print('\nElapsed time getting the PARALLEL full concat diff took {:.2f} s\n'
          .format(toc - tic))




    check_interval = len(concat_matrix_diff)//args.even_check_point_num
    check_points = [i for i in range(check_interval, len(concat_matrix_diff)-1, check_interval)]

    for check_index in check_points:
        pca = PCA(n_components=args.n_components)

        tic = time.time()
        pca.fit(concat_matrix_diff[:check_index])
        toc = time.time()
        logger.log('\nElapsed time computing the full PCA {:.2f} s\n'
              .format(toc - tic))

        logger.log(pca.explained_variance_ratio_)

        num_to_use, total_explained = calculate_num_axis_to_explain(pca, args.explain_ratio_threshold)
        proj_errors = calculate_projection_error(pca, concat_matrix_diff[check_index:], num_axis_to_use=num_to_use)

        dump_my(plot_dir_alg, proj_errors, f"all next proj_errors, check index: {check_index}")

        plot_2d_check_index(plot_dir_alg, proj_errors,
                            f'proj errors',
                            f'proj errors to check_index {check_index}PCA using {num_to_use} pca components, total variance explained {total_explained}',
                            check_index=check_index, xlabel='update_number', show=args.show)

    final_pca = PCA(n_components=args.n_components)

    tic = time.time()
    final_pca.fit(concat_matrix_diff)
    toc = time.time()
    logger.log('\nElapsed time computing the full PCA {:.2f} s\n'
          .format(toc - tic))

    logger.log(final_pca.explained_variance_ratio_)


    logger.log(f"project all params")

    num_to_use, total_explained = calculate_num_axis_to_explain(final_pca, args.explain_ratio_threshold)
    proj_errors = calculate_projection_error(final_pca, concat_matrix_diff, num_axis_to_use=num_to_use)
    dump_my(plot_dir_alg, proj_errors, "full proj_errors")
    plot_2d_check_index(plot_dir_alg, proj_errors,
                        f'proj errors',
                        f'proj errors to final PCA using {num_to_use} pca components, total variance explained {total_explained}',
                        check_index=None, xlabel='update_number', show=args.show)


    distances_traveled = get_distances(concat_matrix_diff)
    plot_distances(plot_dir_alg, distances_traveled, show=args.show)
    dump_my(plot_dir_alg, np.sum(distances_traveled), "total distance travelled")


    print("gc the big thing")
    del concat_matrix_diff
    import gc
    gc.collect()

