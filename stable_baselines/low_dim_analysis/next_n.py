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
from stable_baselines.low_dim_analysis.eval_util import get_full_params_dir, get_plot_dir, get_dir_path_for_this_run, \
    get_full_param_traj_file_path

from stable_baselines import logger

import pandas as pd
from sklearn.decomposition import PCA
from numpy import linalg as LA
from matplotlib import pyplot as plt
from stable_baselines.low_dim_analysis.common import calculate_projection_error, \
    calculate_num_axis_to_explain, plot_2d_check_index, get_allinone_concat_matrix_diff



def plot_distances(plot_next_n_dir,
                   distances,
                   show=False):

    fig = plt.figure()
    plt.plot(range(len(distances)), distances)
    plt.xlabel('update number')
    plt.ylabel('projection_errors')
    # plt.title('projection error to final PCA plane')
    # plt.clabel(CS1, inline=1, fontsize=6)
    # plt.clabel(CS2, inline=1, fontsize=6)
    fig.savefig(f"{plot_next_n_dir}/full_distances.pdf", dpi=300,
                bbox_inches='tight', format='pdf')
    if show: plt.show()





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
    #TODO save your shits to files!!!
    from stable_baselines.low_dim_analysis.common_parser import get_common_parser
    parser = get_common_parser()
    args = parser.parse_args()


    threads_or_None = 'threads' if args.use_threads else None
    logger.log(f"THREADS OR NOT: {threads_or_None}")

    plot_dir = get_plot_dir(args.alg, args.num_timesteps, args.env, args.normalize, args.run_num)

    plot_next_n_dir = f"{plot_dir}/next_n"
    this_run_dir = get_dir_path_for_this_run(args.alg, args.num_timesteps,
                                             args.env, args.normalize, args.run_num, args.n_steps, args.nminibatches)

    traj_params_dir_name = get_full_params_dir(this_run_dir)
    if os.path.exists(plot_next_n_dir):
        import shutil
        shutil.rmtree(plot_next_n_dir)
    os.makedirs(plot_next_n_dir)

    logger.log("grab final params")
    final_file = get_full_param_traj_file_path(traj_params_dir_name, "final")
    final_concat_params = pd.read_csv(final_file, header=None).values[0]


    tic = time.time()
    concat_matrix_diff = get_allinone_concat_matrix_diff(dir_name=traj_params_dir_name,
                                                         final_concat_params=final_concat_params)
    toc = time.time()
    print('\nElapsed time getting the full concat diff took {:.2f} s\n'
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

        # num_to_use, total_explained = calculate_num_axis_to_explain(pca, args.explain_ratio_threshold)

        proj_errors = calculate_projection_error(pca, concat_matrix_diff[check_index:], num_axis_to_use=args.n_comp_to_use)

        dump_my(plot_next_n_dir, proj_errors, f"all next proj_errors, check index: {check_index}")

        plot_2d_check_index(plot_next_n_dir, proj_errors,
                            f'proj errors',
                            f'proj errors to check_index {check_index}PCA using {args.n_comp_to_use} pca components, total variance explained {np.sum(pca.explained_variance_ratio_[:args.n_comp_to_use])}',
                            check_index=check_index, xlabel='update_number', show=False)

    final_pca = PCA(n_components=args.n_components)

    tic = time.time()
    final_pca.fit(concat_matrix_diff)
    toc = time.time()
    logger.log('\nElapsed time computing the full PCA {:.2f} s\n'
          .format(toc - tic))

    logger.log(final_pca.explained_variance_ratio_)


    logger.log(f"project all params")

    num_to_use, total_explained = calculate_num_axis_to_explain(final_pca, args.explain_ratio_threshold)
    proj_errors = calculate_projection_error(final_pca, concat_matrix_diff, num_axis_to_use=args.n_comp_to_use)
    dump_my(plot_next_n_dir, proj_errors, "full proj_errors")
    plot_2d_check_index(plot_next_n_dir, proj_errors,
                        f'proj errors',
                        f'proj errors to final PCA using {args.n_comp_to_use} pca components, total variance explained {np.sum(final_pca.explained_variance_ratio_[:args.n_comp_to_use])}',
                        check_index=None, xlabel='update_number', show=False)


    distances_traveled = get_distances(concat_matrix_diff)
    plot_distances(plot_next_n_dir, distances_traveled, show=False)
    dump_my(plot_next_n_dir, np.sum(distances_traveled), "total distance travelled")


    print("gc the big thing")
    del concat_matrix_diff
    import gc
    gc.collect()

