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
from baselines.low_dim_analysis.eval_util import get_full_params_dir, get_plot_dir, get_dir_path_for_this_run, get_full_param_traj_file_path


from baselines import logger

import pandas as pd
from sklearn.decomposition import IncrementalPCA, PCA
from numpy import linalg as LA
from matplotlib import pyplot as plt





def calculate_projection_loss(pca, data_point):
    # project CENTERED data to it's components( which origin is at center of data )
    # X_train_pca2 = (X_train - X_train.mean(0)).dot(pca.components_.T)
    X_train_pca = pca.transform(data_point.reshape((1, -1)))

    # goes back to original data space with mean restored.
    # X_projected2 = X_train_pca.dot(pca.components_) + pca.mean_
    X_projected_in_old_basis = pca.inverse_transform(X_train_pca)

    # assert_array_almost_equal(X_projected, X_projected2)

    loss = np.sum((data_point - X_projected_in_old_basis) ** 2)
    return loss


def cal_angle(vec1, vec2):
    return np.ndarray.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def project_1D(w, d):
    assert len(w) == len(d), 'dimension does not match for w and '
    scale = np.dot(w, d) / LA.norm(d, 2)
    return scale

def project_2D(d, dx, dy, proj_method):
    if proj_method == 'cos':
        # when dx and dy are orthorgonal
        x = project_1D(d, dx)
        y = project_1D(d, dy)
    elif proj_method == 'lstsq':
        # this is actually actually shifted projection
        # real projection should be (d - X_train.mean(0)).dot(A)
        A = np.vstack([dx, dy]).T
        [x,y] = d.dot(A)

    return x, y



def generate_coords(xmin, xmax, xnum, ymin, ymax, ynum, y=True):
    # Create the coordinates(resolutions) at which the function is evaluated
    xcoordinates = np.linspace(xmin, xmax, xnum)
    ycoordinates = np.linspace(ymin, ymax, ynum)

    return xcoordinates, ycoordinates


def plot_contour_trajectory(plot_dir_alg, proj_xcoord, proj_ycoord, explained_variance_ratio, show=False):
    """2D contour + trajectory"""

    fig = plt.figure()

    # plot trajectories
    plt.plot(proj_xcoord, proj_ycoord, marker='.')

    # plot red points when learning rate decays
    # for e in [150, 225, 275]:
    #     plt.plot([pf['proj_xcoord'][e]], [pf['proj_ycoord'][e]], marker='.', color='r')

    # add PCA notes
    ratio_x, ratio_y = explained_variance_ratio
    plt.xlabel('1st PC: %.2f %%' % (ratio_x*100), fontsize='xx-large')
    plt.ylabel('2nd PC: %.2f %%' % (ratio_y*100), fontsize='xx-large')

    fig.savefig(f"{plot_dir_alg}/projected_traj.pdf", dpi=300,
                bbox_inches='tight', format='pdf')
    if show: plt.show()

def get_local_mins(l, indexes):
    ms = []
    for i in indexes:

        if l[i] < l[i-1] and l[i] < l[i+1]:
            ms.append(i)
    return ms

def get_first_local_min_of_local_mins(proj_errors):
    indexes = range(1,len(proj_errors)-1)
    local_mins = get_local_mins(proj_errors, indexes)


    min_of_mins = get_local_mins(proj_errors, local_mins)
    return min_of_mins[0]


def plot_pca_vs_ipca(plot_dir_alg, pca_ipca, show=False):



    fig = plt.figure()
    plt.plot(range(len(pca_ipca)), pca_ipca)

    plt.xlabel('update number')
    plt.ylabel('pca - ipca')
    plt.title('pca - ipca')

    fig.savefig(f"{plot_dir_alg}/pca-ipca.pdf", dpi=300,
                bbox_inches='tight', format='pdf')
    if show: plt.show()


def plot_trajectory_wtih_num_to_explain_only(plot_dir_alg, proj_xcoord, proj_ycoord,
                                             num_to_explain,
                                             explained_variance_ratio, show=False):

    fig = plt.figure()
    plt.plot(range(len(proj_xcoord)), num_to_explain)
    plt.xlabel('update number')
    plt.ylabel('number of components')
    plt.title('number of components to explain 90% variance')

    fig.savefig(f"{plot_dir_alg}/to_explain_90percent_variance.pdf", dpi=300,
                bbox_inches='tight', format='pdf')
    if show: plt.show()


def gen_name_traj(xmin, xmax, xnum, ymin, ymax, ynum, total_timesteps):
    return f"traj_only_{xmin}:{xmax}:{xnum}:{ymin}:{ymax}:{ynum}_{total_timesteps}"

def gen_name_error(total_timesteps):
    return f"error_only__{total_timesteps}"


def get_concat_matrix_diff(dir_name, index, final_concat_params):
    theta_file = get_full_param_traj_file_path(dir_name,  index)

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



def projection(concat_matrix_diff, dx, dy, pca):
    proj_xcoord, proj_ycoord = [], []
    proj_errors = []

    for param in concat_matrix_diff:
        d = param
        x, y = project_2D(d, dx, dy, proj_method='lstsq')

        proj_xcoord.append(x)
        proj_ycoord.append(y)
        proj_errors.append(calculate_projection_loss(pca, param))

    return proj_xcoord, proj_ycoord, proj_errors

def projection_parallel(thread_or_None, concat_matrix_diff, dx, dy, pca):
    #TODO this returns mmap. so list it before use. but just seperate projection with stepwise PCA
    folder = f"{this_run_dir_path}/joblib_memmap"

    total_num_of_points = len(concat_matrix_diff)

    import shutil

    if os.path.exists(folder):
        shutil.rmtree(folder)

    try:
        os.mkdir(folder)
        print(f"MEMMAP DIR{folder}")
    except FileExistsError:
        raise Exception("RACING CONDITION!!!!!!!!!! IT DIDN'T EXIST")



    data_filename_memmap = os.path.join(folder, 'data_memmap')

    dump(concat_matrix_diff, data_filename_memmap)
    concat_matrix_diff = load(data_filename_memmap, mmap_mode='r')

    # print("deleting original big matrix")
    # del concat_matrix_diff
    # import gc
    # _ = gc.collect()


    proj_xcoord_memmap = os.path.join(folder, 'proj_xcoord')
    proj_xcoord = np.memmap(proj_xcoord_memmap, dtype=float,
                       shape=total_num_of_points, mode='w+')
    proj_ycoord_memmap = os.path.join(folder, 'proj_ycoord')
    proj_ycoord = np.memmap(proj_ycoord_memmap, dtype=float,
                       shape=total_num_of_points, mode='w+')
    proj_errors_memmap = os.path.join(folder, 'proj_errors')
    proj_errors = np.memmap(proj_errors_memmap, dtype=float,
                       shape=total_num_of_points, mode='w+')


    def proj_param(concat_matrix_diff, dx, dy, pca, proj_xcoord, proj_ycoord, proj_errors, index):
        print(f"proj_param index: {index}")
        print("[Worker %d] for index %d" % (os.getpid(), index))
        param = concat_matrix_diff[index]

        x, y = project_2D(param, dx, dy, proj_method='lstsq')
        print(x)
        proj_xcoord[index] = x
        proj_ycoord[index] = y
        proj_errors[index] = calculate_projection_loss(pca, param)


    tic = time.time()
    Parallel(n_jobs=args.cores_to_use, prefer=thread_or_None)(
        delayed(proj_param)(concat_matrix_diff, dx, dy, pca, proj_xcoord, proj_ycoord, proj_errors, index)
        for index in range(total_num_of_points))
    toc = time.time()
    print('\nElapsed time computing projection {:.2f} s\n'
          .format(toc - tic))
    print("proj:")
    print(proj_xcoord)

    # try:
    #     shutil.rmtree(folder)
    # except:  # noqa
    #     print('Could not clean-up automatically.')


    return proj_xcoord, proj_ycoord, proj_errors


def calc_insofar_pca_explain_num(first_n, milestone_IPCA):
    # IPCA in bigger chunks
    pca = IncrementalPCA(n_components=n_components)
    index = 0
    while os.path.exists(get_full_param_traj_file_path(traj_params_dir_name, index)):
        logger.log(f"IPCA params index: {index}")
        concat_matrix_diff = get_concat_matrix_diff(traj_params_dir_name, index, final_concat_params)
        pca.partial_fit(concat_matrix_diff)
        index += 1

    pc1 = np.array(pca.components_[0])
    pc2 = np.array(pca.components_[1])
    print(cal_angle(pc1, pc2))
    print(pca.explained_variance_ratio_)



def calculate_num_axis_to_explain(pca, ratio_threshold):
    num = 1
    total_explained = 0
    while total_explained < ratio_threshold:
        total_explained += pca.explained_variance_ratio_[num - 1]
        num += 1

    return num

if __name__ == '__main__':
    '''
    plot args: --alg: ppo2_eval --load_path: ... 
    '''
    from copy import deepcopy
    import time
    import os
    from joblib import dump, load
    from joblib import Parallel, delayed

    import argparse
    parser = argparse.ArgumentParser(description='load and pca')

    # PCA parameters
    parser.add_argument('--machine', default='dev', help='dev machine is my laptop, prod is berlin.cc.gatech.edu')
    parser.add_argument('--cores_to_use', default=8, type=int, help='cores to use to parallel')
    parser.add_argument('--alg', default='ppo2', help='algorithm to train on')
    parser.add_argument('--env_id', default='Hopper-v2', help='algorithm to train on')
    parser.add_argument('--total_timesteps', default=10000, type=int, help='total timesteps agent runs')
    parser.add_argument('--run_num', default=0, type=int, help='which run number')
    parser.add_argument('--use_IPCA', action='store_true', default=False)
    parser.add_argument('--use_threads', action='store_true', default=False)


    parser.add_argument('--n_components', default=10, type=int, help='n_components of PCA')
    parser.add_argument('--explain_ratio_threshold', default=0.9, type=float)

    # plot parameters
    parser.add_argument('--x', default='-1:1:51', help='A string with format xmin:x_max:xnum')
    parser.add_argument('--y', default='-1:1:51', help='A string with format ymin:ymax:ynum')
    parser.add_argument('--num_levels', default='15', help='number of levels on the contour plot')
    parser.add_argument('--proj_file', default='', help='the .h5 file contains projected optimization trajectory.')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    parser.add_argument('--show', action='store_true', default=False, help='show plotted figures')
    parser.add_argument('--log', action='store_true', default=False, help='use log scale for loss values')
    parser.add_argument('--plot', action='store_true', default=False, help='plot figures after computation')
    parser.add_argument('--surf_file', default='', help='customize the name of surface file, could be an existing file.')
    args = parser.parse_args()


    threads_or_None = 'threads' if args.use_threads else None
    print(f"THREADS OR NOT: {threads_or_None}")

    n_components = args.n_components
    args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
    args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]

    plot_dir_alg = get_plot_dir(args.machine, args.alg, args.total_timesteps, args.env_id, args.run_num)
    traj_params_dir_name = get_full_params_dir(args.machine, args.alg, args.total_timesteps, args.env_id, args.run_num)


    if not os.path.exists(plot_dir_alg):
        os.makedirs(plot_dir_alg)


    logger.log("grab final params")
    final_file = get_full_param_traj_file_path(traj_params_dir_name, "final")
    final_concat_params = pd.read_csv(final_file, header=None).values[0]



    import csv

    def dump(dir_name, data, file_name):
        var_output_file = f"{dir_name}/{file_name}.csv"

        with open(var_output_file, 'a') as fp:
            wr = csv.writer(fp)
            wr.writerow(data)
    '''
    ==========================================
    IPCA all data
    ===========================================
    '''
    tic = time.time()

    ipca = IncrementalPCA(n_components=n_components)
    index = 0
    while os.path.exists(get_full_param_traj_file_path(traj_params_dir_name, index)):
        logger.log(f"IPCA params index: {index}")
        concat_matrix_diff = get_concat_matrix_diff(traj_params_dir_name, index, final_concat_params)
        ipca.partial_fit(concat_matrix_diff)
        index += 1

    toc = time.time()
    print('\nElapsed time computing the IPCA {:.2f} s\n'.format(toc - tic))

    pc1 = np.array(ipca.components_[0])
    pc2 = np.array(ipca.components_[1])
    print(ipca.explained_variance_ratio_)



    pca = PCA(n_components=n_components)
    concat_matrix_diff = get_allinone_concat_matrix_diff(dir_name=traj_params_dir_name,
                                                         final_concat_params=final_concat_params)
    tic = time.time()
    pca.fit(concat_matrix_diff)
    toc = time.time()
    print('\nElapsed time computing the full PCA {:.2f} s\n'
          .format(toc - tic))

    pca_vs_ipca = pca.explained_variance_ratio_-ipca.explained_variance_ratio_

    dump(plot_dir_alg, pca_vs_ipca, "pca-ipca explained_ratios")
    dump(plot_dir_alg, [LA.norm(pca.explained_variance_ratio_-ipca.explained_variance_ratio_, 2)], "2 norm of pca-ipca explained_ratios")

    dump(plot_dir_alg, np.cumsum(pca.explained_variance_ratio_), "cum_sum PCA explained_variance")
    dump(plot_dir_alg, np.cumsum(ipca.explained_variance_ratio_), "cum_sum IPCA explained_variance")


    plot_pca_vs_ipca(plot_dir_alg, pca_vs_ipca, show=args.show)
