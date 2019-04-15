from numpy import linalg as LA
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA, PCA
import csv
from collections.abc import Iterable
from stable_baselines.low_dim_analysis.eval_util import *
from stable_baselines import logger
import time
from joblib import Parallel, delayed
import math
from functools import partial
from datetime import datetime




def postize_angle(angle):
    if angle > 90:
        angle = 180 - angle
    return angle


def get_run_num(file_func, **kargs):
    f = partial(file_func, **kargs)
    run_num = 0
    while os.path.exists(f(run_num=run_num)):
        run_num += 1
    return run_num


def generate_run_dir(get_cma_returns_dirname, **kargs):

    cma_run_num = get_run_num(get_cma_returns_dirname, **kargs)
    cma_intermediate_data_dir = get_cma_returns_dirname(run_num=cma_run_num, **kargs)
    if os.path.exists(cma_intermediate_data_dir):
        import shutil
        shutil.rmtree(cma_intermediate_data_dir)
    os.makedirs(cma_intermediate_data_dir)

    return cma_run_num, cma_intermediate_data_dir
def gen_subspace_coords(plot_args, proj_coord):
    proj_xcoord, proj_ycoord = proj_coord[0], proj_coord[1]

    xmin, xmax = np.min(proj_xcoord), np.max(proj_xcoord)
    ymin, ymax = np.min(proj_ycoord), np.max(proj_ycoord)

    x_len = xmax - xmin
    y_len = ymax - ymin
    assert(x_len>=0)
    assert(y_len>=0)

    xmin -= plot_args.padding_fraction * x_len
    xmax += plot_args.padding_fraction * x_len
    ymin -= plot_args.padding_fraction * y_len
    ymax += plot_args.padding_fraction * y_len

    xcoordinates_to_eval = np.linspace(xmin, xmax, plot_args.xnum)
    ycoordinates_to_eval = np.linspace(ymin, ymax, plot_args.ynum)

    np.insert(xcoordinates_to_eval, xcoordinates_to_eval.searchsorted(0), 0)
    np.insert(ycoordinates_to_eval, ycoordinates_to_eval.searchsorted(0), 0)

    return xcoordinates_to_eval, ycoordinates_to_eval

# def do_proj_(concat_matrix_diff, first_n_pcs, intermediate_data_dir, mean_param, origin="final_param"):
#     logger.log(f"project all params")
#     proj_xcoord, proj_ycoord = [], []
#     for param in concat_matrix_diff:
#         x, y = project_2D_final_param_origin(d=param, dx=first_n_pcs[0], dy=first_n_pcs[1])
#
#         proj_xcoord.append(x)
#         proj_ycoord.append(y)
#
#     proj_coords = np.array([proj_xcoord, proj_ycoord])
#
#     return proj_coords


def do_proj_on_first_n(all_param_matrix, first_n_pcs, origin_param):
    components = first_n_pcs
    if len(all_param_matrix.shape) == 1:
        all_param_matrix = all_param_matrix.reshape(1, -1)


    proj_coords = (all_param_matrix - origin_param).dot(components.T)
    return proj_coords

def do_proj_on_first_n_IPCA(concat_df, first_n_pcs, origin_param):
    # IPCA
    assert isinstance(concat_df, pd.io.parsers.TextFileReader)
    first_chunk = concat_df.__next__()
    result = do_proj_on_first_n(first_chunk.values, first_n_pcs, origin_param)

    for chunk in concat_df:
        result = np.vstack((result, do_proj_on_first_n(chunk.values, first_n_pcs, origin_param)))

    return result


def do_eval_returns(plot_args, intermediate_data_dir, first_n_pcs, origin_param,
                    xcoordinates_to_eval, ycoordinates_to_eval, save_dir, pca_center="final_param", reuse=True):

    eval_string = f"xnum_{np.min(xcoordinates_to_eval)}:{np.max(xcoordinates_to_eval)}:{plot_args.xnum}_" \
                    f"ynum_{np.min(ycoordinates_to_eval)}:{np.max(ycoordinates_to_eval)}:{plot_args.ynum}"

    if not reuse and not os.path.exists(get_eval_returns_filename(intermediate_dir=intermediate_data_dir,
                                                    eval_string=eval_string, n_comp=2, pca_center=pca_center)):

        from stable_baselines.ppo2.run_mujoco import eval_return
        thetas_to_eval = [origin_param + x * first_n_pcs[0] + y * first_n_pcs[1] for y in ycoordinates_to_eval for x in xcoordinates_to_eval]

        tic = time.time()

        eval_returns = Parallel(n_jobs=plot_args.cores_to_use, max_nbytes='100M')\
            (delayed(eval_return)(plot_args, save_dir, theta, plot_args.eval_num_timesteps, i) for (i, theta) in enumerate(thetas_to_eval))
        toc = time.time()
        logger.log(f"####################################1st version took {toc-tic} seconds")


        np.savetxt(get_eval_returns_filename(intermediate_dir=intermediate_data_dir,
                                             eval_string=eval_string, n_comp=2, pca_center=pca_center), eval_returns, delimiter=',')
    else:
        eval_returns = np.loadtxt(get_eval_returns_filename(intermediate_dir=intermediate_data_dir,
                                                            eval_string=eval_string, n_comp=2, pca_center=pca_center), delimiter=',')

    return eval_returns


def do_pca(n_components, n_comp_to_use, traj_params_dir_name,
           intermediate_data_dir, proj, origin="final_param", use_IPCA=False, chunk_size=None, reuse=True):
    logger.log("grab final params")
    final_file = get_full_param_traj_file_path(traj_params_dir_name, "final")
    final_concat_params = pd.read_csv(final_file, header=None).values[0]
    proj_coords = None
    if not reuse or \
            not os.path.exists(get_pcs_filename(intermediate_dir=intermediate_data_dir, n_comp=n_components))\
        or not os.path.exists(get_mean_param_filename(intermediate_dir=intermediate_data_dir)) \
        or (proj and not os.path.exists(get_projected_full_path_filename(intermediate_dir=intermediate_data_dir,
                                                                         n_comp=n_components, pca_center=origin))):
        if use_IPCA:
            assert chunk_size != 0
            final_pca = IncrementalPCA(n_components=n_components)  # for sparse PCA to speed up

            tic = time.time()
            concat_df = get_allinone_concat_df(dir_name=traj_params_dir_name,
                                               use_IPCA=use_IPCA, chunk_size=chunk_size)
            toc = time.time()
            print('\nElapsed time getting the chunk concat diff took {:.2f} s\n'
                  .format(toc - tic))

            tic = time.time()
            for chunk in concat_df:
                logger.log(f"currnet at : {concat_df._currow}")

                if chunk.shape[0] < n_components:
                    logger.log(f"last column too few: {chunk.shape[0]}")
                    continue
                final_pca.partial_fit(chunk)

            toc = time.time()
            logger.log('\nElapsed time computing the chunked PCA {:.2f} s\n'
                  .format(toc - tic))

        else:
            tic = time.time()
            concat_df = get_allinone_concat_df(dir_name=traj_params_dir_name)
            concat_matrix = concat_df.values

            toc = time.time()
            print('\nElapsed time getting the full concat diff took {:.2f} s\n'
                  .format(toc - tic))



            final_pca = PCA(n_components=n_components) # for sparse PCA to speed up

            tic = time.time()
            final_pca.fit(concat_matrix)
            toc = time.time()
            logger.log('\nElapsed time computing the full PCA {:.2f} s\n'
                  .format(toc - tic))

        logger.log(final_pca.explained_variance_ratio_)

        pcs_components = final_pca.components_

        first_n_pcs = pcs_components[:n_comp_to_use]
        mean_param = final_pca.mean_
        explained_variance_ratio = final_pca.explained_variance_ratio_


        np.savetxt(get_pcs_filename(intermediate_dir=intermediate_data_dir, n_comp=n_components), pcs_components, delimiter=',')
        np.savetxt(get_mean_param_filename(intermediate_dir=intermediate_data_dir), mean_param, delimiter=',')
        np.savetxt(get_explain_ratios_filename(intermediate_dir=intermediate_data_dir, n_comp=n_components),
                   explained_variance_ratio, delimiter=',')

        if origin == "final_param":
            origin_param = final_concat_params
        else:
            origin_param = mean_param

        if proj:
            if use_IPCA:
                concat_df = get_allinone_concat_df(dir_name=traj_params_dir_name,
                                                   use_IPCA=use_IPCA, chunk_size=chunk_size)
                proj_coords = do_proj_on_first_n_IPCA(concat_df, first_n_pcs, origin_param)

            else:
                proj_coords = do_proj_on_first_n(concat_matrix, first_n_pcs, origin_param)

            np.savetxt(get_projected_full_path_filename(intermediate_dir=intermediate_data_dir, n_comp=n_components,
                                                            pca_center=origin),
                           proj_coords, delimiter=',')

        print("gc the big thing")
        del concat_df
        # if not use_IPCA:
        #     del concat_matrix_diff
        import gc
        gc.collect()
    else:
        pcs_components = np.loadtxt(
            get_pcs_filename(intermediate_dir=intermediate_data_dir, n_comp=n_components), delimiter=',')
        first_n_pcs = pcs_components[:n_comp_to_use]
        mean_param = np.loadtxt(get_mean_param_filename(intermediate_dir=intermediate_data_dir), delimiter=',')
        explained_variance_ratio = np.loadtxt(get_explain_ratios_filename(intermediate_dir=intermediate_data_dir, n_comp=n_components),
                   delimiter=',')
        if proj:
            proj_coords = np.loadtxt(get_projected_full_path_filename(intermediate_dir=intermediate_data_dir, n_comp=n_components, pca_center=origin), delimiter=',')


    result = {
        "pcs_components": pcs_components,
        "first_n_pcs": first_n_pcs,
        "mean_param":mean_param,
        "final_concat_params":final_concat_params,
        "explained_variance_ratio":explained_variance_ratio,
        "proj_coords":proj_coords
    }
    return result

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def cal_angle(v1, v2):

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return math.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def cal_angle_plane(V, pcs):
    projected = get_projected_vector_in_old_basis(V, pcs, len(pcs))
    return cal_angle(projected, V)

def plane_inner_product(pcs1, pcs2):
    if len(pcs1.shape) == 1:
        pcs1 = pcs1.reshape(1,-1)

    if len(pcs2.shape) == 1:
        pcs2 = pcs2.reshape(1,-1)

    mat = []
    for i in range(len(pcs1)):
        row = []
        for j in range(len(pcs2)):
            row.append(np.dot(pcs1[i], pcs2[j]))
        mat.append(row)

    return np.linalg.det(mat)

def plane_inner_product_2d(pcs1, pcs2):
    return np.dot(pcs1[0], pcs2[0]) * np.dot(pcs1[1], pcs2[1]) - np.dot(pcs1[1], pcs2[0]) * np.dot(pcs1[0], pcs2[1])

def cal_angle_between_2_2d_planes(pcs1, pcs2):
    i = plane_inner_product_2d(pcs1, pcs2)
    np.testing.assert_almost_equal(i, plane_inner_product(pcs1, pcs2))

    cos = i/np.sqrt(plane_inner_product_2d(pcs1, pcs1)*plane_inner_product_2d(pcs2, pcs2))
    return math.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

def cal_angle_between_nd_planes(pcs1, pcs2):
    cos = plane_inner_product(pcs1, pcs2)/np.sqrt(plane_inner_product(pcs1, pcs1)*plane_inner_product(pcs2, pcs2))
    return math.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def get_projected_vector_in_old_basis(vector, all_pcs, num_axis_to_use):
    components = all_pcs[:num_axis_to_use]

    projected_vector = vector.dot(components.T) #(final * compT - start * compT )

    # goes back to original data space with mean restored.
    projected_data_in_old_basis = projected_vector.dot(components) #(final * compT * comp - start * compT * comp )

    return projected_data_in_old_basis

def low_dim_to_old_basis(projected_data, new_axises, origin_param):
    projected_data_in_old_basis = projected_data.dot(new_axises) + origin_param
    return projected_data_in_old_basis

def get_projected_data_in_old_basis(origin_param, all_pcs, data, num_axis_to_use):
    components = all_pcs[:num_axis_to_use]

    projected_data = (data - origin_param).dot(components.T)

    # goes back to original data space with mean restored.
    projected_data_in_old_basis = low_dim_to_old_basis(projected_data, components, origin_param)
    return projected_data_in_old_basis

def calculate_projection_errors(mean_param, all_pcs, data, num_axis_to_use):
    projected_data_in_old_basis = get_projected_data_in_old_basis(mean_param, all_pcs, data, num_axis_to_use)
    # assert_array_almost_equal(X_projected, X_projected2)
    # losses = LA.norm((data - projected_data_in_old_basis), ord=2, axis=1) #TODO: check this
    losses = []
    if len(data.shape) == 1:
        data = data.reshape(1,-1)

    for d in (data - projected_data_in_old_basis):
        losses.append(LA.norm(d, ord=2))
    return losses


def calculate_num_axis_to_explain(pca, ratio_threshold):
    num = 0
    total_explained = 0
    while total_explained < ratio_threshold:
        total_explained += pca.explained_variance_ratio_[num]
        num += 1

    return num, total_explained


def plot_2d_check_index(plot_dir_alg, data, ylabel, file_name, check_index=None, xlabel='update_number', show=False):

    fig = plt.figure()
    plt.plot(range(len(data)), data)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(file_name)
    if check_index is not None:
        fig.savefig(f"{plot_dir_alg}/{file_name}_check_index_{check_index}.pdf", dpi=300,
                bbox_inches='tight', format='pdf')
    else:
        fig.savefig(f"{plot_dir_alg}/{file_name}.pdf", dpi=300,
                bbox_inches='tight', format='pdf')
    if show: plt.show()


def dump_row_write_csv(dir_name, data, file_name):
    if not isinstance(data, Iterable):
        data = [data]

    var_output_file = f"{dir_name}/{file_name}.csv"

    with open(var_output_file, 'w') as fp:
        wr = csv.writer(fp)
        wr.writerow(data)

def dump_row_append_csv(dir_name, data, file_name):
    if not isinstance(data, Iterable):
        data = [data]

    var_output_file = f"{dir_name}/{file_name}.csv"

    with open(var_output_file, 'a') as fp:
        wr = csv.writer(fp)
        wr.writerow(data)

def dump_rows_append_csv(dir_name, data, file_name):
    for row in data:
        dump_row_append_csv(dir_name, row, file_name)

def dump_rows_write_csv(dir_name, data, file_name):
    var_output_file = f"{dir_name}/{file_name}.csv"
    if os.path.isfile(var_output_file):
        os.remove(var_output_file)

    for row in data:
        dump_row_append_csv(dir_name, row, file_name)



def project_1D(w, d):
    assert len(w) == len(d), 'dimension does not match for w and '
    scale = np.dot(w, d) / LA.norm(d, 2)
    return scale





def project_2D_pca_mean_origin(d, components, pca_mean):

        # this is actually actually shifted projection
        # real projection should be (d - X_train.mean(0)).dot(A)

    [x,y] = (d - pca_mean).dot(components.T)

    return x, y


def project_2D_final_param_origin(d, dx, dy, proj_method='lstsq'):
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


def plot_contour_trajectory(plot_dir_alg, name, xcoordinates, ycoordinates, Z, proj_xcoord, proj_ycoord, explained_variance_ratio,
                            num_levels=40, show=False, sub_alg_path=None):
    """2D contour + trajectory"""

    X, Y = np.meshgrid(xcoordinates, ycoordinates)
    Z = np.array(Z)
    Z = Z.reshape(X.shape)
    fig = plt.figure()
    vmin = np.min(Z)
    vmax = np.max(Z)
    vlevel = (vmax - vmin)/num_levels
    CS1 = plt.contour(X, Y, Z, levels=np.arange(vmin, vmax, vlevel))
    cbar = plt.colorbar()
    # plot trajectories
    plt.plot(proj_xcoord, proj_ycoord, marker='.')

    # if cma_path is not None:
    #     plt.plot(cma_path[0], cma_path[1], 'bo')
    if sub_alg_path is not None:
        assert sub_alg_path.shape[1] == 2
        plt.plot(sub_alg_path[0], sub_alg_path[1], marker='8')


    plt.annotate('end', xy=(proj_xcoord[-1], proj_ycoord[-1]), xytext=(proj_xcoord[-1] + 0.2, proj_ycoord[-1] +0.2),
                arrowprops=dict(facecolor='black', shrink=0.05),
                )

    plt.annotate('sub alg end', xy=(sub_alg_path[0][-1], sub_alg_path[1][-1]), xytext=(sub_alg_path[0][-1] + 0.2, sub_alg_path[1][-1] +0.2),
                arrowprops=dict(facecolor='blue', shrink=0.05),
                )

    plt.annotate('sub alg start', xy=(sub_alg_path[0][0], sub_alg_path[1][0]),
                 xytext=(sub_alg_path[0][0] + 0.2, sub_alg_path[1][0] + 0.2),
                 arrowprops=dict(facecolor='blue', shrink=0.05),
                 )
    # plot red points when learning rate decays
    # for e in [150, 225, 275]:
    #     plt.plot([pf['proj_xcoord'][e]], [pf['proj_ycoord'][e]], marker='.', color='r')

    # add PCA notes
    ratio_x, ratio_y = explained_variance_ratio
    plt.xlabel('1st PC: %.2f %%' % (ratio_x*100), fontsize='xx-large')
    plt.ylabel('2nd PC: %.2f %%' % (ratio_y*100), fontsize='xx-large')
    plt.clabel(CS1, inline=1, fontsize=6)
    print(f"~~~~~~~~~~~~~~~~~~~~~~saving to {plot_dir_alg}/{name}.pdf")
    file_path = f"{plot_dir_alg}/{name}.pdf"
    if os.path.isfile(file_path):
        os.remove(file_path)
    fig.savefig(file_path, dpi=300,
                bbox_inches='tight', format='pdf')
    if show: plt.show()



def plot_2d(plot_dir_alg, name, X, Y, xlabel, ylabel, show):


    fig, ax = plt.subplots()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    ax.plot(X, Y)
    file_path = f"{plot_dir_alg}/{name}.pdf"
    if os.path.isfile(file_path):
        os.remove(file_path)
    logger.log(f"####saving to {file_path}")
    fig.savefig(file_path, dpi=300,
                bbox_inches='tight', format='pdf')
    if show: plt.show()



def plot_3d_trajectory(plot_dir_alg, name, xcoordinates, ycoordinates, Z, proj_xcoord, proj_ycoord, explained_variance_ratio,
                            num_levels=15, show=False):
    """3d + trajectory"""

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter


    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(xcoordinates, ycoordinates)
    Z = np.array(Z)
    Z = Z.reshape(X.shape)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8)
    # Add a color bar which maps values to colors.

    fig.colorbar(surf, shrink=0.5, aspect=5)

    zero_plane = np.min(Z) - 100
    contour_projection = ax.contour(X, Y, Z, zdir='z', offset=zero_plane, cmap=cm.coolwarm)
    opt_path = ax.plot(proj_xcoord, proj_ycoord, zs=zero_plane, zdir='z', label='optimization path', linewidth=3.5)
    ax.legend()
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ratio_x, ratio_y = explained_variance_ratio
    plt.xlabel('1st PC: %.2f %%' % (ratio_x*100), fontsize='xx-large')
    plt.ylabel('2nd PC: %.2f %%' % (ratio_y*100), fontsize='xx-large')

    print(f"~~~~~~~~~~~~~~~~~~~~~~saving to {plot_dir_alg}/{name}.pdf")
    file_path = f"{plot_dir_alg}/{name}.pdf"
    if os.path.isfile(file_path):
        os.remove(file_path)
    fig.savefig(file_path, dpi=300,
                bbox_inches='tight', format='pdf')
    if show: plt.show()


def get_allinone_concat_df(dir_name, use_IPCA=False, chunk_size=None, index = 0, skip_rows=None):

    theta_file = get_full_param_traj_file_path(dir_name, index)
    if use_IPCA:
        assert chunk_size is not None
        assert chunk_size != 0
        concat_df = pd.read_csv(theta_file, header=None, chunksize=chunk_size, skip_rows=skip_rows)
    else:
        concat_df = pd.read_csv(theta_file, header=None, skip_rows=skip_rows)


    return concat_df



if __name__ == "__main__":
    print(get_current_timestamp())
    # from numpy.testing import assert_array_almost_equal
    #
    # X_train = np.random.randn(100, 50)
    #
    # pca = PCA(n_components=2)
    # test_pca = PCA(n_components=5)
    # pca.fit(X_train)
    # test_pca.fit(X_train)
    #
    #
    # X_train_pca = pca.transform(X_train)
    # X_train_pca2 = (X_train - test_pca.mean_).dot(test_pca.components_[:2].T)
    #
    # assert_array_almost_equal(X_train_pca, X_train_pca2)
    #
    # X_projected = pca.inverse_transform(X_train_pca)
    # X_projected2 = X_train_pca.dot(test_pca.components_[:2]) + test_pca.mean_
    # k = get_projected_data_in_old_basis(test_pca, X_train, num_axis_to_use=2)
    # assert_array_almost_equal(X_projected, k)
    #
    # # assert_array_almost_equal(X_projected, X_projected2)
    # a = calculate_projection_error(test_pca, X_train, num_axis_to_use=2)
    # loss = LA.norm((X_train - X_projected2), ord=2, axis=1)
    # assert_array_almost_equal(a, loss)

