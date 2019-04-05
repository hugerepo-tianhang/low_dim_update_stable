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

def do_proj_(concat_matrix_diff, first_n_pcs, intermediate_data_dir, mean_param, origin="final_param"):
    logger.log(f"project all params")
    proj_xcoord, proj_ycoord = [], []
    for param in concat_matrix_diff:
        x, y = project_2D_final_param_origin(d=param, dx=first_n_pcs[0], dy=first_n_pcs[1])

        proj_xcoord.append(x)
        proj_ycoord.append(y)

    proj_coords = np.array([proj_xcoord, proj_ycoord])

    return proj_coords

def do_proj(concat_matrix_diff, first_n_pcs, intermediate_data_dir, mean_param, origin="final_param"):
    components = first_n_pcs[:2]
    if "final_param" == origin:
        proj_coords = concat_matrix_diff.dot(components.T)
    else:
        proj_coords = (concat_matrix_diff - mean_param).dot(components.T)
    return proj_coords.T

def do_eval_returns(plot_args, intermediate_data_dir, first_n_pcs, origin_param, xcoordinates_to_eval, ycoordinates_to_eval, save_dir, pca_center="final_param"):

    eval_string = f"xnum_{np.min(xcoordinates_to_eval)}:{np.max(xcoordinates_to_eval)}:{plot_args.xnum}_" \
                    f"ynum_{np.min(ycoordinates_to_eval)}:{np.max(ycoordinates_to_eval)}:{plot_args.ynum}"

    if not os.path.exists(get_eval_returns_filename(intermediate_dir=intermediate_data_dir,
                                                    eval_string=eval_string, n_comp=2, pca_center=pca_center)):

        from stable_baselines.ppo2.run_mujoco import eval_return

        tic = time.time()
        thetas_to_eval = [origin_param + x * first_n_pcs[0] + y * first_n_pcs[1] for y in ycoordinates_to_eval for x in xcoordinates_to_eval]

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


def do_pca(n_components, n_comp_to_use, traj_params_dir_name, intermediate_data_dir, proj, origin="final_param"):
    logger.log("grab final params")
    final_file = get_full_param_traj_file_path(traj_params_dir_name, "final")
    final_concat_params = pd.read_csv(final_file, header=None).values[0]
    proj_coords = None
    if not os.path.exists(get_pcs_filename(intermediate_dir=intermediate_data_dir, n_comp=n_components))\
        or not os.path.exists(get_mean_param_filename(intermediate_dir=intermediate_data_dir)) \
        or (proj and not os.path.exists(get_projected_full_path_filename(intermediate_dir=intermediate_data_dir, n_comp=2, pca_center=origin))):


        tic = time.time()
        concat_matrix_diff = get_allinone_concat_matrix_diff(dir_name=traj_params_dir_name,
                                                             final_concat_params=final_concat_params)
        toc = time.time()
        print('\nElapsed time getting the full concat diff took {:.2f} s\n'
              .format(toc - tic))



        final_pca = PCA(n_components=n_components) # for sparse PCA to speed up

        tic = time.time()
        final_pca.fit(concat_matrix_diff)
        toc = time.time()
        logger.log('\nElapsed time computing the full PCA {:.2f} s\n'
              .format(toc - tic))

        logger.log(final_pca.explained_variance_ratio_)

        pcs_components = final_pca.components_

        first_n_pcs = pcs_components[:n_comp_to_use]
        mean_param = final_pca.mean_
        explained_variance_ratio = final_pca.explained_variance_ratio_

        if proj:
            proj_coords = do_proj(concat_matrix_diff, first_n_pcs, intermediate_data_dir, mean_param, origin)
            np.savetxt(get_projected_full_path_filename(intermediate_dir=intermediate_data_dir, n_comp=2, pca_center=origin),
                        proj_coords, delimiter=',')
        np.savetxt(get_pcs_filename(intermediate_dir=intermediate_data_dir, n_comp=n_components), pcs_components, delimiter=',')
        np.savetxt(get_mean_param_filename(intermediate_dir=intermediate_data_dir), mean_param, delimiter=',')
        np.savetxt(get_explain_ratios_filename(intermediate_dir=intermediate_data_dir, n_comp=2),
                   explained_variance_ratio, delimiter=',')

        print("gc the big thing")
        del concat_matrix_diff
        import gc
        gc.collect()
    else:
        pcs_components = np.loadtxt(
            get_pcs_filename(intermediate_dir=intermediate_data_dir, n_comp=n_components), delimiter=',')
        first_n_pcs = pcs_components[:n_comp_to_use]
        mean_param = np.loadtxt(get_mean_param_filename(intermediate_dir=intermediate_data_dir), delimiter=',')
        explained_variance_ratio = np.loadtxt(get_explain_ratios_filename(intermediate_dir=intermediate_data_dir, n_comp=2),
                   delimiter=',')
        if proj:
            proj_coords = np.loadtxt(get_projected_full_path_filename(intermediate_dir=intermediate_data_dir, n_comp=2, pca_center=origin), delimiter=',')


    result = {
        "pcs_components": pcs_components,
        "first_n_pcs": first_n_pcs,
        "mean_param":mean_param,
        "final_concat_params":final_concat_params,
        "explained_variance_ratio":explained_variance_ratio,
        "proj_coords":proj_coords
    }
    return result



def get_projected_data_in_old_basis(pca, data, num_axis_to_use):
    components = pca.components_[:num_axis_to_use]

    projected_data = (data - pca.mean_).dot(components.T)

    # goes back to original data space with mean restored.
    projected_data_in_old_basis = projected_data.dot(components) + pca.mean_

    return projected_data_in_old_basis

def calculate_projection_error(pca, data, num_axis_to_use):
    projected_data_in_old_basis = get_projected_data_in_old_basis(pca, data, num_axis_to_use)
    # assert_array_almost_equal(X_projected, X_projected2)
    # losses = LA.norm((data - projected_data_in_old_basis), ord=2, axis=1) #TODO: check this
    losses = []
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
                            num_levels=40, show=False, cma_path=None, cma_path_mean=None):
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

    if cma_path is not None:
        plt.plot(cma_path[0], cma_path[1], 'bo')
    if cma_path_mean is not None:
        plt.plot(cma_path_mean[0], cma_path_mean[1], marker='8')


    plt.annotate('end', xy=(proj_xcoord[-1], proj_ycoord[-1]), xytext=(proj_xcoord[-1] + 0.2, proj_ycoord[-1] +0.2),
                arrowprops=dict(facecolor='black', shrink=0.05),
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

    fig.savefig(f"{plot_dir_alg}/{name}.pdf", dpi=300,
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

    fig.savefig(f"{plot_dir_alg}/{name}.pdf", dpi=300,
                bbox_inches='tight', format='pdf')
    if show: plt.show()


def get_allinone_concat_matrix_diff(dir_name, final_concat_params):
    index = 0
    theta_file = get_full_param_traj_file_path(dir_name, index)

    concat_df = pd.read_csv(theta_file, header=None)

    result_matrix_diff = concat_df.sub(final_concat_params, axis='columns')

    index += 1

    while os.path.exists(get_full_param_traj_file_path(dir_name, index)):
        theta_file = get_full_param_traj_file_path(dir_name, index)

        part_concat_df = pd.read_csv(theta_file, header=None)

        part_concat_df = part_concat_df.sub(final_concat_params, axis='columns')

        result_matrix_diff = result_matrix_diff.append(part_concat_df, ignore_index=True)
        index += 1

    return result_matrix_diff.values



if __name__ == "__main__":
    from numpy.testing import assert_array_almost_equal

    X_train = np.random.randn(100, 50)

    pca = PCA(n_components=2)
    test_pca = PCA(n_components=5)
    pca.fit(X_train)
    test_pca.fit(X_train)


    X_train_pca = pca.transform(X_train)
    X_train_pca2 = (X_train - test_pca.mean_).dot(test_pca.components_[:2].T)

    assert_array_almost_equal(X_train_pca, X_train_pca2)

    X_projected = pca.inverse_transform(X_train_pca)
    X_projected2 = X_train_pca.dot(test_pca.components_[:2]) + test_pca.mean_
    k = get_projected_data_in_old_basis(test_pca, X_train, num_axis_to_use=2)
    assert_array_almost_equal(X_projected, k)

    # assert_array_almost_equal(X_projected, X_projected2)
    a = calculate_projection_error(test_pca, X_train, num_axis_to_use=2)
    loss = LA.norm((X_train - X_projected2), ord=2, axis=1)
    assert_array_almost_equal(a, loss)

