from numpy import linalg as LA
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA, PCA
import csv
from collections.abc import Iterable

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

def project_2D(d, dx, dy, proj_method='cos'):
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
                            num_levels=40, show=False):
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
