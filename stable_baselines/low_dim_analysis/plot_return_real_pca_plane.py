import numpy as np
from stable_baselines.low_dim_analysis.eval_util import *

from stable_baselines import logger

import pandas as pd
from sklearn.decomposition import PCA

from stable_baselines.low_dim_analysis.common import project_2D_pca_mean_origin, plot_contour_trajectory, plot_3d_trajectory
from joblib import Parallel, delayed



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


def parse_unknown_args(args):
    """
    Parse arguments not consumed by arg parser into a dicitonary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval
def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''

    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k, v in parse_unknown_args(args).items()}



if __name__ == '__main__':

    import time
    import os
    from stable_baselines.common.cmd_util import mujoco_arg_parser
    from stable_baselines.low_dim_analysis.common_parser import get_common_parser
    parser = get_common_parser()
    openai_arg_parser = mujoco_arg_parser()

    plot_args, plot_unknown_args = parser.parse_known_args()
    openai_args, openai_unknown_args = openai_arg_parser.parse_known_args()


    plot_unknown_args = parse_cmdline_kwargs(plot_unknown_args)
    openai_unknown_args = parse_cmdline_kwargs(openai_unknown_args)
    both_unknown_args = dict(plot_unknown_args.items() & openai_unknown_args.items())


    threads_or_None = 'threads' if plot_args.use_threads else None
    logger.log(f"THREADS OR NOT: {threads_or_None}")


    plot_dir_alg = get_plot_dir(plot_args.alg, plot_args.num_timesteps, plot_args.env, plot_args.normalize, plot_args.run_num)
    this_run_dir = get_dir_path_for_this_run(plot_args.alg, plot_args.num_timesteps,
                                             plot_args.env, plot_args.normalize, plot_args.run_num)

    traj_params_dir_name = get_full_params_dir(this_run_dir)
    intermediate_data_dir = get_intermediate_data_dir(this_run_dir)
    save_dir = get_save_dir( this_run_dir)

    if not os.path.exists(plot_dir_alg):
        os.makedirs(plot_dir_alg)
    if not os.path.exists(intermediate_data_dir):
        os.makedirs(intermediate_data_dir)


    logger.log("grab final params")
    final_file = get_full_param_traj_file_path(traj_params_dir_name, "final")
    final_concat_params = pd.read_csv(final_file, header=None).values[0]


    '''
    ==========================================================================================
    get the pc vectors
    ==========================================================================================
    '''

    if not os.path.exists(get_pcs_filename(intermediate_dir=intermediate_data_dir, n_comp=2))\
        or not os.path.exists(get_explain_ratios_filename(intermediate_dir=intermediate_data_dir, n_comp=2))\
        or not os.path.exists(get_projected_full_path_filename(intermediate_dir=intermediate_data_dir, n_comp=2))\
        or not os.path.exists(get_mean_param_filename(intermediate_dir=intermediate_data_dir)):

        tic = time.time()
        concat_matrix_diff = get_allinone_concat_matrix_diff(dir_name=traj_params_dir_name,
                                                             final_concat_params=final_concat_params)
        toc = time.time()
        print('\nElapsed time getting the full concat diff took {:.2f} s\n'
              .format(toc - tic))



        final_pca = PCA(n_components=plot_args.n_components) # for sparse PCA to speed up

        tic = time.time()
        final_pca.fit(concat_matrix_diff)
        toc = time.time()
        logger.log('\nElapsed time computing the full PCA {:.2f} s\n'
              .format(toc - tic))

        logger.log(final_pca.explained_variance_ratio_)


        pcs_components = final_pca.components_

        first_2_pcs = pcs_components[:2]
        explained_variance_ratio = final_pca.explained_variance_ratio_
        mean_param = final_pca.mean_

        np.savetxt(get_pcs_filename(intermediate_dir=intermediate_data_dir,
                                    n_comp=plot_args.n_components), pcs_components, delimiter=',')
        np.savetxt(get_mean_param_filename(intermediate_dir=intermediate_data_dir), mean_param, delimiter=',')
        np.savetxt(get_explain_ratios_filename(intermediate_dir=intermediate_data_dir, n_comp=2),
                   explained_variance_ratio, delimiter=',')



        '''
        ==========================================================================================
        project full path to first 2 pcs
        ==========================================================================================
        '''


        logger.log(f"project all params")
        proj_xcoord, proj_ycoord = [], []
        for param in concat_matrix_diff:

            x, y = project_2D_pca_mean_origin(d=param, dx=first_2_pcs[0], dy=first_2_pcs[1], pca_mean=mean_param)

            proj_xcoord.append(x)
            proj_ycoord.append(y)

        proj_coords = np.array([proj_xcoord, proj_ycoord])
        np.savetxt(get_projected_full_path_filename(intermediate_dir=intermediate_data_dir, n_comp=2), proj_coords, delimiter=',')

        print("gc the big thing")
        del concat_matrix_diff
        import gc
        gc.collect()

    else:
        pcs_components = np.loadtxt(
            get_pcs_filename(intermediate_dir=intermediate_data_dir, n_comp=plot_args.n_components), delimiter=',')
        first_2_pcs = pcs_components[:2]

        explained_variance_ratio = np.loadtxt(get_explain_ratios_filename(intermediate_dir=intermediate_data_dir, n_comp=2),
                   delimiter=',')
        proj_coords = np.loadtxt(get_projected_full_path_filename(intermediate_dir=intermediate_data_dir, n_comp=2), delimiter=',')
        mean_param = np.loadtxt(get_mean_param_filename(intermediate_dir=intermediate_data_dir), delimiter=',')


    '''
    ==========================================================================================
    generate the coords to eval
    ==========================================================================================
    '''
    proj_xcoord, proj_ycoord = proj_coords
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
    '''
    ==========================================================================================
    eval all xy coords
    ==========================================================================================
    '''
    if not os.path.exists(get_eval_returns_filename(intermediate_dir=intermediate_data_dir,
                                                    xnum=plot_args.xnum, ynum=plot_args.ynum, n_comp=2)):

        from stable_baselines.ppo2.run_mujoco import eval_return

        tic = time.time()
        thetas_to_eval = [mean_param + x * first_2_pcs[0] + y * first_2_pcs[1] for y in ycoordinates_to_eval for x in xcoordinates_to_eval]

        eval_returns = Parallel(n_jobs=plot_args.cores_to_use, max_nbytes='100M')\
            (delayed(eval_return)(openai_args, save_dir, theta, plot_args.eval_num_timesteps, i) for (i, theta) in enumerate(thetas_to_eval))
        toc = time.time()
        logger.log(f"####################################1st version took {toc-tic} seconds")


        np.savetxt(get_eval_returns_filename(intermediate_dir=intermediate_data_dir,
                                                    xnum=plot_args.xnum, ynum=plot_args.ynum, n_comp=2), eval_returns, delimiter=',')
    else:
        eval_returns = np.loadtxt(get_eval_returns_filename(intermediate_dir=intermediate_data_dir,
                                                    xnum=plot_args.xnum, ynum=plot_args.ynum, n_comp=2), delimiter=',')


    '''
    ==========================================================================================
    plot
    ==========================================================================================
    '''

    plot_contour_trajectory(plot_dir_alg, "eval_return_contour_plot", xcoordinates_to_eval, ycoordinates_to_eval, eval_returns, proj_xcoord, proj_ycoord,
                            explained_variance_ratio[:2],
                            num_levels=15, show=False)
    plot_3d_trajectory(plot_dir_alg, "eval_return_3d_plot", xcoordinates_to_eval, ycoordinates_to_eval,
                            eval_returns, proj_xcoord, proj_ycoord,
                            explained_variance_ratio[:2],
                            num_levels=15, show=False)
#TODO Give filenames more info to identify which hyperparameter is the data for



