

import numpy as np
from stable_baselines.low_dim_analysis.eval_util import *

from stable_baselines import logger

import pandas as pd
from sklearn.decomposition import PCA

from joblib import Parallel, delayed
from matplotlib import pyplot as plt


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

def plot_cma_returns(plot_dir_alg, name, mean_rets, min_rets, max_rets, show):

    X = np.arange(len(mean_rets))
    fig, ax = plt.subplots()
    plt.xlabel('num of eval')
    plt.ylabel('mean returns with min and max filled')

    ax.plot(X, mean_rets)
    ax.fill_between(X, min_rets, max_rets, alpha=0.5)
    fig.savefig(f"{plot_dir_alg}/{name}.pdf", dpi=300,
                bbox_inches='tight', format='pdf')
    if show: plt.show()


def get_cma_run_num(intermediate_data_dir, n_comp):
    run_num = 0
    while os.path.exists(get_cma_returns_dirname(intermediate_data_dir, n_comp=n_comp, run_num=run_num)):
        run_num += 1
    return run_num


if __name__ == '__main__':

    import time
    import os
    from stable_baselines.common.cmd_util import mujoco_arg_parser
    from stable_baselines.low_dim_analysis.common_parser import get_common_parser
    from stable_baselines.cmaes.cma_arg_parser import get_cma_parser
    parser = get_common_parser()
    openai_arg_parser = mujoco_arg_parser()
    cma_arg_parser = get_cma_parser()

    cma_args, cma_unknown_args = cma_arg_parser.parse_known_args()
    openai_args, openai_unknown_args = openai_arg_parser.parse_known_args()



    this_run_dir = get_dir_path_for_this_run(cma_args.alg, cma_args.num_timesteps,
                                             cma_args.env, cma_args.normalize, cma_args.run_num)

    traj_params_dir_name = get_full_params_dir(this_run_dir)
    intermediate_data_dir = get_intermediate_data_dir(this_run_dir)
    save_dir = get_save_dir( this_run_dir)


    if not os.path.exists(intermediate_data_dir):
        os.makedirs(intermediate_data_dir)


    '''
    ==========================================================================================
    get the pc vectors
    ==========================================================================================
    '''

    if not os.path.exists(get_pcs_filename(intermediate_dir=intermediate_data_dir, n_comp=cma_args.n_components))\
        or not os.path.exists(get_mean_param_filename(intermediate_dir=intermediate_data_dir)):

        logger.log("grab final params")
        final_file = get_full_param_traj_file_path(traj_params_dir_name, "final")
        final_concat_params = pd.read_csv(final_file, header=None).values[0]

        tic = time.time()
        concat_matrix_diff = get_allinone_concat_matrix_diff(dir_name=traj_params_dir_name,
                                                             final_concat_params=final_concat_params)
        toc = time.time()
        print('\nElapsed time getting the full concat diff took {:.2f} s\n'
              .format(toc - tic))



        final_pca = PCA(n_components=cma_args.n_components) # for sparse PCA to speed up

        tic = time.time()
        final_pca.fit(concat_matrix_diff)
        toc = time.time()
        logger.log('\nElapsed time computing the full PCA {:.2f} s\n'
              .format(toc - tic))

        logger.log(final_pca.explained_variance_ratio_)

        pcs_components = final_pca.components_

        first_n_pcs = pcs_components[:cma_args.n_comp_to_use]
        mean_param = final_pca.mean_

        np.savetxt(get_pcs_filename(intermediate_dir=intermediate_data_dir, n_comp=cma_args.n_components), pcs_components, delimiter=',')
        np.savetxt(get_mean_param_filename(intermediate_dir=intermediate_data_dir), mean_param, delimiter=',')



        print("gc the big thing")
        del concat_matrix_diff
        import gc
        gc.collect()

    else:
        pcs_components = np.loadtxt(
            get_pcs_filename(intermediate_dir=intermediate_data_dir, n_comp=cma_args.n_components), delimiter=',')
        first_n_pcs = pcs_components[:cma_args.n_comp_to_use]
        mean_param = np.loadtxt(get_mean_param_filename(intermediate_dir=intermediate_data_dir), delimiter=',')
