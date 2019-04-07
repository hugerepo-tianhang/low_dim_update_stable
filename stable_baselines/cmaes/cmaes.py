from stable_baselines.ppo2.run_mujoco import eval_return
import cma

import numpy as np
from stable_baselines.low_dim_analysis.eval_util import *

from stable_baselines import logger

import pandas as pd
from sklearn.decomposition import PCA

from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import time
import os
from stable_baselines.common.cmd_util import mujoco_arg_parser
from stable_baselines.low_dim_analysis.common_parser import get_common_parser




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


def do_cma(cma_args, first_n_pcs, orgin_param, save_dir, starting_coord):

    tic = time.time()

    #TODO better starting locations, record how many samples,

    logger.log(f"CMAES STARTING :{starting_coord}")
    es = cma.CMAEvolutionStrategy(starting_coord, 2)
    total_num_of_evals = 0
    total_num_timesteps = 0


    mean_rets = []
    min_rets = []
    max_rets = []
    eval_returns = None

    optimization_path = []
    while total_num_timesteps < cma_args.cma_num_timesteps and not es.stop():
        solutions = es.ask()
        optimization_path.extend(solutions)
        thetas = [np.matmul(coord, first_n_pcs) + orgin_param for coord in solutions]
        logger.log(f"eval num: {cma_args.eval_num_timesteps}")
        eval_returns = Parallel(n_jobs=cma_args.cores_to_use) \
            (delayed(eval_return)(cma_args, save_dir, theta, cma_args.eval_num_timesteps, i) for
             (i, theta) in enumerate(thetas))


        mean_rets.append(np.mean(eval_returns))
        min_rets.append(np.min(eval_returns))
        max_rets.append(np.max(eval_returns))


        total_num_of_evals += len(eval_returns)
        total_num_timesteps += cma_args.eval_num_timesteps * len(eval_returns)

        logger.log(f"current eval returns: {str(eval_returns)}")
        logger.log(f"total timesteps so far: {total_num_timesteps}")
        negative_eval_returns = [-r for r in eval_returns]

        es.tell(solutions, negative_eval_returns)
        es.logger.add()  # write data to disc to be plotted
        es.disp()

    toc = time.time()
    logger.log(f"####################################CMA took {toc-tic} seconds")

    es_logger = es.logger

    if not hasattr(es_logger, 'xmean'):
        es_logger.load()

    optimization_path_mean = [starting_coord]
    optimization_path_mean.extend(es_logger.xmean[:,5:5+2])

    return mean_rets, min_rets, max_rets, np.array(optimization_path).T, np.array(optimization_path_mean).T


def main(origin="final_param"):
    import sys
    logger.log(sys.argv)
    common_arg_parser = get_common_parser()
    cma_args, cma_unknown_args = common_arg_parser.parse_known_args()


    this_run_dir = get_dir_path_for_this_run(cma_args)

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
    from stable_baselines.low_dim_analysis.common import do_pca
    result = do_pca(cma_args.n_components, cma_args.n_comp_to_use, traj_params_dir_name, intermediate_data_dir, proj=True,
                    origin=origin, use_IPCA=cma_args.use_IPCA, chunk_size=cma_args.chunk_size)
    '''
    ==========================================================================================
    eval all xy coords
    ==========================================================================================
    '''
    from stable_baselines.low_dim_analysis.common import plot_contour_trajectory, gen_subspace_coords,do_eval_returns
    proj_coord = result["proj_coords"]
    proj_xcoord = proj_coord[0]
    proj_ycoord = proj_coord[1]

    # starting_coord = np.zeros((1, cma_args.n_comp_to_use)) # use mean
    # starting_coord = (np.random.uniform(np.min(proj_xcoord), np.max(proj_xcoord)),
    #                 np.random.uniform(np.min(proj_ycoord), np.max(proj_ycoord)))
    starting_coord = (proj_xcoord[0],proj_ycoord[0])
    logger.log(f"CMA STASRTING CORRD: {starting_coord}")

    if origin=="final_param":
        origin_param = result["final_concat_params"]
    else:
        origin_param = result["mean_param"]



    cma_run_num = get_cma_run_num(intermediate_data_dir, n_comp=cma_args.n_comp_to_use)
    cma_intermediate_data_dir = get_cma_returns_dirname(intermediate_data_dir, cma_args.n_comp_to_use, cma_run_num)
    if not os.path.exists(cma_intermediate_data_dir):
        os.makedirs(cma_intermediate_data_dir)




    # starting_coord = (1/2*np.max(xcoordinates_to_eval), 1/2*np.max(ycoordinates_to_eval)) # use mean
    assert result["first_n_pcs"].shape[0] == 2
    mean_rets, min_rets, max_rets, opt_path, opt_path_mean = do_cma(cma_args, result["first_n_pcs"],
                                                                    origin_param, save_dir, starting_coord)

    np.savetxt(f"{cma_intermediate_data_dir}/mean_rets.txt",mean_rets, delimiter=',')
    np.savetxt(f"{cma_intermediate_data_dir}/min_rets.txt", min_rets, delimiter=',')
    np.savetxt(f"{cma_intermediate_data_dir}/max_rets.txt", max_rets, delimiter=',')


    plot_dir = get_plot_dir(cma_args)
    cma_plot_dir = get_cma_plot_dir(plot_dir, cma_args.n_comp_to_use, cma_run_num)
    if not os.path.exists(cma_plot_dir):
        os.makedirs(cma_plot_dir)

    ret_plot_name = f"cma return on {cma_args.n_comp_to_use} dim space of real pca plane"
    plot_cma_returns(cma_plot_dir, ret_plot_name, mean_rets, min_rets, max_rets, show=False)




    xcoordinates_to_eval, ycoordinates_to_eval = gen_subspace_coords(cma_args, np.hstack((proj_coord, opt_path_mean)))

    eval_returns = do_eval_returns(cma_args, intermediate_data_dir, result["first_n_pcs"], origin_param,
                    xcoordinates_to_eval, ycoordinates_to_eval, save_dir, pca_center=origin)

    plot_contour_trajectory(cma_plot_dir, "end_point_origin_eval_return_contour_plot", xcoordinates_to_eval,
                            ycoordinates_to_eval, eval_returns, proj_xcoord, proj_ycoord,
                            result["explained_variance_ratio"][:2],
                            num_levels=25, show=False, sub_alg_path=opt_path_mean)
    # plot_3d_trajectory(cma_plot_dir, "end_point_origin_eval_return_3d_plot", xcoordinates_to_eval, ycoordinates_to_eval,
    #                         eval_returns, proj_xcoord, proj_ycoord,
    #                         result["explained_variance_ratio"][:2],
    #                         num_levels=15, show=False)



if __name__ == '__main__':

    main(origin="mean_param")

#TODO Give filenames more info to identify which hyperparameter is the data for

