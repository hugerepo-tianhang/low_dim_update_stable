from stable_baselines.ppo2.run_mujoco import eval_return
import cma

import numpy as np
from stable_baselines.low_dim_analysis.eval_util import *
from stable_baselines.low_dim_analysis.common import do_pca, plot_2d, \
    dump_rows_write_csv, generate_run_dir, do_proj_on_first_n_IPCA, get_allinone_concat_df
from sklearn.decomposition import IncrementalPCA


from stable_baselines import logger

import pandas as pd
from sklearn.decomposition import PCA

from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import time
import os
from stable_baselines.common.cmd_util import mujoco_arg_parser
from stable_baselines.low_dim_analysis.common_parser import get_common_parser
from numpy import linalg as LA



def plot_cma_returns(plot_dir_alg, name, mean_rets, min_rets, max_rets, show):

    X = np.arange(len(mean_rets))
    fig, ax = plt.subplots()
    plt.xlabel('num of eval')
    plt.ylabel('mean returns with min and max filled')

    ax.plot(X, mean_rets)
    ax.fill_between(X, min_rets, max_rets, alpha=0.5)
    file_path = f"{plot_dir_alg}/{name}.pdf"
    if os.path.isfile(file_path):
        os.remove(file_path)

    logger.log(f"saving cma plot to {file_path}")
    fig.savefig(file_path, dpi=300,
                bbox_inches='tight', format='pdf')
    if show: plt.show()




def do_cma(cma_args, first_n_pcs, orgin_param, save_dir, starting_coord, var):

    tic = time.time()

    #TODO better starting locations, record how many samples,

    logger.log(f"CMAES STARTING :{starting_coord}")
    es = cma.CMAEvolutionStrategy(starting_coord, var)
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
        logger.log(f"current time steps num: {total_num_timesteps} total time steps: {cma_args.cma_num_timesteps}")
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


    n_comp_used = first_n_pcs.shape[0]
    optimization_path_mean = np.vstack((starting_coord, es_logger.xmean[:,5:5+n_comp_used]))

    return mean_rets, min_rets, max_rets, np.array(optimization_path), np.array(optimization_path_mean)


def main():


    import sys
    logger.log(sys.argv)
    common_arg_parser = get_common_parser()
    cma_args, cma_unknown_args = common_arg_parser.parse_known_args()

    origin = "mean_param"


    this_run_dir = get_dir_path_for_this_run(cma_args)

    traj_params_dir_name = get_full_params_dir(this_run_dir)
    intermediate_data_dir = get_intermediate_data_dir(this_run_dir)
    save_dir = get_save_dir( this_run_dir)


    if not os.path.exists(intermediate_data_dir):
        os.makedirs(intermediate_data_dir)

    cma_run_num, cma_intermediate_data_dir = generate_run_dir(get_cma_returns_dirname, intermediate_dir=intermediate_data_dir, n_comp=cma_args.n_comp_to_use)
    '''
    ==========================================================================================
    get the pc vectors
    ==========================================================================================
    '''

    logger.log("grab final params")
    final_file = get_full_param_traj_file_path(traj_params_dir_name, "final")
    final_param = pd.read_csv(final_file, header=None).values[0]


    final_pca = IncrementalPCA(n_components=2)  # for sparse PCA to speed up

    theta_file = get_full_param_traj_file_path(traj_params_dir_name, 0)
    concat_df = pd.read_csv(theta_file, header=None, chunksize=10000)


    tic = time.time()
    for chunk in concat_df:
        logger.log(f"currnet at : {concat_df._currow}")

        if chunk.shape[0] < 2:
            logger.log(f"last column too few: {chunk.shape[0]}")
            continue
        final_pca.partial_fit(chunk.values)

    toc = time.time()
    logger.log('\nElapsed time computing the chunked PCA {:.2f} s\n'
               .format(toc - tic))

    logger.log(final_pca.explained_variance_ratio_)

    pcs_components = final_pca.components_

    first_2_pcs = pcs_components[:2]
    mean_param = final_pca.mean_

    origin_param = mean_param


    theta_file = get_full_param_traj_file_path(traj_params_dir_name, 0)
    concat_df = pd.read_csv(theta_file, header=None, chunksize=10000)

    proj_coords = do_proj_on_first_n_IPCA(concat_df, first_2_pcs, origin_param)


    '''
    ==========================================================================================
    eval all xy coords
    ==========================================================================================
    '''


    from stable_baselines.low_dim_analysis.common import plot_contour_trajectory, gen_subspace_coords,do_eval_returns, \
        get_allinone_concat_df, do_proj_on_first_n

    from stable_baselines.ppo2.run_mujoco import eval_return

    last_proj_coord = do_proj_on_first_n(final_param, first_2_pcs, origin_param)
    starting_coord = last_proj_coord

    tic = time.time()

    #TODO better starting locations, record how many samples,

    logger.log(f"CMAES STARTING :{starting_coord}")
    es = cma.CMAEvolutionStrategy(starting_coord, 5)
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
        thetas = [np.matmul(coord, first_2_pcs) + origin_param for coord in solutions]
        logger.log(f"current time steps num: {total_num_timesteps} total time steps: {cma_args.cma_num_timesteps}")
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


    n_comp_used = first_2_pcs.shape[0]
    optimization_path_mean = np.vstack((starting_coord, es_logger.xmean[:,5:5+n_comp_used]))


    dump_rows_write_csv(cma_intermediate_data_dir, optimization_path_mean, "opt_mean_path")



    plot_dir = get_plot_dir(cma_args)
    cma_plot_dir = get_cma_plot_dir(plot_dir, cma_args.n_comp_to_use, cma_run_num, origin=origin)
    if not os.path.exists(cma_plot_dir):
        os.makedirs(cma_plot_dir)

    ret_plot_name = f"cma return on {cma_args.n_comp_to_use} dim space of real pca plane, " \
                    f"explained {np.sum(final_pca.explained_variance_ratio_[:2])}"
    plot_cma_returns(cma_plot_dir, ret_plot_name, mean_rets, min_rets, max_rets, show=False)




    assert proj_coords.shape[1] == 2

    xcoordinates_to_eval, ycoordinates_to_eval = gen_subspace_coords(cma_args, np.vstack((proj_coords, optimization_path_mean)).T)

    from stable_baselines.ppo2.run_mujoco import eval_return
    thetas_to_eval = [origin_param + x * first_2_pcs[0] + y * first_2_pcs[1] for y in ycoordinates_to_eval for x in
                      xcoordinates_to_eval]

    tic = time.time()

    eval_returns = Parallel(n_jobs=-1, max_nbytes='100M') \
        (delayed(eval_return)(cma_args, save_dir, theta, cma_args.eval_num_timesteps, i) for (i, theta) in
         enumerate(thetas_to_eval))
    toc = time.time()
    logger.log(f"####################################1st version took {toc-tic} seconds")

    plot_contour_trajectory(cma_plot_dir, f"cma redo___{origin}_origin_eval_return_contour_plot", xcoordinates_to_eval,
                            ycoordinates_to_eval, eval_returns, proj_coords[:, 0], proj_coords[:, 1],
                            final_pca.explained_variance_ratio_,
                            num_levels=25, show=False, sub_alg_path=optimization_path_mean.T)



    opt_mean_path_in_old_basis = [mean_projected_param.dot(first_2_pcs) + mean_param for mean_projected_param in optimization_path_mean]
    distance_to_final = [LA.norm(opt_mean - final_param, ord=2) for opt_mean in opt_mean_path_in_old_basis]
    distance_to_final_plot_name = f"cma redo distance_to_final over generations "
    plot_2d(cma_plot_dir, distance_to_final_plot_name, np.arange(len(distance_to_final)), distance_to_final, "num generation", "distance_to_final", False)

    # plot_3d_trajectory(cma_plot_dir, "end_point_origin_eval_return_3d_plot", xcoordinates_to_eval, ycoordinates_to_eval,
    #                         eval_returns, proj_xcoord, proj_ycoord,
    #                         result["explained_variance_ratio"][:2],
    #                         num_levels=15, show=False)



if __name__ == '__main__':

    main()

#TODO Give filenames more info to identify which hyperparameter is the data for

