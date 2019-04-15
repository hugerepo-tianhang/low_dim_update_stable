from stable_baselines.ppo2.run_mujoco import eval_return
import cma

import numpy as np
from stable_baselines.low_dim_analysis.eval_util import *
from stable_baselines.low_dim_analysis.common import *

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
import numpy as np
import gym

from stable_baselines.common.cmd_util import mujoco_arg_parser
from stable_baselines import bench, logger
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import csv
import os
from stable_baselines.low_dim_analysis.eval_util import get_full_param_traj_file_path, get_full_params_dir, get_dir_path_for_this_run, get_log_dir, get_save_dir



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
    best_ever_theta = None
    best_ever_return = -float("inf")

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



        if np.max(eval_returns) > best_ever_return:
            logger.debug(f"current best return: {np.max(eval_returns)}, last best return {best_ever_return}")
            best_ever_theta = thetas[np.argmax(eval_returns)]

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

    return mean_rets, min_rets, max_rets, np.array(optimization_path), np.array(optimization_path_mean), best_ever_theta


def do_ppo(args, start_theta, parent_this_run_dir, full_space_save_dir):

    """
    Runs the test
    """

    logger.log(f"#######CMA and then PPO TRAIN: {args}")

    this_conti_ppo_run_dir = get_ppo_part(parent_this_run_dir)
    log_dir = get_log_dir(this_conti_ppo_run_dir)
    conti_ppo_save_dir = get_save_dir(this_conti_ppo_run_dir)
    logger.configure(log_dir)



    def make_env():
        env_out = gym.make(args.env)
        env_out.env.disableViewer = True
        env_out.env.visualize = False
        env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
        return env_out

    env = DummyVecEnv([make_env])
    if args.normalize:
        env = VecNormalize(env)


    set_global_seeds(args.seed)
    policy = MlpPolicy

    # extra run info I added for my purposes


    full_param_traj_dir_path = get_full_params_dir(this_conti_ppo_run_dir)

    if os.path.exists(full_param_traj_dir_path):
        import shutil
        shutil.rmtree(full_param_traj_dir_path)
    os.makedirs(full_param_traj_dir_path)

    if os.path.exists(conti_ppo_save_dir):
        import shutil
        shutil.rmtree(conti_ppo_save_dir)
    os.makedirs(conti_ppo_save_dir)

    run_info = {"run_num": args.run_num,
                "env_id": args.env,
                "full_param_traj_dir_path": full_param_traj_dir_path}

    # model = PPO2(policy=policy, env=env, n_steps=args.n_steps, nminibatches=args.nminibatches, lam=0.95, gamma=0.99,
    #              noptepochs=10,
    #              ent_coef=0.0, learning_rate=3e-4, cliprange=0.2, optimizer=args.optimizer)

    model = PPO2.load(f"{full_space_save_dir}/ppo2")
    model.set_env(env)
    model.tell_run_info(run_info)

    if args.normalize:
        env.load_running_average(full_space_save_dir)

    model.set_from_flat(start_theta)

    episode_returns = model.learn(total_timesteps=args.ppo_num_timesteps)

    model.save(f"{conti_ppo_save_dir}/ppo2")

    env.save_running_average(conti_ppo_save_dir)
    return episode_returns, full_param_traj_dir_path

def main():


    import sys
    logger.log(sys.argv)
    common_arg_parser = get_common_parser()
    cma_args, cma_unknown_args = common_arg_parser.parse_known_args()

    # origin = "final_param"
    origin = cma_args.origin


    this_run_dir = get_dir_path_for_this_run(cma_args)

    traj_params_dir_name = get_full_params_dir(this_run_dir)
    intermediate_data_dir = get_intermediate_data_dir(this_run_dir)
    save_dir = get_save_dir( this_run_dir)


    if not os.path.exists(intermediate_data_dir):
        os.makedirs(intermediate_data_dir)

    cma_run_num, cma_intermediate_data_dir = generate_run_dir(get_cma_and_then_ppo_run_dir,
                                                              intermediate_dir=intermediate_data_dir,
                                                              n_comp=cma_args.n_comp_to_use,
                                                              cma_steps=cma_args.cma_num_timesteps,
                                                              ppo_steps=cma_args.ppo_num_timesteps)

    '''
    ==========================================================================================
    get the pc vectors
    ==========================================================================================
    '''
    proj_or_not = (cma_args.n_comp_to_use == 2)
    result = do_pca(cma_args.n_components, cma_args.n_comp_to_use, traj_params_dir_name, intermediate_data_dir,
                    proj=proj_or_not,
                    origin=origin, use_IPCA=cma_args.use_IPCA, chunk_size=cma_args.chunk_size, reuse=True)
    logger.debug("after pca")
    '''
    ==========================================================================================
    eval all xy coords
    ==========================================================================================
    '''


    from stable_baselines.low_dim_analysis.common import plot_contour_trajectory, gen_subspace_coords,do_eval_returns, \
        do_proj_on_first_n

    if origin=="final_param":
        origin_param = result["final_concat_params"]
    else:
        origin_param = result["mean_param"]

    logger.log("grab start params")
    start_file = get_full_param_traj_file_path(traj_params_dir_name, "start")
    start_params = pd.read_csv(start_file, header=None).values[0]
    starting_coord = do_proj_on_first_n(start_params, result["first_n_pcs"], origin_param)
    # starting_coord = np.random.rand(1, cma_args.n_comp_to_use)





    # starting_coord = (1/2*np.max(xcoordinates_to_eval), 1/2*np.max(ycoordinates_to_eval)) # use mean
    assert result["first_n_pcs"].shape[0] == cma_args.n_comp_to_use
    mean_rets, min_rets, max_rets, opt_path, opt_path_mean, best_theta = do_cma(cma_args, result["first_n_pcs"],
                                                                    origin_param, save_dir, starting_coord, cma_args.cma_var)
    dump_rows_write_csv(cma_intermediate_data_dir, opt_path_mean, "opt_mean_path")
    best_theta_file_name = "best theta from cma"

    dump_row_write_csv(cma_intermediate_data_dir, best_theta, best_theta_file_name)



    episode_returns, conti_ppo_full_param_traj_dir_path = do_ppo(args=cma_args, start_theta=best_theta, parent_this_run_dir=cma_intermediate_data_dir, full_space_save_dir=save_dir)
    dump_row_write_csv(cma_intermediate_data_dir, episode_returns, "ppo part returns")



    plot_dir = get_plot_dir(cma_args)
    cma_and_then_ppo_plot_dir = get_cma_and_then_ppo_plot_dir(plot_dir, cma_args.n_comp_to_use,
                                                 cma_run_num, cma_num_steps=cma_args.cma_num_timesteps,
                                                              ppo_num_steps=cma_args.ppo_num_timesteps,
                                                              origin=origin)
    if not os.path.exists(cma_and_then_ppo_plot_dir):
        os.makedirs(cma_and_then_ppo_plot_dir)

    ret_plot_name = f"cma return on {cma_args.n_comp_to_use} dim space of real pca plane, " \
                    f"explained {np.sum(result['explained_variance_ratio'][:cma_args.n_comp_to_use])}"
    plot_cma_returns(cma_and_then_ppo_plot_dir, ret_plot_name, mean_rets, min_rets, max_rets, show=False)


    final_ppo_ep_name = f"final episodes returns after CMA"
    plot_2d(cma_and_then_ppo_plot_dir, final_ppo_ep_name, np.arange(len(episode_returns)),
            episode_returns, "num episode", "episode returns", False)

    #
    # if cma_args.n_comp_to_use == 2:
    #     proj_coords = result["proj_coords"]
    #     assert proj_coords.shape[1] == 2
    #
    #     xcoordinates_to_eval, ycoordinates_to_eval = gen_subspace_coords(cma_args, np.vstack((proj_coords, opt_path_mean)).T)
    #
    #     eval_returns = do_eval_returns(cma_args, intermediate_data_dir, result["first_n_pcs"], origin_param,
    #                     xcoordinates_to_eval, ycoordinates_to_eval, save_dir, pca_center=origin, reuse=False)
    #
    #     plot_contour_trajectory(cma_and_then_ppo_plot_dir, f"{origin}_origin_eval_return_contour_plot", xcoordinates_to_eval,
    #                             ycoordinates_to_eval, eval_returns, proj_coords[:, 0], proj_coords[:, 1],
    #                             result["explained_variance_ratio"][:2],
    #                             num_levels=25, show=False, sub_alg_path=opt_path_mean)


    skip_rows = lambda x: x%2 == 0
    conti_ppo_params = get_allinone_concat_df(conti_ppo_full_param_traj_dir_path, index=0, skip_rows=skip_rows).values
    opt_mean_path_in_old_basis = [mean_projected_param.dot(result["first_n_pcs"]) + result["mean_param"] for mean_projected_param in opt_path_mean]
    distance_to_final = [LA.norm(opt_mean - result["final_concat_params"], ord=2) for opt_mean in np.vstack((opt_mean_path_in_old_basis, conti_ppo_params))]
    distance_to_final_plot_name = f"distance_to_final over generations "
    plot_2d(cma_and_then_ppo_plot_dir, distance_to_final_plot_name, np.arange(len(distance_to_final)), distance_to_final, "num generation", "distance_to_final", False)



    # plot_3d_trajectory(cma_plot_dir, "end_point_origin_eval_return_3d_plot", xcoordinates_to_eval, ycoordinates_to_eval,
    #                         eval_returns, proj_xcoord, proj_ycoord,
    #                         result["explained_variance_ratio"][:2],
    #                         num_levels=15, show=False)



if __name__ == '__main__':

    main()

#TODO Give filenames more info to identify which hyperparameter is the data for

