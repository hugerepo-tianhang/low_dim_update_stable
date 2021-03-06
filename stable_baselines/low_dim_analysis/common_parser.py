import argparse

def str2bool(v):
  #susendberg's function
  return v.lower() in ("yes", "True", "true", "t", "1")



def get_common_parser():
    parser = argparse.ArgumentParser(description='load and pca')
    parser.register('type', 'bool', str2bool)
    parser.add_argument("--normalize", type='bool', default=True)
    parser.add_argument('--use_IPCA', type='bool', default=True)
    parser.add_argument('--chunk_size', default=10000, type=int, help='total timesteps agent runs')
    parser.add_argument('--optimizer', help='environment ID', type=str, default='adam')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--additional_notes', default="", type=str, help='which run number')

    # PCA parameters
    parser.add_argument('--alg', default='ppo2', help='algorithm to train on')
    parser.add_argument('--env', default='DartWalker2d-v1', help='algorithm to train on')
    parser.add_argument('--num-timesteps', default=3000000, type=int, help='total timesteps agent runs')
    parser.add_argument('--run_num', default=0, type=int, help='which run number')

    parser.add_argument('--nminibatches', default=64, type=int, help='which run number')
    parser.add_argument('--n_steps', default=4096, type=int, help='which run number')

    parser.add_argument('--cores_to_use', default=-1, type=int, help='cores to use to parallel')
    parser.add_argument('--eval_num_timesteps', default=3000, type=int, help='total timesteps agent runs')
    parser.add_argument('--padding_fraction', default=0.4, type=float)
    parser.add_argument('--xnum', default=3, type=int)
    parser.add_argument('--ynum', default=3, type=int)
    parser.add_argument('--n_comp_to_use', default=1, type=int, help='n_components of PCA')
    parser.add_argument('--n_components', default=10, type=int, help='n_components of PCA')

    parser.add_argument('--even_check_point_num', default=5, type=int, help='even_check_point_num')
    parser.add_argument('--explain_ratio_threshold', default=0.99, type=float)
    parser.add_argument('--use_threads', action='store_true', default=False)

    parser.add_argument('--other_pca_index', default="3:4", help='cores to use to parallel')

    #cma params
    parser.add_argument('--cma_num_timesteps', default=10000, type=int, help='total timesteps agent runs')
    parser.add_argument('--cma_var', default=1, type=float, help='total timesteps agent runs')
    parser.add_argument("--origin", type=str,  default="mean_param")

    #PPOs
    parser.add_argument('--ppos_num_timesteps', default=10000, type=int, help='total timesteps agent runs')

    #pc1 chunk size
    parser.add_argument('--pc1_chunk_size', default=100, type=int, help='total timesteps agent runs')
    parser.add_argument('--deque_len', default=200, type=int, help='total timesteps agent runs')
    parser.add_argument('--skipped_chunks', default=1, type=int, help='total timesteps agent runs')

    parser.add_argument('--num_comp_to_load', default=150, type=int, help='total timesteps agent runs')

    #for cma adn the ppo
    parser.add_argument('--ppo_num_timesteps', default=5000, type=int, help='total timesteps agent runs')


    #WPCA VS last plane
    parser.add_argument('--func_index_to_use', default=0, type=int, help='total timesteps agent runs')

    #are final parameters
    parser.add_argument('--run_nums_to_check', default="0:1:2", help='total timesteps agent runs')




    return parser