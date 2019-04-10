import argparse

def str2bool(v):
  #susendberg's function
  return v.lower() in ("yes", "True", "true", "t", "1")



def get_common_parser():
    parser = argparse.ArgumentParser(description='load and pca')
    parser.register('type', 'bool', str2bool)
    parser.add_argument("--normalize", type='bool', default=True)
    parser.add_argument('--use_IPCA', type='bool', default=True)
    parser.add_argument('--chunk_size', default=100, type=int, help='total timesteps agent runs')

    # PCA parameters
    parser.add_argument('--alg', default='ppo2', help='algorithm to train on')
    parser.add_argument('--env', default='DartWalker2d-v1', help='algorithm to train on')
    parser.add_argument('--num-timesteps', default=5000, type=int, help='total timesteps agent runs')
    parser.add_argument('--run_num', default=0, type=int, help='which run number')

    parser.add_argument('--nminibatches', default=32, type=int, help='which run number')
    parser.add_argument('--n_steps', default=2048, type=int, help='which run number')

    parser.add_argument('--cores_to_use', default=-1, type=int, help='cores to use to parallel')
    parser.add_argument('--eval_num_timesteps', default=1024, type=int, help='total timesteps agent runs')
    parser.add_argument('--padding_fraction', default=0.4, type=float)
    parser.add_argument('--xnum', default=3, type=int)
    parser.add_argument('--ynum', default=3, type=int)
    parser.add_argument('--n_comp_to_use', default=10, type=int, help='n_components of PCA')
    parser.add_argument('--n_components', default=400, type=int, help='n_components of PCA')

    parser.add_argument('--even_check_point_num', default=5, type=int, help='even_check_point_num')
    parser.add_argument('--explain_ratio_threshold', default=0.99, type=float)
    parser.add_argument('--use_threads', action='store_true', default=False)

    parser.add_argument('--other_pca_index', default="8:9", help='cores to use to parallel')

    #cma params
    parser.add_argument('--cma_num_timesteps', default=50000, type=int, help='total timesteps agent runs')
    parser.add_argument("--origin", type=str,  default="mean_param")

    #PPOs
    parser.add_argument('--ppos_num_timesteps', default=10000, type=int, help='total timesteps agent runs')

    #pc1 chunk size
    parser.add_argument('--pc1_chunk_size', default=100, type=int, help='total timesteps agent runs')
    parser.add_argument('--deque_len', default=200, type=int, help='total timesteps agent runs')
    parser.add_argument('--skipped_chunks', default=10, type=int, help='total timesteps agent runs')


    return parser