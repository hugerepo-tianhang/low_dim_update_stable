import argparse
def get_common_parser():
    parser = argparse.ArgumentParser(description='load and pca')

    # PCA parameters
    parser.add_argument('--cores_to_use', default=2, type=int, help='cores to use to parallel')
    parser.add_argument('--alg', default='ppo2', help='algorithm to train on')
    parser.add_argument('--env', default='Hopper-v2', help='algorithm to train on')
    parser.add_argument('--num_timesteps', default=50000, type=int, help='total timesteps agent runs')
    parser.add_argument('--eval_num_timesteps', default=5000, type=int, help='total timesteps agent runs')
    parser.add_argument('--run_num', default=0, type=int, help='which run number')
    parser.add_argument('--padding_fraction', default=0.4, type=float)
    parser.add_argument('--xnum', default=3, type=int)
    parser.add_argument('--ynum', default=3, type=int)
    parser.add_argument('--n_comp_to_use', default=10, type=int, help='n_components of PCA')


    parser.add_argument('--even_check_point_num', default=5, type=int, help='even_check_point_num')
    parser.add_argument('--n_components', default=10, type=int, help='n_components of PCA')
    parser.add_argument('--explain_ratio_threshold', default=0.9, type=float)
    parser.add_argument('--use_IPCA', action='store_true', default=False)
    parser.add_argument('--use_threads', action='store_true', default=False)


    return parser