import argparse
from stable_baselines.low_dim_analysis.common_parser import str2bool

def get_cma_parser():
    parser = argparse.ArgumentParser(description='load and pca')
    parser.register('type', 'bool', str2bool)
    parser.add_argument("--normalize", type='bool', default=False)
    parser.add_argument('--cma_num_timesteps', default=800000, type=int, help='total timesteps agent runs')
    # PCA parameters
    parser.add_argument('--alg', default='ppo2', help='algorithm to train on')
    parser.add_argument('--env', default='Hopper-v2', help='algorithm to train on')
    parser.add_argument('--num-timesteps', default=625000, type=int, help='total timesteps agent runs')
    parser.add_argument('--run_num', default=0, type=int, help='which run number')

    parser.add_argument('--cores_to_use', default=1, type=int, help='cores to use to parallel')
    parser.add_argument('--eval_num_timesteps', default=2048, type=int, help='total timesteps agent runs')

    parser.add_argument('--n_comp_to_use', default=15, type=int, help='n_components of PCA')
    parser.add_argument('--n_components', default=15, type=int, help='n_components of PCA')

    return parser