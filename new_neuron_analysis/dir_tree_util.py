import shutil
import os
def create_dir_remove(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)


def create_dir_if_not(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_proj_dir():
    project_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..'))

    return f"{project_dir}/new_neuron_analysis"


def get_run_name(env, policy_num_timesteps, policy_run_num, policy_seed, run_seed, run_run_num):
    run_name = f"{env}_policy_num_timesteps_{policy_num_timesteps}" \
               f"_policy_run_num_{policy_run_num}_policy_seed_{policy_seed}_run_seed_{run_seed}_run_run_num_{run_run_num}"
    return run_name
def get_data_dir(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num):
    data_dir = f"{get_proj_dir()}/data/" \
               f"{get_run_name(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num)}"
    return data_dir

def get_plot_dir(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num):
    data_dir = f"{get_proj_dir()}/plot/" \
               f"{get_run_name(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num)}"
    return data_dir

def get_result_dir(env, policy_num_timesteps, policy_run_num, policy_seed, run_seed, run_run_num):
    data_dir = f"{get_proj_dir()}/result/" \
               f"{get_run_name(env, policy_num_timesteps, policy_run_num, policy_seed, run_seed, run_run_num)}"
    return data_dir