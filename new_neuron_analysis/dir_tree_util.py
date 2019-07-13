import shutil
import os
import datetime
def get_time_stamp(style='%Y-%m-%d %H:%M:%S'):
    return datetime.datetime.now().strftime(style)


def create_dir_remove(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)


def create_dir_if_not(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_project_dir():
    project_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..'))

    return f"{project_dir}/new_neuron_analysis"


def get_run_name_dir_tree(env, policy_num_timesteps, policy_run_num, policy_seed, run_seed, run_run_num):
    run_name = f"{env}_policy_num_timesteps_{policy_num_timesteps}" \
               f"_policy_run_num_{policy_run_num}_policy_seed_{policy_seed}_run_seed_{run_seed}_run_run_num_{run_run_num}"
    return run_name

def get_data_dir(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num, additional_note):
    data_dir = f"{get_project_dir()}/data/" \
               f"{get_run_name_dir_tree(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num)}_additional_note_{additional_note}"
    return data_dir

def get_plot_dir(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num, additional_note):
    run_name = get_run_name_dir_tree(policy_env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num)
    data_dir = f"{get_project_dir()}/plot/" \
               f"{run_name}_additional_note_{additional_note}"
    return data_dir

def get_result_dir(env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num, additional_note):
    data_dir = f"{get_project_dir()}/result/" \
               f"{get_run_name_dir_tree(env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num)}_additional_note_{additional_note}"
    return data_dir

def get_test_dir(env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num, augment_seed, additional_note):
    data_dir = f"{get_project_dir()}/test/" \
               f"{get_run_name_dir_tree(env, policy_num_timesteps, policy_run_num, policy_seed, eval_seed, eval_run_num)}_additional_note_{additional_note}"
    return data_dir

def get_original_env_test_dir(env,augment_seed):
    data_dir = f"{get_project_dir()}/test/" \
               f"original: {env} augment_seed{augment_seed}"
    return data_dir