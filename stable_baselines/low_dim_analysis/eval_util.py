# def get_param_traj_file_path(dir_name, net_name, index):
#     return f'{dir_name}/{net_name}_{index}.txt'
import os

def get_project_dir():
    project_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
    return project_dir


def get_dir_path_for_this_run(alg, total_timesteps, env_id, run_num):
    return f'{get_project_dir()}/stable_baselines/{alg}/env_{env_id}_time_step_{total_timesteps}_run_{run_num}'

def get_log_dir(alg, total_timesteps, env_id, run_num):
    return f"{get_dir_path_for_this_run(alg, total_timesteps, env_id, run_num)}/the_log_dir"

def get_save_dir(alg, total_timesteps, env_id, run_num):
    return f"{get_dir_path_for_this_run(alg, total_timesteps, env_id, run_num)}/the_save_dir"

def get_eval_losses_file_path(dir_name, total_timesteps):
    return f'{dir_name}/eval_loss_{total_timesteps}.hdf5'

def get_full_param_traj_file_path(dir_name, index):
    return f'{dir_name}/all_params_{index}.txt'






def get_plot_dir(alg, total_timesteps, env_id, run_num):
    return f'{get_project_dir()}/plots/{alg}/env_{env_id}_time_step_{total_timesteps}_run_{run_num}'

def get_full_params_dir(alg, total_timesteps, env_id, run_num):
    this_run_dir = get_dir_path_for_this_run(alg, total_timesteps, env_id, run_num)

    return f"{this_run_dir}/full_params"

def get_intermediate_data_dir(alg, total_timesteps, env_id, run_num):
    this_run_dir = get_dir_path_for_this_run(alg, total_timesteps, env_id, run_num)

    return f"{this_run_dir}/intermediate_data"

def get_pcs_filename(intermediate_dir, n_comp):

    return f"{intermediate_dir}/n_comp_{n_comp}_pcs"
def get_explain_ratios_filename(intermediate_dir, n_comp):

    return f"{intermediate_dir}/n_comp_{n_comp}_explain_ratios"
def get_projected_full_path_filename(intermediate_dir, n_comp):

    return f"{intermediate_dir}/n_comp_{n_comp}_projected_full_path"
def get_eval_returns_filename(intermediate_dir, xnum, ynum, n_comp):

    return f"{intermediate_dir}/xnum_{xnum}_ynum_{ynum}_n_comp_{n_comp}_eval_returns"

if __name__ == '__main__':
    print(get_project_dir())