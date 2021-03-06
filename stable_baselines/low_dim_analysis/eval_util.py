# def get_param_traj_file_path(dir_name, net_name, index):
#     return f'{dir_name}/{net_name}_{index}.txt'
import os
from datetime import datetime
def get_current_timestamp():
    return datetime.now().strftime('%Y-%m-%d-%H:%M:%S')



def get_project_dir():
    project_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
    return project_dir


def get_run_name(args):
    if args.additional_notes == "":
        add_note = ""
    else:
        add_note = f'_additional_notes_{args.additional_notes}'

    return f'optimizer_{args.optimizer}_env_{args.env}_time_step_{args.num_timesteps}_' \
           f'normalize_{args.normalize}_n_steps_{args.n_steps}_nminibatches_{args.nminibatches}_seed_{args.seed}' \
           f'_run_{args.run_num}' \
           f'{add_note}'

def get_dir_path_for_this_run(args, proj_dir=None):
    if proj_dir is not None:
        return f'{proj_dir}/stable_baselines/{args.alg}/{get_run_name(args)}'
    else:
        return f'{get_project_dir()}/stable_baselines/{args.alg}/{get_run_name(args)}'



def get_log_dir(this_run_dir):
    return f"{this_run_dir}/the_log_dir"

def get_save_dir(this_run_dir):
    return f"{this_run_dir}/the_save_dir"

def get_test_data_dir(this_run_dir):

    return f"{this_run_dir}/test_data"

def get_full_params_dir(this_run_dir):

    return f"{this_run_dir}/full_params"

def get_aug_plot_dir(this_run_dir):

    return f"{this_run_dir}/aug_plot_dir"

def get_intermediate_data_dir(this_run_dir, params_scope="pi"):

    return f"{this_run_dir}/{params_scope}_intermediate_data"


def get_eval_losses_file_path(dir_name, total_timesteps):
    return f'{dir_name}/eval_loss_{total_timesteps}.hdf5'

def get_full_param_traj_file_path(dir_name, index):
    return f'{dir_name}/all_params_{index}.txt'






def get_plot_dir(args):
    return f'{get_project_dir()}/plots/{args.alg}/{get_current_timestamp()}_{get_run_name(args)}'


def get_cma_plot_dir(plot_dir, n_comp_to_use, run_num, origin):
    return f'{plot_dir}/cma/cma_n_comp_{n_comp_to_use}_origin_{origin}_run_num_{run_num}'

def get_cma_and_then_ppo_plot_dir(plot_dir, pca_indexes, run_num, cma_num_steps, ppo_num_steps, origin):
    return f'{plot_dir}/cma_and_then_ppo/cma_and_then_ppo_pca_indexes_{pca_indexes}' \
           f'_ppo_num_steps_{ppo_num_steps}_cma_num_steps_{cma_num_steps}_origin_{origin}_run_num_{run_num}'

def get_other_pcs_plane_plot_dir(plot_dir, other_pcs):
    return f'{plot_dir}/other_pcs_{other_pcs}'


def get_ppos_plot_dir(plot_dir, n_comp_to_use, cma_run_num):
    return f'{plot_dir}/ppos/ppos_n_comp_{n_comp_to_use}_run_num_{cma_run_num}'

def get_first_n_pc1_vs_V_plot_dir(plot_dir, granularity):
    return f'{plot_dir}/first_n_pc1_vs_V/first_n_pc1_vs_V_granularity_{granularity}'
def get_plane_angles_vs_final_plane_along_the_way_plot_dir(plot_dir, n_comp_to_use):
    return f'{plot_dir}/plane_angles_vs_final_plane/plane_angles_vs_final_plane_n_comp_to_use_{n_comp_to_use}'


def get_pcs_filename(intermediate_dir, n_comp):

    return f"{intermediate_dir}/n_comp_{n_comp}_pcs"
def get_mean_param_filename(intermediate_dir):

    return f"{intermediate_dir}/mean_param"

def get_explain_ratios_filename(intermediate_dir, n_comp):

    return f"{intermediate_dir}/n_comp_{n_comp}_explain_ratios"
def get_projected_full_path_filename(intermediate_dir, n_comp, pca_center, which_components=(1,2)):

    return f"{intermediate_dir}/n_comp_{n_comp}_pca_center_{pca_center}_which_components_{which_components}_projected_full_path"
def get_eval_returns_filename(intermediate_dir, eval_string, n_comp, pca_center, which_components=(1,2)):

    return f"{intermediate_dir}/{eval_string}_n_comp_{n_comp}_pca_center_{pca_center}_which_components_{which_components}eval_returns"
def get_projected_finals_eval_returns_filename(intermediate_dir, n_comp_start, np_comp_end, pca_center):

    return f"{intermediate_dir}/n_comp_start_{n_comp_start}_np_comp_end_{np_comp_end}_pca_center_{pca_center}eval_returns"

def get_cma_returns_dirname(intermediate_dir, n_comp, run_num):

    return f"{intermediate_dir}/cma/cma_n_comp_{n_comp}_run_num_{run_num}"
def get_ppos_returns_dirname(intermediate_dir, n_comp, run_num):

    return f"{intermediate_dir}/ppos/ppos_n_comp_{n_comp}_run_num_{run_num}"

def get_cma_and_then_ppo_run_dir(intermediate_dir, pca_indexes, run_num, cma_steps):

    return f"{intermediate_dir}/cma_and_then_ppo/ctp_pca_index_{pca_indexes}_cma_steps_{cma_steps}_run_num_{run_num}"

def get_ppo_part(this_run_dir):
    return f"{this_run_dir}/ppo_part"

if __name__ == '__main__':
    print(get_log_dir("a", 1, "s", False, 0))