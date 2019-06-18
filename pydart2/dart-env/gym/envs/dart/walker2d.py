import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from gym.envs.dart.dart_world import DartWorld
from matplotlib import pyplot as plt
# from new_neuron_analysis.dir_tree_util import *
import os
def translate_to_lagrangian_index_and_plot(argtop, num_tri_M, num_C, num_COM, flattened_M_indes,
                                           max_over_neurons_concat, aug_plot_dir, lagrangian_values, layers_values):

    result = {"M": [], "Coriolis": [], "COM": []}
    if len(argtop) == 0:
        return result

    for ind in argtop:
        neuron_coord = max_over_neurons_concat[ind][-2:]
        linear_co =  max_over_neurons_concat[ind][0]
        normalized_SSE =  max_over_neurons_concat[ind][1]
        if ind < num_tri_M:
            lagrangian_key = "M"
            #ind is actually tri_M_index
            M_ind = flattened_M_indes[ind]
            lagrangian_index = M_ind


        elif ind < num_C + num_tri_M:
            lagrangian_key = "Coriolis"

            C_ind = ind - num_tri_M
            lagrangian_index = C_ind


        elif ind < num_tri_M + num_C + num_COM:
            lagrangian_key = "COM"

            COM_ind = ind - (num_tri_M + num_C)
            lagrangian_index = COM_ind


        else:
            raise Exception(f"WHAT? ind{ ind }")

        result[lagrangian_key].append(lagrangian_index)

        lagrangian_l = lagrangian_values[lagrangian_key][lagrangian_index]
        neuron_l = layers_values[int(neuron_coord[0]), int(neuron_coord[1]), :]
        fig_name = f"{lagrangian_key}_{lagrangian_index}_VS_layer{neuron_coord[0]}" \
                   f"_neuron_{neuron_coord[1]}_linear_co_{linear_co} normalized_SSE{normalized_SSE}.jpg"


        plot_best(lagrangian_l, neuron_l, fig_name, aug_plot_dir)

    return result


def plot_best(lagrangian_l, neuron_l, fig_name, aug_plot_dir):

    plt.figure()

    plt.scatter(lagrangian_l, neuron_l)
    plt.xlabel("lagrange")
    plt.ylabel("neuron")

    plt.savefig(f"{aug_plot_dir}/{fig_name}")
    plt.close()

import shutil
def create_dir_remove(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)


def lagrangian_to_include_in_state(linear_global_dict, non_linear_global_dict, top_to_include, aug_plot_dir,
                                   lagrangian_values,layers_values):
    linear_M_nd = np.array(linear_global_dict["M"])
    linear_C_nd = np.array(linear_global_dict["Coriolis"])
    linear_COM_nd = np.array(linear_global_dict["COM"])

    num_M = linear_M_nd.shape[0]
    num_C = linear_C_nd.shape[0]
    num_COM = linear_COM_nd.shape[0]

    n = np.sqrt(num_M)
    upper_tri_inds = np.triu_indices(n)
    flattened_ind = [int(row * n + col) for row, col in zip(upper_tri_inds[0], upper_tri_inds[1])]
    upper_tri_linear_M_nd = linear_M_nd[flattened_ind,:]
    num_tri_M = upper_tri_linear_M_nd.shape[0]

    concat = np.abs(np.vstack((upper_tri_linear_M_nd, linear_C_nd, linear_COM_nd)))


    linear_cos = concat[:,:,0]
    normalized_SSE = concat[:,:,1]
    max_normalized_SSE = 150 #hard code since > 150 will be made 0
    new_metric_matrix = 0.5*linear_cos + (1 - normalized_SSE/max_normalized_SSE) * 0.5
    argmax_for_each = np.argmax(new_metric_matrix, axis=1)

    max_over_neurons_concat = concat[np.arange(len(argmax_for_each)), argmax_for_each]
    max_for_each_lagrange = max_over_neurons_concat[:,0]


    top_to_include = min(len(max_for_each_lagrange), top_to_include)
    argtop = np.argpartition(max_for_each_lagrange, -top_to_include)[len(max_for_each_lagrange)-top_to_include:]

    create_dir_remove(aug_plot_dir)
    lagrangian_inds_to_include = translate_to_lagrangian_index_and_plot(argtop, num_tri_M, num_C, num_COM, flattened_ind,
                                                                        max_over_neurons_concat, aug_plot_dir, lagrangian_values, layers_values)



    return lagrangian_inds_to_include

class DartWalker2dEnv_aug_input(dart_env.DartEnv, utils.EzPickle):
    def __init__(self, linear_global_dict, non_linear_global_dict, top_to_include,
                 aug_plot_dir, lagrangian_values, layers_values):

        self.control_bounds = np.array([[1.0]*6,[-1.0]*6])
        self.action_scale = np.array([100, 100, 20, 100, 100, 20])


        model_paths = ['walker2d.skel']
        dt = 0.002
        # convert everything to fullpath
        full_paths = []
        for model_path in model_paths:
            if model_path.startswith("/"):
                fullpath = model_path
            else:
                fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
            if not os.path.exists(fullpath):
                raise IOError("File %s does not exist"%fullpath)
            full_paths.append(fullpath)

        if full_paths[0][-5:] == '.skel':
            self.dart_world = DartWorld(dt, full_paths[0])
        else:
            self.dart_world = DartWorld(dt)
            for fullpath in full_paths:
                self.dart_world.add_skeleton(fullpath)


        self.robot_skeleton = self.dart_world.skeletons[-1] # assume that the skeleton of interest is always the last one


        self.lagrangian_inds_to_include = lagrangian_to_include_in_state(linear_global_dict,
                                                                    non_linear_global_dict,
                                                                    top_to_include, aug_plot_dir,
                                                                         lagrangian_values,layers_values)


        num_inds_to_add = sum(map(len, self.lagrangian_inds_to_include.values()))
        obs_dim = 17 + num_inds_to_add


        dart_env.DartEnv.__init__(self, model_paths, 4, obs_dim, self.control_bounds, disableViewer=False)

        try:
            self.dart_world.set_collision_detector(3)
        except Exception as e:
            print('Does not have ODE collision detector, reverted to bullet collision detector')
            self.dart_world.set_collision_detector(2)

        utils.EzPickle.__init__(self)


    def step(self, a):
        pre_state = [self.state_vector()]



        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale
        posbefore = self.robot_skeleton.q[0]
        self.do_simulation(tau, self.frame_skip)


        # a = np.matmul(self.robot_skeleton.M, self.robot_skeleton.accelerations) + self.robot_skeleton.coriolis_and_gravity_forces() - self.robot_skeleton.forces() - self.robot_skeleton.constraint_forces()
        # assert abs(a - 0) < 1e-6

        posafter,ang = self.robot_skeleton.q[0,2]
        height = self.robot_skeleton.bodynodes[2].com()[1]

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()

        alive_bonus = 1.0
        vel = (posafter - posbefore) / self.dt
        reward = vel
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()

        # uncomment to enable knee joint limit penalty
        '''joint_limit_penalty = 0
        for j in [-2, -5]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)

        reward -= 5e-1 * joint_limit_penalty'''

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .8) and (height < 2.0) and (abs(ang) < 1.0))

        ob = self._get_obs()

        #
        # JJ = self.robot_skeleton.bodynodes[0].jacobian().T
        # for body in self.robot_skeleton.bodynodes[1:]:
        #     JJ += body.jacobian().T


        '''
            I = body->getSpatialInertia();
            J = body->getJacobian();
        
            EXPECT_EQ(I.rows(), 6);
            EXPECT_EQ(I.cols(), 6);
            EXPECT_EQ(J.rows(), 6);
            EXPECT_EQ(J.cols(), dof);
        
            M = J.transpose() * I * J;  // (dof x dof) matrix
        
            for (int j = 0; j < dof; ++j)
            {
              int jIdx = body->getDependentGenCoordIndex(j);
        
              for (int k = 0; k < dof; ++k)
              {
                int kIdx = body->getDependentGenCoordIndex(k);
        
                skelM(jIdx, kIdx) += M(j, k);
              }'''

        return ob, reward, done, {"contacts":contacts}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            np.clip(self.robot_skeleton.dq,-10,10)
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]


        lagrangian_to_add = []
        for key, inds in self.lagrangian_inds_to_include.items():
            if key == "M":
                lagrangian_to_add.extend(self.robot_skeleton.M.reshape(-1)[inds])
            elif key == "Coriolis":
                lagrangian_to_add.extend(self.robot_skeleton.c.reshape(-1)[inds])
            elif key == "COM":
                lagrangian_to_add.extend(self.robot_skeleton.C.reshape(-1)[inds])
            else:
                raise Exception("what???")

        state = np.hstack((state, np.array(lagrangian_to_add)))
        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5



class DartWalker2dEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0] * 6, [-1.0] * 6])
        self.action_scale = np.array([100, 100, 20, 100, 100, 20])

        model_paths = ['walker2d.skel']
        dt = 0.002
        # convert everything to fullpath
        full_paths = []
        for model_path in model_paths:
            if model_path.startswith("/"):
                fullpath = model_path
            else:
                fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
            if not os.path.exists(fullpath):
                raise IOError("File %s does not exist" % fullpath)
            full_paths.append(fullpath)

        if full_paths[0][-5:] == '.skel':
            self.dart_world = DartWorld(dt, full_paths[0])
        else:
            self.dart_world = DartWorld(dt)
            for fullpath in full_paths:
                self.dart_world.add_skeleton(fullpath)

        self.robot_skeleton = self.dart_world.skeletons[
            -1]  # assume that the skeleton of interest is always the last one

        obs_dim = 17

        dart_env.DartEnv.__init__(self, model_paths, 4, obs_dim, self.control_bounds, disableViewer=False)

        try:
            self.dart_world.set_collision_detector(3)
        except Exception as e:
            print('Does not have ODE collision detector, reverted to bullet collision detector')
            self.dart_world.set_collision_detector(2)

        utils.EzPickle.__init__(self)

    def step(self, a):
        pre_state = [self.state_vector()]

        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale
        posbefore = self.robot_skeleton.q[0]
        self.do_simulation(tau, self.frame_skip)
        posafter, ang = self.robot_skeleton.q[0, 2]
        height = self.robot_skeleton.bodynodes[2].com()[1]

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()

        alive_bonus = 1.0
        vel = (posafter - posbefore) / self.dt
        reward = vel
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()

        # uncomment to enable knee joint limit penalty
        '''joint_limit_penalty = 0
        for j in [-2, -5]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)

        reward -= 5e-1 * joint_limit_penalty'''

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .8) and (height < 2.0) and (abs(ang) < 1.0))

        ob = self._get_obs()

        #
        # JJ = self.robot_skeleton.bodynodes[0].jacobian().T
        # for body in self.robot_skeleton.bodynodes[1:]:
        #     JJ += body.jacobian().T


        '''
            I = body->getSpatialInertia();
            J = body->getJacobian();

            EXPECT_EQ(I.rows(), 6);
            EXPECT_EQ(I.cols(), 6);
            EXPECT_EQ(J.rows(), 6);
            EXPECT_EQ(J.cols(), dof);

            M = J.transpose() * I * J;  // (dof x dof) matrix

            for (int j = 0; j < dof; ++j)
            {
              int jIdx = body->getDependentGenCoordIndex(j);

              for (int k = 0; k < dof; ++k)
              {
                int kIdx = body->getDependentGenCoordIndex(k);

                skelM(jIdx, kIdx) += M(j, k);
              }'''

        return ob, reward, done, {"contacts": contacts}

    def _get_obs(self):
        state = np.concatenate([
            self.robot_skeleton.q[1:],
            np.clip(self.robot_skeleton.dq, -10, 10)
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]
        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5
