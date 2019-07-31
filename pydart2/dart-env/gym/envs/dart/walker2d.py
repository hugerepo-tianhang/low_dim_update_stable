import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from gym.envs.dart.dart_world import DartWorld
import os



class DartWalker2dEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self, num_inds_to_add=0):
        self.control_bounds = np.array([[1.0] * 6, [-1.0] * 6])
        self.action_scale = np.array([100, 100, 20, 100, 100, 20])

        model_paths = ['walker2d.skel']
        obs_dim = 17 + num_inds_to_add
        dart_env.DartEnv.__init__(self, model_paths, 4, obs_dim, self.control_bounds, disableViewer=False)

        try:
            self.dart_world.set_collision_detector(3)
        except Exception as e:
            print('Does not have ODE collision detector, reverted to bullet collision detector')
            self.dart_world.set_collision_detector(2)

        utils.EzPickle.__init__(self)

    def get_contact_bodynode(self):
        return self.robot_skeleton.body("h_foot_left")


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





class DartWalker2dEnv_aug_input(DartWalker2dEnv):
    def __init__(self, lagrangian_inds_to_include):
        if lagrangian_inds_to_include is None:
            raise Exception("don't give none, give empty list")
        else:
            self.lagrangian_inds_to_include = lagrangian_inds_to_include

        num_inds_to_add = len(self.lagrangian_inds_to_include)

        super().__init__(num_inds_to_add=num_inds_to_add)
        # DartWalker2dEnv.__init__(self, obs_dim=obs_dim)

    def _get_obs(self):
        state = super()._get_obs()

        lagrangian_to_add = []
        for (key, ind) in self.lagrangian_inds_to_include:
            flat_array = self.get_lagrangian_flat_array(key)
            lagrangian_to_add.append(flat_array.reshape(-1)[ind])

        state = np.hstack((state, np.array(lagrangian_to_add)))
        return state

class DartWalker2dEnv_ddq_action(DartWalker2dEnv):

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

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .8) and (height < 2.0) and (abs(ang) < 1.0))

        ob = self._get_obs()

        return ob, reward, done, {"contacts": contacts}

if __name__ == "__main__":
    def check_get_upper_tri():
        linear_M_matrix = np.array([[[1,0],[2,0],[3,0]],[[2,0],[4,0],[5,0]],[[3,0],[5,0],[6,0]]])
        linear_M_nd = linear_M_matrix.reshape((-1,2))


        upper, flattened_ind = get_upper_tri(linear_M_nd)
        assert len(upper) == 6
        assert (upper == np.array([[1,0],[2,0],[3,0],[4,0],[5,0],[6,0]])).all()

    check_get_upper_tri()
    # def check_lagrangian_to_include_in_state():

