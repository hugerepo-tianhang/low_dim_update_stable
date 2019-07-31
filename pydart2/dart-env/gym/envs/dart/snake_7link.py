import numpy as np
from gym import utils
from gym.envs.dart import dart_env

from matplotlib import pyplot as plt


class DartSnake7LinkEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self,num_inds_to_add=0):
        self.control_bounds = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]])
        self.action_scale = 200
        self.include_action_in_obs = False
        self.randomize_dynamics = False

        obs_dim = 17
        if self.include_action_in_obs:
            obs_dim += len(self.control_bounds[0])
            self.prev_a = np.zeros(len(self.control_bounds[0]))


        obs_dim += num_inds_to_add
        dart_env.DartEnv.__init__(self, 'snake_7link.skel', 4, obs_dim, self.control_bounds, disableViewer=True)

        if self.randomize_dynamics:
            self.bodynode_original_masses = []
            self.bodynode_original_frictions = []
            for bn in self.robot_skeleton.bodynodes:
                self.bodynode_original_masses.append(bn.mass())
                self.bodynode_original_frictions.append(bn.friction_coeff())

        self.dart_world.set_collision_detector(3)

        for i in range(0, len(self.robot_skeleton.bodynodes)):
            self.robot_skeleton.bodynodes[i].set_friction_coeff(0)
        self.robot_skeleton.bodynodes[-1].set_friction_coeff(0)

        utils.EzPickle.__init__(self)

        # self.Q_history = []
        self.steps = 0

    def do_simulation(self, tau, n_frames):
        for _ in range(n_frames):
            for bn in self.robot_skeleton.bodynodes:
                bn_vel = bn.com_spatial_velocity()
                norm_dir = bn.to_world([0, 0, 1]) - bn.to_world([0, 0, 0])
                vel_pos = bn_vel[3:] + np.cross(bn_vel[0:3], norm_dir) * 0.05
                vel_neg = bn_vel[3:] - np.cross(bn_vel[0:3], norm_dir) * 0.05
                fluid_force = [0.0, 0.0, 0.0]
                if np.dot(vel_pos, norm_dir) > 0.0:
                    fluid_force = -20.0 * np.dot(vel_pos, norm_dir) * norm_dir
                if np.dot(vel_neg, norm_dir) < 0.0:
                    fluid_force = -20.0 * np.dot(vel_neg, norm_dir) * norm_dir
                bn.add_ext_force(fluid_force)

            self.robot_skeleton.set_forces(tau)
            self.dart_world.step()

    def advance(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]

        if self.include_action_in_obs:
            self.prev_a = np.copy(clamped_control)

        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale

        self.do_simulation(tau, self.frame_skip)

    def step(self, a):


        self.steps += 1
        pre_state = [self.state_vector()]

        posbefore = self.robot_skeleton.q[0]
        self.advance(a)


        # self.Q_history.append(self.robot_skeleton.q.reshape(-1,1))


        posafter = self.robot_skeleton.q[0]
        deviation = self.robot_skeleton.q[2]

        alive_bonus = 0.1
        reward = (posafter - posbefore) / self.dt
        if reward > 30:
            print("s")
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        reward -= np.abs(deviation) * 0.1
        s = self.state_vector()
        self.accumulated_rew += reward
        self.num_steps += 1.0
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and abs(deviation) < 1.5)
        ob = self._get_obs()

        # if self.steps > 800:
        #     self.Q_history = np.hstack(self.Q_history)
        #     for i in range(self.Q_history.shape[0]):
        #         fig = plt.figure()
        #         plt.plot(range(len(self.Q_history[i,:])), self.Q_history[i,:])
        #         fig.savefig(f"2600{i}.png")
        #
        #     print("s")


        return ob, reward, done, {}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])

        if self.include_action_in_obs:
            state = np.concatenate([state, self.prev_a])

        return state


    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        state = self._get_obs()

        if self.include_action_in_obs:
            self.prev_a = np.zeros(len(self.control_bounds[0]))

        self.accumulated_rew = 0.0
        self.num_steps = 0.0

        if self.randomize_dynamics:
            for i in range(len(self.robot_skeleton.bodynodes)):
                self.robot_skeleton.bodynodes[i].set_mass(
                    self.bodynode_original_masses[i] + np.random.uniform(-1.5, 1.5))
                self.robot_skeleton.bodynodes[i].set_friction_coeff(
                    self.bodynode_original_frictions[i] + np.random.uniform(-0.5, 0.5))

        return state

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5

    def get_contact_bodynode(self):


        return self.robot_skeleton.body("h_head")

class DartSnake7LinkEnv_aug_input(DartSnake7LinkEnv):
    def __init__(self, lagrangian_inds_to_include):

        if lagrangian_inds_to_include is None:
            raise Exception("don't give none, give empty list")
        else:
            self.lagrangian_inds_to_include = lagrangian_inds_to_include

        num_inds_to_add = len(self.lagrangian_inds_to_include)

        super().__init__(num_inds_to_add=num_inds_to_add)

    def _get_obs(self):
        state = super()._get_obs()

        lagrangian_to_add = []
        for (key, ind) in self.lagrangian_inds_to_include:
            flat_array = self.get_lagrangian_flat_array(key)
            lagrangian_to_add.append(flat_array.reshape(-1)[ind])

        state = np.hstack((state, np.array(lagrangian_to_add)))
        return state
