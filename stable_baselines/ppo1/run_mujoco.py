#!/usr/bin/env python3

from stable_baselines.ppo1 import PPO1
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from stable_baselines import logger


def train(env_id, num_timesteps, seed):
    """
    Train PPO1 model for the Mujoco environment, for testing purposes

    :param env_id: (str) Environment ID
    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    """
    env = make_mujoco_env(env_id, seed)
    model = PPO1(MlpPolicy, env, timesteps_per_actorbatch=2048, clip_param=0.2, entcoeff=0.0, optim_epochs=10,
                 optim_stepsize=3e-4, optim_batchsize=64, gamma=0.99, lam=0.95, schedule='linear')
    model.learn(total_timesteps=num_timesteps)
    env.close()


def main():
    """
    Runs the test
    """
    args = mujoco_arg_parser().parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

    env = make_mujoco_env(args.env, args.seed)
    model = PPO1(MlpPolicy, env, timesteps_per_actorbatch=2048, clip_param=0.2, entcoeff=0.0, optim_epochs=10,
                 optim_stepsize=3e-4, optim_batchsize=64, gamma=0.99, lam=0.95, schedule='linear')
    model.learn(total_timesteps=args.num_timesteps)

    model.save("ppo1")
    # env.close()

    del model # remove to demonstrate saving and loading
    # env = make_mujoco_env(args.env, args.seed)

    model = PPO1.load("ppo1")
    logger.log("~!!!!!!!!")
    episode_rew = 0
    obs = env.reset()

    while True:
        action, _states = model.predict(obs)
        ob, reward, done, info = env.step(action)
        episode_rew += reward
        env.render()
        if done:
            print(f'episode_rew={episode_rew}')
            episode_rew = 0
            obs = env.reset()


if __name__ == '__main__':
    main()
