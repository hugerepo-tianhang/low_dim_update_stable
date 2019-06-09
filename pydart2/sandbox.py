import gym
env = gym.make('DartWalker2d-v1')
observation = env.reset()
for i in range(100):
    observation, reward, done, envinfo = env.step(env.action_space.sample())
    env.render()