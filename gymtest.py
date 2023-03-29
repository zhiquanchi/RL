import gym

env = gym.make('CarRacing-v2')

import time

obs = env.reset()
for i in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.01)
    if done:
        print("Episode finished after {} timesteps".format(i+1))
        break

action = [0.0, 1.0, 0.0] # 向右转弯
obs, reward, done, info = env.step(action)

print("Action space:", env.action_space)
print("Observation space:", env.observation_space)
