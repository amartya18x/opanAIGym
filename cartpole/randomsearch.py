import gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def activation(inp):
    if inp > 0:
        return 1
    else:
        return 0


def complete_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    for i in xrange(500):
        action = activation(np.dot(parameters, observation))
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward

def check_better(curr_score, parameters):
    flags = 0
    total_tries = 50
    aver_score = 0
    for i in xrange(total_tries):
        new_score = complete_episode(env, parameters)
        if new_score > curr_score:
            flags += 1
            aver_score += new_score
        else:
            return False, 1
    return True, aver_score * 1.0 / flags


def train(env):
    observation = env.reset()
    parameters = np.random.randn(len(observation))
    curr_score = 0
    num_episodes = 0
    while(True):
        new_param = np.random.randn(len(observation))
        is_better, new_score = check_better(curr_score, new_param)
        if is_better:
            curr_score = new_score
            parameters = new_param
            if curr_score > 250:
                break
        num_episodes += 1
    return num_episodes, curr_score, parameters

env = gym.make('CartPole-v0')

num_tries = 1
param_arr = np.zeros((num_tries, 4))
episode_arr = np.zeros(num_tries)

for i in tqdm(xrange(num_tries)):
    num_episodes, curr_score, parameters = train(env)
    param_arr[i] = parameters
    episode_arr[i] = num_episodes
    env.reset()
rew = np.mean(episode_arr)
plt.hist(episode_arr, 50, normed=1, facecolor='green', alpha=0.75)
plt.xlabel('Number of steps')
plt.ylabel('Number of occurunces')
plt.title(r'$\mathrm{Histogram\ of\ steps:}$')
plt.grid(True)

plt.show()


env.monitor.start('cartpole-experiments/', force=True)
score = 0
for i in tqdm(xrange(100)):
    observation = env.reset()
    score += complete_episode(env, parameters)

print("Total steps taken is " + str(score/100))
env.monitor.close()
