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
    total_tries = 10
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
    parameters = np.random.rand(len(observation)) * 2 - 1
    curr_score = 0
    num_episodes = 0
    while(True):
        new_add = 0.1 * (np.random.rand(len(observation)) * 2 - 1)
        new_param = parameters + new_add
        is_better, new_score = check_better(curr_score, new_param)
        if is_better:
            curr_score = new_score
            parameters = new_param
            if curr_score > 200:
                break
        num_episodes += 1
        if num_episodes > 1000:
            break
    return num_episodes, curr_score, parameters


def choose_param(param_arr, episode_arr):
    for i in xrange(0, len(episode_arr)):
        if episode_arr[i] < 999:
            return param_arr[i]


def get_average_successfull(num_episodes, avoid=1001):
    sums = 0
    entries = 0
    for val in num_episodes:
        if val != avoid:
            sums += val
            entries += 1
    print entries, sums
    return sums, entries

env = gym.make('CartPole-v0')

num_tries = 10000
param_arr = np.zeros((num_tries, 4))
episode_arr = np.zeros(num_tries)

for i in tqdm(xrange(num_tries)):
    num_episodes, curr_score, parameters = train(env)
    param_arr[i] = parameters
    episode_arr[i] = num_episodes
    env.reset()

plt.hist(episode_arr, 50, normed=1, facecolor='green', alpha=0.75)
plt.xlabel('Number of steps')
plt.ylabel('Number of occurunces')
plt.title(r'$\mathrm{Histogram\ of\ steps:}$')
plt.grid(True)

plt.show()

rew = np.mean(episode_arr)
succesful, entries = get_average_successfull(episode_arr)
env.monitor.start('cartpole-experiments/', force=True)
observation = env.reset()
complete_episode(env, parameters)
print("Total steps taken is " + str(rew))
print("Total steps taken(succesfull) is " + str(succesful * 1.0 / entries))
env.monitor.close()
