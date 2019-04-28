import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

if "../" not in sys.path:
  sys.path.append("../") 

from collections import defaultdict
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = WindyGridworldEnv()

def epsilon_greedy_policy(Q, epsilon, num_actions):

	def policy(observation):
		A = np.ones(num_actions, dtype=float) * epsilon/num_actions
		best_action = np.argmax(Q[observation])
		A[best_action] += 1.0 - epsilon
		return A

	return policy


def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
	Q = defaultdict(lambda: np.zeros(env.action_space.n))

	policy = epsilon_greedy_policy(Q, epsilon, env.action_space.n)

	episode_lengths = []
	episode_rewards = []

	for episode in range(num_episodes):
		if episode % 10 == 0:
			print('\repisode: {}/{}'.format(episode + 1, num_episodes), end='')
			sys.stdout.flush()

		state = env.reset()
		action = np.random.choice(np.arange(env.action_space.n), p=policy(state))

		episode_length = 0
		episode_reward = 0
		while True:
			next_state, reward, done, _ = env.step(action)
			next_action = np.random.choice(np.arange(env.action_space.n), p=policy(next_state))
			Q[state][action] += alpha * (reward + discount_factor * Q[next_state][next_action] - Q[state][action])
			episode_length += 1
			episode_reward += reward
			if done:
				episode_lengths.append(episode_length)
				episode_rewards.append(episode_reward)
				break
			state = next_state
			action = next_action

	stats = plotting.EpisodeStats(episode_lengths=episode_lengths, episode_rewards=episode_rewards)
	return Q, stats



Q, stats = sarsa(env, 200)
plotting.plot_episode_stats(stats)
