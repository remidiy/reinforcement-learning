""" SARSAMAX updates """

import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

if "../" not in sys.path:
	sys.path.append("../") 

from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

matplotlib.style.use('ggplot')


env = CliffWalkingEnv()

def epsilon_greedy_policy(Q, epsilon, num_actions):

	def policy(observation):
		A = np.ones(num_actions, dtype=float) * epsilon/num_actions
		best_action = np.argmax(Q[observation])
		A[best_action] += 1.0 - epsilon
		return A

	return policy


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
	Q = defaultdict(lambda: np.zeros(env.action_space.n))

	behaviour_policy = epsilon_greedy_policy(Q, epsilon, env.action_space.n)

	episode_lengths = np.zeros(num_episodes, dtype=float)
	episode_rewards = np.zeros(num_episodes, dtype=float)

	for episode in range(num_episodes):
		if episode % 10 == 0:
			print('\repisode: {}/{}'.format(episode + 1, num_episodes), end='')
			sys.stdout.flush()

		state = env.reset()
		while True:
			action = np.random.choice(np.arange(env.action_space.n), p=behaviour_policy(state))
			next_state, reward, done, _ = env.step(action)
			Q[state][action] += alpha * (reward + discount_factor * np.max(Q[next_state]) - Q[state][action])
			episode_lengths[episode] += 1
			episode_rewards[episode] += reward
			if done:
				break
			state = next_state

	stats = plotting.EpisodeStats(episode_lengths=episode_lengths, episode_rewards=episode_rewards)
	return Q, stats


Q, stats = q_learning(env, 1000)
plotting.plot_episode_stats(stats)