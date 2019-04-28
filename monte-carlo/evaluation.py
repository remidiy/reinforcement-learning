import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict

if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()

def prediction(policy, env, num_episodes, discount_factor=1.0):

	def G(rewards):
		g = 0
		for i, r in enumerate(rewards):
			g += (discount_factor ** i) * r 
		return g


	returns_sum = defaultdict(float)
	returns_count = defaultdict(float)

	V = defaultdict(float)

	for episode in range(num_episodes):
		observation = env.reset()
		states = []
		rewards = []
		done = False
		while not done:
			action = policy(observation)
			next_observation, reward, done, _ = env.step(action)
			states.append(observation)
			rewards.append(reward)
			observation = next_observation

		for state in set(states):
			first_occurence_idx = next(i for i,x in enumerate(states) if x == state)
			G = sum(rewards[first_occurence_idx] * (discount_factor ** i) for i, _ in enumerate(states[first_occurence_idx:]))

			returns_sum[state] += G
			returns_count[state] += 1.0	
			V[state] = 	returns_sum[state]/returns_count[state]
		
	return V

def sample_policy(observation):
	score, dealer_score, usable_ace = observation
	return 1 if score < 20 else 0

# print(prediction(sample_policy, env, num_episodes=100))

# V_10k = prediction(sample_policy, env, num_episodes=10000)
# plotting.plot_value_function(V_10k, title="10,000 Steps")

V_500k = prediction(sample_policy, env, num_episodes=500000)
plotting.plot_value_function(V_500k, title="500,000 Steps")
