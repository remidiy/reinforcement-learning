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

def create_random_policy(num_actions):
	A = np.ones(num_actions, dtype=float) / num_actions
	def policy_fn(observation):
		return A
	return policy_fn

def create_greedy_policy(Q, num_actions):
	def policy_fn(observation):
		A = np.zeros(num_actions)
		action = np.argmax(Q[observation])
		A[action] = 1.0
		return A
	return policy_fn

def control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
	Q = defaultdict(lambda: np.zeros(env.action_space.n))
	return_sum = defaultdict(float)
	return_count = defaultdict(float)

	for episode in range(num_episodes):
		if episode % 10000 == 0:
			print('episode: {}/{}'.format(episode + 1, num_episodes))
		state = env.reset()
		episode = []
		while True:
			mu_probs = behavior_policy(state)
			target_probs = create_greedy_policy(Q, env.action_space.n)(state)
			action = np.random.choice(np.arange(env.action_space.n), p=mu_probs)
			next_state, reward, done, _ = env.step(action)
			episode.append((state, action, mu_probs, target_probs, reward))
			if done:
				break
			state  = next_state

		for i, e in enumerate(episode):
			G = sum([x[4] * (discount_factor ** k) for k, x in enumerate(episode[i:])])
			importance = np.prod([np.max(x[3]/np.max(x[2])) for x in episode[i:]])

			sa = (e[0], e[1])
			return_sum[sa] += G * importance
			return_count[sa] += 1.0
			Q[e[0]][e[1]] = return_sum[sa]/return_count[sa]


	target_policy = create_greedy_policy(Q, env.action_space.n)

	return Q, target_policy

random_policy = create_random_policy(env.action_space.n)
Q, policy = control_importance_sampling(env, num_episodes=500000, behavior_policy=random_policy)

V = defaultdict(float)
for state, action_values in Q.items():
    action_value = np.max(action_values)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")