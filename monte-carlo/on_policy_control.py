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

def epsilon_greedy_policy(Q, epsilon, num_actions):
	
	def policy_function(observation):
		score, dealer_score, usable_ace = observation
		best_action = np.argmax([Q[observation][action] for action in range(num_actions)])
		if np.random.rand() < epsilon:
			return np.random.choice(range(num_actions))
		else:
			return best_action

	return policy_function

def control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
	returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
	returns_count = defaultdict(lambda: np.zeros(env.action_space.n))
	Q = defaultdict(lambda: np.zeros(env.action_space.n))

	for episode in range(num_episodes):
		print('episode: {}/{}'.format(episode + 1, num_episodes))
		state = env.reset()
		episode = []
		while True:
			action = epsilon_greedy_policy(Q, epsilon, env.action_space.n)(state)
			next_state, reward, done, _ = env.step(action)
			episode.append((next_state, action, reward))
			if done:
				break
			state = next_state
		
		state_actions = set([(e[0], e[1]) for e in episode])
		for sa in state_actions:
			first_idx = next(i for i,e in enumerate(episode) if e[0] == sa[0] and e[1] == sa[1])
			G = sum(e[2] * (discount_factor ** i) for i, e in enumerate(episode[first_idx: ]))

			returns_sum[sa[0]][sa[1]] += G
			returns_count[sa[0]][sa[1]] += 1.0

			Q[sa[0]][sa[1]] = returns_sum[sa[0]][sa[1]]/returns_count[sa[0]][sa[1]]
			

	policy = epsilon_greedy_policy(Q, epsilon, env.action_space.n)

	return Q, policy


Q, policy = control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)

V = defaultdict(float)
for state, actions in Q.items():
	action_value = np.max(actions)
	V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")








