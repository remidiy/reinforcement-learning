import numpy as np
import sys
sys.path.append('../')

from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()

def value_iteration(env, theta=0.0001, discount_factor=1.0):

	V = np.zeros(env.nS)
	policy = np.zeros([env.nS, env.nA])
	while True:
		delta = 0.0
		for state in range(env.nS):
			v = []
			for action in range(env.nA):
				for prob, next_state, reward, done in env.P[state][action]:
					v.append(prob * (reward + discount_factor * V[next_state]))
			best_action_value = np.max(v)
			delta = max(delta, np.abs(best_action_value - V[state]))
			V[state] = best_action_value
		
		if delta < theta:
			for state in range(env.nS):
				v = []
				for action in range(env.nA):
					for prob, next_state, reward, done in env.P[state][action]:
						v.append(prob * (reward + discount_factor * V[next_state]))

				best_action = np.argmax(v)
				policy[state, best_action] = 1.0
			return policy, V


policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")


expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)