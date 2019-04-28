import numpy as np
import sys
sys.path.append('../')

from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()

def policy_eval(policy, env, discount_factor=1.0, theta=0.000001):
	V = np.zeros(env.nS)

	while True:
		delta = 0
		for state in range(env.nS):
			v = 0
			for action, action_prob in enumerate(policy[state]):
				for prob, next_state, reward, done in env.P[state][action]:
					v += prob * action_prob * (reward + discount_factor * V[next_state]) 
			delta = max(delta, np.abs(v - V[state]))
			V[state] = v
		if delta < theta:
			break
		else:
			print("delta : ({})".format(delta))
	return np.array(V)

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
	policy = np.ones([env.nS, env.nA]) / env.nA
	eps = 1e-8
	while True:
		stable_policy = True
		V = policy_eval(policy, env)
		new_policy = np.zeros([env.nS, env.nA])
		for state in range(env.nS):
			action_arr = []
			for action, action_prob in enumerate(policy[state]):
				for prob, next_state, reward, _ in env.P[state][action]:
					action_arr.append(prob * (reward + discount_factor * V[next_state]))

			best_a = np.argmax(np.exp(action_arr)/np.sum(np.exp(action_arr)))
			new_policy[state, :] = np.eye(env.nA)[best_a]

			if np.argmax(policy[state]) != best_a:
				stable_policy = False
		policy = new_policy

		if stable_policy:
			return policy, V

policy, v = policy_improvement(env)
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