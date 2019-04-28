import numpy as np
import sys 
sys.path.append("../")

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

random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)
print(v)
expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=4)
