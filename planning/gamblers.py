import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append('../')

def value_iteration_gamblers(p_h, theta=0.0001, discount_factor=1.0):

	rewards = np.zeros(101)
	rewards[100] = 1 

	V = np.zeros(101)

	def lookahead(state, V, rewards):
		A = np.zeros(101)
		stakes = range(1, min(s, 100-s)+1)

		for a in stakes:
			A[a] = p_h * (rewards[s+a] + V[s+a]*discount_factor) + (1-p_h) * (rewards[s-a] + V[s-a]*discount_factor)

		return A

	while True:
		delta = 0

		for s in range(1, 100):
			A = lookahead(s, V, rewards)
			best_action = np.max(A)

			delta = max(delta, np.abs(best_action - V[s]))
			V[s] = best_action

		if delta < theta:
			break

	policy = np.zeros(100)

	for s in range(1, 100):
		A = lookahead(s, V, rewards)
		best_action = np.argmax(A)
		policy[s] = best_action

	return policy, V
	


policy, v = value_iteration_gamblers(0.25)

print("Optimized Policy:")
print(policy)
print("")

print("Optimized Value Function:")
print(v)
print("")

