import gym
import numpy as np


env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])
lr = 0.8
gamma = 0.95

num_episodes = 5000

reward_list = []

for episode in range(num_episodes):
	state = env.reset()
	total_rewards = 0
	solved = False
	steps = 100

	for step in range(steps):
		action = np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n)*(1./(episode+1)))
		new_state, reward, solved, _ = env.step(action)
		Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :] - Q[state, action]))
		total_rewards += reward
		state = new_state
		if solved:
			break
	
	reward_list.append(total_rewards)

print("score overtime ({}) episodes: {:.2f}".format(num_episodes, sum(reward_list)/num_episodes))