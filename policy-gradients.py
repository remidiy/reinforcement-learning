import numpy as np
import gym
import tensorflow as tf
import pickle
import math


env = gym.make('CartPole-v0')
env.reset()

# random_episodes = 0
# random_rewards = 0

# while random_episodes < 10:
# 	env.render()
# 	observation, reward, done, info = env.step(np.random.randint(0, 2))
# 	random_rewards += reward
# 	if done:
# 		random_episodes += 1
# 		print("Reward for this episode was:",random_rewards)
# 		random_rewards = 0
# 		env.reset()

hidden_neurons = 10
batch_size = 5
learning_rate = 1e-2
gamma = 0.99

dimensions = 4

tf.reset_default_graph()
observations = tf.placeholder(tf.float32, [None,dimensions] , name="input_x")
w1 = tf.get_variable('w1', shape=[dimensions, hidden_neurons], initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, w1))

w2 = tf.get_variable('w2', shape=[hidden_neurons, 1], initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, w2)
probability = tf.nn.sigmoid(score)

tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32, [None, 1], name='input_y')
advantages = tf.placeholder(tf.float32, name='reward_signal')


log_likelihood = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))
loss = -tf.reduce_mean(log_likelihood * advantages)
new_gradients = tf.gradients(loss, tvars)

adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32,name="batch_grad1")
W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
batchGrad = [W1Grad,W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))


def discount_rewards(r):
	""" take 1D float array of rewards and compute discounted reward """
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(0, r.size)):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r


xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[]

running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
init = tf.global_variables_initializer()


with tf.Session() as session:
	rendering = False
	session.run(init)
	observation = env.reset()


	gradBuffer = session.run(tvars)    # zero grad 
	for ix,grad in enumerate(gradBuffer):
		gradBuffer[ix] = grad * 0

	while episode_number <= total_episodes:
		if reward_sum/batch_size > 100 or rendering == True : 
			env.render()
			rendering = True

		x = np.reshape(observation,[1,dimensions])
		tfprob = session.run(probability, feed_dict={observations: x})
		action = 1 if np.random.rand(1) < tfprob else 0

		xs.append(x)
		y = 1 if action == 0 else 0
		ys.append(y)

		observation, reward, done, info = env.step(action)
		reward_sum += reward

		drs.append(reward)

		if done:
			episode_number += 1
			epx = np.vstack(xs)
			epy = np.vstack(ys)
			epr = np.vstack(drs)
			tfp = tfps
			xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[]

			discounted_epr = discount_rewards(epr)
			discounted_epr -= np.mean(discounted_epr)
			discounted_epr //= np.std(discounted_epr)

			tGrad = session.run(new_gradients,feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
			for ix,grad in enumerate(tGrad):
				gradBuffer[ix] += grad
	
			if episode_number % batch_size == 0: 
				session.run(updateGrads,feed_dict={W1Grad: gradBuffer[0],W2Grad:gradBuffer[1]})

				for ix,grad in enumerate(gradBuffer):
					gradBuffer[ix] = grad * 0

				running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
				print('Average reward for episode %f.  Total average reward %f.' % (reward_sum//batch_size, running_reward//batch_size))
				
				if reward_sum//batch_size > 200: 
					print("Task solved in",episode_number,'episodes!')
					break
				
				reward_sum = 0
			
			observation = env.reset()
		
print(episode_number,'Episodes completed.')