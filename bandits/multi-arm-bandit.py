"""
Introduction to reinforcement learning -> rewards are only from the action no states
are involved.

actions --> rewards 
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np


bandit_arms = [0.2,0,-0.2,-2]
num_arms = len(bandit_arms)

def pullbandit(bandit):
	res = np.random.randn(1)

	if res > bandit:
		return 1
	else:
		return -1


tf.reset_default_graph()

weights = tf.Variable(tf.ones([num_arms]))
output = tf.nn.softmax(weights)

reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)

responsible_output = tf.slice(output, action_holder, [1])
loss = -(tf.log(responsible_output)*reward_holder)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
update = optimizer.minimize(loss)


total_episodes = 1000
total_rewards = np.zeros(num_arms)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	i = 0
	while i < total_episodes:
		actions  = sess.run(output)
		a = np.random.choice(actions, p=actions)
		action = np.argmax(actions == a)

		reward = pullbandit(bandit_arms[action])
		_, resp, ww = sess.run([update,responsible_output,weights], feed_dict={reward_holder:[reward],action_holder:[action]})

		total_rewards[action] += reward

		if i % 50 == 0:
			print("Running reward for the " + str(num_arms) + " arms of the bandit: " + str(total_rewards))
		i+=1
print("\nThe agent thinks arm " + str(np.argmax(ww)+1) + " is the most promising....")
if np.argmax(ww) == np.argmax(-np.array(bandit_arms)):
	print("...and it was right!")
else:
	print("...and it was wrong!")

