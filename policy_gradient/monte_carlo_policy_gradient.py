""" 
REINFORCE Algorithm 
"""

import sys
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, model
import gym
import matplotlib
from collections import namedtuple

if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = CliffWalkingEnv()
valid_actions = [0, 1, 2, 3]
actions_len = len(valid_actions)


class ValueEstimator(object):
    def __init__(self):
        self.model = self._build_model()

    @staticmethod
    def state_processor(state):
        return tf.expand_dims(tf.one_hot(state, len(env.observation_space.n)), 0)

    @staticmethod
    def target_processor(target):
        return target

    @staticmethod
    def _build_model():
        model = tf.keras.model.sequential()
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        model.summary()
        return model

    def predict(self, state):
        return self.model.predict(self.state_processor(state))

    def update(self, state, target):
        self.model.train(self.state_processor(state), self.target_processor(target))


class PolicyEstimator(object):
    def __init__(self):
        pass

    def predict(self):
        pass

    def update(self):
        pass


def reinforce():
    pass












