""" 
REINFORCE Algorithm 
"""

import sys
import itertools
import numpy as np
import tensorflow as tf
import gym
import matplotlib

if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = CliffWalkingEnv()











