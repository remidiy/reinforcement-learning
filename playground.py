import numpy as np

from collections import defaultdict

a = defaultdict(float)
a['help'] += 0.2
print(a['help'])

print(np.prod([1, 2, 4]))


# minibatch = np.sample(replay_memory, batch_size)
# states, actions, rewards, next_states, dones = replay_memory
# selection_actions = np.argmax(q_estimator.predict(next_states), axis=1)
# targets = rewards + discount_factor * (target_estimator.predict(next_states)[:, selection_action])
# q_estimator.update(states, actions, targets)