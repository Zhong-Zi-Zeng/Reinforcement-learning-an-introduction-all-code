import numpy as np
import matplotlib.pyplot as plt

STEPS = 200
TRAIN_TIMES = 1000

class k_armed_bandit:
    def __init__(self, epsilon, initial_value=0):
        self.epsilon = epsilon

        self.Q = np.full(10, fill_value=initial_value, dtype=np.float64)
        self.num_of_action = np.zeros(10)
        self.reward_history = np.zeros(STEPS)

    def run(self):
        mean_reward = 0

        for t in range(STEPS):
            if np.random.random() > self.epsilon:
                action = np.argmax(self.Q)
            else:
                action = np.random.randint(10)

            reward = np.random.normal(MU[action], 1)
            mean_reward = mean_reward + (reward - mean_reward) / (t + 1)

            self.reward_history[t] = mean_reward
            self.num_of_action[action] += 1
            self.Q[action] = self.Q[action] + (1 / self.num_of_action[action]) * (reward - self.Q[action])


reward_1 = np.zeros(STEPS)
reward_2 = np.zeros(STEPS)

for i in range(TRAIN_TIMES):
    MU = np.random.normal(0, 1, 10)

    env1 = k_armed_bandit(epsilon=0, initial_value=5)
    env2 = k_armed_bandit(epsilon=0.1)

    env1.run()
    env2.run()

    reward_1 = reward_1 + (env1.reward_history - reward_1) / (i + 1)
    reward_2 = reward_2 + (env2.reward_history - reward_2) / (i + 1)

plt.plot(reward_1, label='epsilon=0,   initial_value=5')
plt.plot(reward_2, label='epsilon=0.1, initial_value=0')
plt.legend()
plt.grid()
plt.show()
