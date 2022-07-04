import numpy as np
import matplotlib.pyplot as plt

STEPS = 200
TRAIN_TIMES = 1000
C = 2  # Use for UCB confidence level.

class k_armed_bandit:
    def __init__(self, epsilon, UCB=False):
        self.epsilon = epsilon
        self.UCB = UCB  # Use UCB algorithm
        self.t = 0

        self.Q = np.zeros(10, dtype=np.float64)
        self.num_of_action = np.zeros(10, dtype=np.float64)
        self.reward_history = np.zeros(STEPS, dtype=np.float64)

    def run(self):
        mean_reward = 0

        for t in range(STEPS):
            self.t += 1
            action = self.choose_action()

            reward = np.random.normal(MU[action], 1)
            mean_reward = mean_reward + (reward - mean_reward) / (t + 1)

            self.reward_history[t] = mean_reward
            self.num_of_action[action] += 1
            self.Q[action] = self.Q[action] + (1 / self.num_of_action[action]) * (reward - self.Q[action])

    def choose_action(self):
        if self.UCB:
            if 0 in self.num_of_action:
                action = np.where(self.Q == 0)[0][0]
            else:
                unc = C * np.sqrt(np.log(self.t) / self.num_of_action)
                Q_temp = self.Q * unc
                action = np.argmax(Q_temp)
        else:
            if np.random.random() > self.epsilon:
                action = np.argmax(self.Q)
            else:
                action = np.random.randint(10)

        return action


reward_1 = np.zeros(STEPS)
reward_2 = np.zeros(STEPS)

for i in range(TRAIN_TIMES):
    MU = np.random.normal(0, 1, 10)

    env1 = k_armed_bandit(epsilon=0, UCB=True)
    env2 = k_armed_bandit(epsilon=0.1)

    env1.run()
    env2.run()

    reward_1 = reward_1 + (env1.reward_history - reward_1) / (i + 1)
    reward_2 = reward_2 + (env2.reward_history - reward_2) / (i + 1)

plt.plot(reward_1, label='UCB')
plt.plot(reward_2, label='epsilon=0.1')
plt.legend()
plt.grid()
plt.show()
