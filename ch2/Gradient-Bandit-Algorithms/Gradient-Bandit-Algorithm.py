import numpy as np
import matplotlib.pyplot as plt

STEPS = 500
TRAIN_TIMES = 1000


class k_armed_bandit:
    def __init__(self, alpha, use_baseline):
        self.alpha = alpha
        self.use_baseline = use_baseline

        self.H = np.zeros(10, dtype=np.float64)
        self.reward_history = np.zeros(STEPS, dtype=np.float64)

    def run(self):
        mean_reward = 0

        for t in range(STEPS):
            action, prob = self.choose_action()

            reward = np.random.normal(MU[action], 1)
            mean_reward = mean_reward + (reward - mean_reward) / (t + 1)

            self.reward_history[t] = mean_reward
            self.update_h(action, reward, mean_reward, prob)

    def update_h(self, action, reward, mean_reward, prob):
        for a in range(10):
            if self.use_baseline:
                if a == action:
                    self.H[a] = self.H[a] + self.alpha * (reward - 4) * (1 - prob[a])
                else:
                    self.H[a] = self.H[a] - self.alpha * (reward - 4) * prob[a]
            else:
                if a == action:
                    self.H[a] = self.H[a] + self.alpha * reward * (1 - prob[a])
                else:
                    self.H[a] = self.H[a] - self.alpha * reward * prob[a]

    def choose_action(self):
        prob = self.softmax()
        action = np.random.choice(np.arange(10), p=prob)

        return action, prob

    def softmax(self):
        numerator = np.exp(self.H - np.max(self.H))
        denominator = np.sum(numerator)

        return numerator / denominator


reward_1 = np.zeros(STEPS)
reward_2 = np.zeros(STEPS)
reward_3 = np.zeros(STEPS)
reward_4 = np.zeros(STEPS)

for i in range(TRAIN_TIMES):
    MU = np.random.normal(4, 1, 10)

    env1 = k_armed_bandit(alpha=0.1, use_baseline=False)
    env2 = k_armed_bandit(alpha=0.1, use_baseline=True)
    env3 = k_armed_bandit(alpha=0.4, use_baseline=False)
    env4 = k_armed_bandit(alpha=0.4, use_baseline=True)

    env1.run()
    env2.run()
    env3.run()
    env4.run()

    reward_1 = reward_1 + (env1.reward_history - reward_1) / (i + 1)
    reward_2 = reward_2 + (env2.reward_history - reward_2) / (i + 1)
    reward_3 = reward_3 + (env3.reward_history - reward_3) / (i + 1)
    reward_4 = reward_4 + (env4.reward_history - reward_4) / (i + 1)

plt.plot(reward_1, label='alpha=0.1, no_baseline')
plt.plot(reward_2, label='alpha=0.1, use_baseline')
plt.plot(reward_3, label='alpha=0.4, no_baseline')
plt.plot(reward_4, label='alpha=0.4, use_baseline')
plt.legend()
plt.grid()
plt.show()
