import gymnasium as gym
import time
import numpy as np
import random
from matplotlib import pyplot as plt
from gymnasium import Env

env = gym.make("Taxi-v3")
env.action_space.seed(42)
np.random.seed(42)
random.seed(42)

n_actions = 6
n_states = 500
iterations = 100
n_trajectories = 100
q = 0.3
alpha = 0.01


def render(env: Env, observation, reward, action):
    env.render()
    print("OBSERAVTION: ", observation, " REWARD: ", reward, " ACTION: ", action)
    time.sleep(0.1)


class CrossEntropyAgent():
    
    def __init__(self, n_actions, n_states, q):
        self.n_actions = n_actions
        self.n_states = n_states
        self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions
        self.quantile = q
        
    def action(self, state):
        probs = self.policy[state]
        action = np.random.choice(np.arange(self.n_actions), p=probs).item()
        return action
    
    def fit(self, trajectories):
        total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
        quantile = np.quantile(total_rewards, self.quantile)
        elite_trajectories = []
        
        for trajectory, total in zip(trajectories, total_rewards):
            if total > quantile:
                elite_trajectories.append(trajectory)
        
        new_policy = np.zeros((self.n_states, self.n_actions))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_policy[state][action] += 1

        new_policy += alpha
        for state in range(self.n_states):
            if np.sum(new_policy[state]) > 0:
                new_policy[state] /= np.sum(new_policy[state])
            else:
                new_policy[state] = self.policy[state].copy()

        self.policy = new_policy
        return np.mean(total_rewards)

agent = CrossEntropyAgent(n_actions, n_states, q)

def get_trajectory(visualize = False):
    observation, info = env.reset()
    trajectory = {'states': [], 'actions': [], 'rewards': []}
    
    for _ in range(1000):
        action = agent.action(observation)
        trajectory['states'].append(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        
        if visualize:
            render(env, observation, reward, action)
        
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)
        if terminated or truncated:
            break
    return trajectory

totals = []
for j in range(iterations):
    trajectories = [get_trajectory() for _ in range(n_trajectories)]
    average_total = agent.fit(trajectories)
    
    print('Average total: ', average_total)
    totals.append(average_total)
    


env.close()
env = gym.make("Taxi-v3", render_mode='human')

plt.plot(totals)
plt.show()
get_trajectory(True)
env.close()