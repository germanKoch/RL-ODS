import gymnasium as gym
import time
import numpy as np
import random
import torch
import torch.nn.functional as F

from torch.optim import AdamW
from torch import nn
from matplotlib import pyplot as plt
from gymnasium import Env

env = gym.make("LunarLander-v3")
env.action_space.seed(42)
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

n_actions = 4
state_dim = 8
iterations = 500
n_trajectories = 20
traectory_length = 10000
q_initial = 0.95
q_final = 0.5

def get_trajectory():
    observation, info = env.reset()
    trajectory = {'states': [], 'actions': [], 'rewards': []}
    
    for _ in range(traectory_length):
        action = agent.action(observation)
        trajectory['states'].append(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)
        if terminated or truncated:
            break
    return trajectory

def render(env: Env, observation, reward, action):
    env.render()
    print("OBSERAVTION: ", observation, " REWARD: ", reward, " ACTION: ", action)
    time.sleep(0.1)


class CrossEntropyAgent():
    
    def __init__(self, n_actions, state_dim, q_initial, q_final, iterations):
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.q_initial = q_initial
        self.q_final = q_final
        self.iterations = iterations
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
        nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.n_actions),
        )
        self.optimizer = AdamW(self.network.parameters(), lr=0.001)
        self.loss = nn.CrossEntropyLoss()
        
    def action(self, state):
        state = torch.FloatTensor(state)
        logits = self.network(state)
        probs = F.softmax(logits, dim=0).detach().numpy()
        return np.random.choice(self.n_actions, p=probs)
    
    def fit(self, trajectories, iter_idx):
        total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
        quantile = self.current_quantile(iter_idx)
        quantile = np.quantile(total_rewards, quantile)
        elite_trajectories = []
        
        for trajectory, total in zip(trajectories, total_rewards):
            if total > quantile:
                elite_trajectories.append(trajectory)
        
        states = []
        actions = []
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                states.append(state)
                actions.append(action)

        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        predicted_actions = self.network(states_tensor)
        
        loss_val: torch.Tensor = self.loss(predicted_actions, actions_tensor)
        loss_val.backward()
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return np.mean(total_rewards)
    
    
    def current_quantile(self, iter_idx):
        frac = iter_idx / (self.iterations - 1)
        return self.q_initial * (1 - frac) + self.q_final * frac

agent = CrossEntropyAgent(n_actions, state_dim, q_initial, q_final, iterations)
totals = []

for j in range(iterations):
    trajectories = [get_trajectory() for _ in range(n_trajectories)]
    average_total = agent.fit(trajectories, j)
    
    print('Average total: ', average_total)
    totals.append(average_total)

env.close()
env = gym.make("LunarLander-v3", render_mode="human")

plt.plot(totals)
plt.show()
get_trajectory()
get_trajectory()
get_trajectory()
get_trajectory()
env.close()