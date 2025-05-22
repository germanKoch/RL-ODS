import gymnasium as gym
import time

env = gym.make("Taxi-v3", render_mode="human")
env.action_space.seed(42)


observation, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    env.render()
    print("OBSERAVTION: ", observation, " REWARD: ", reward, " ACTION: ", action)
    time.sleep(0.1)
    if terminated or truncated:
        observation, info = env.reset()

env.close()