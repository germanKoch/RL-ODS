import gymnasium as gym

env = gym.make("Taxi-v3")
env.action_space.seed(42)


observation, info = env.reset(seed=42)

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated or truncated:
        observation, info = env.reset()

env.close()