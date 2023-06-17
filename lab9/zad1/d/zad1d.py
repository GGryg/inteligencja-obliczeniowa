import gym

env = gym.make('FrozenLake8x8-v1', render_mode="human", is_slippery=False)

actions = [1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2]

observation, info = env.reset(seed=42)

for _ in range(len(actions)):
   action = actions[_]
   observation, reward, terminated, truncated, info = env.step(action)
   
   if terminated or truncated:
      observation, info = env.reset()
env.close()