import gym

env = gym.make('Taxi-v3', render_mode="human")

actions = [1, 1, 1, 4, 3, 3, 0, 0, 3, 3, 1, 1, 5]

observation, info = env.reset(seed=42)

for _ in range(len(actions)):
   action = actions[_]
   observation, reward, terminated, truncated, info = env.step(action)
   
   if terminated or truncated:
      observation, info = env.reset()
env.close()

