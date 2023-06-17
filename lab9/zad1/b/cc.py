import gym
import numpy as np 
    
env = gym.make('Acrobot-v1', render_mode="human")

observation, info = env.reset(seed=42)

for _ in range(600):
   action = env.action_space.sample()
   observation, reward, terminated, truncated, info = env.step(action)

env.close()