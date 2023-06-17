import gym

env = gym.make('Ant-v4',render_mode="human")

env.reset()


for _ in range(300):
   action = env.action_space.sample()
   env.step(action)
env.close()

## NIE DZIAŁA zamistat