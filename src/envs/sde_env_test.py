import gym
import src  # Don't remove this. It registers the env.
import numpy as np


if __name__ == "__main__":
    env = gym.make('simple-sde-v0')

    env.reset()
    S0 = env.S
    n = S0.shape[0]

    # step 1: uniform action
    action = np.ones((n,)) / 4
    print(action)

    obs, reward, done, info = env.step(action)
    
    print(obs)
    print(reward)
    print(done)
    print(info)

    # step 2: random action
    action = np.random.uniform(size=(n,))
    action /= np.sum(action)  # sum to 1
    print(action)

    obs, reward, done, info = env.step(action)
    
    print(obs)
    print(reward)
    print(done)
    print(info)

    env.reset()
