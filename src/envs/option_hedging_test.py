import gym
import src  # Don't remove this. It registers the env.
import numpy as np


if __name__ == "__main__":
    
    env = gym.make('basket-option-hedging-v0')

    S0 = env.reset()
    num_stocks = int((S0.shape[0] - 2) / 2)

    # step 1: uniform action
    action = np.zeros((num_stocks + 1,))
    action[:num_stocks] = np.ones((num_stocks,)) / num_stocks
    print(action)

    obs, reward, done, info = env.step(action)
    
    print(obs)
    print(reward)
    print(done)
    print(info)

    # step 2: random action
    action = np.zeros((num_stocks + 1,))
    action[:num_stocks] = np.random.uniform(size=(num_stocks,))
    action /= np.sum(action)  # sum to 1
    print(action)

    obs, reward, done, info = env.step(action)
    
    print(obs)
    print(reward)
    print(done)
    print(info)

    env.reset()
