import gym
import src  # Don't remove this. It registers the env.
import numpy as np
import unittest

from consts import NUM_STEPS, NUM_STOCKS, SPOT_INIT, PPY, CONSTANT

class OptionHedgingTest(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('basket-option-hedging-v0')

    def test_reset(self):
        state = self.env.reset()
        self.assertEqual(state.shape, (2 * NUM_STOCKS + 2, ))
        self.assertTrue((state[:NUM_STOCKS] == SPOT_INIT).all())  # initial stock price
        self.assertTrue((state[NUM_STOCKS: 2 * NUM_STOCKS] == 0).all())  # initial holding of each stock, assumed to be 0
        self.assertEqual(state[-2], 0)  # initial cash-holding
        self.assertAlmostEqual(state[-1], NUM_STEPS / PPY)  # initial time to maturity

    def test_step_with_zero_holding_and_non_zero_cash(self):
        state = self.env.reset()
        action = np.zeros((NUM_STOCKS + 1))
        cash_init = -1
        action[-1] = cash_init
        state, reward, done, _ = self.env.step(action)
        self.assertTrue((state[NUM_STOCKS: 2 * NUM_STOCKS] == 0).all())  # holding still 0
        self.assertEqual(state[-2], cash_init)
        self.assertAlmostEqual(state[-1], (NUM_STEPS - 1) / PPY)
        self.assertEqual(reward, 0)
        self.assertFalse(done)

    def test_step_with_const_holding(self):
        state = self.env.reset()
        action = np.zeros((NUM_STOCKS + 1))
        new_holding = 1
        action[:-1] = 1
        state, reward, done, _ = self.env.step(action)
        self.assertTrue((state[NUM_STOCKS: 2 * NUM_STOCKS] == new_holding).all())
        self.assertAlmostEqual(state[-1], (NUM_STEPS - 1) / PPY)
        self.assertEqual(reward, 0)
        self.assertFalse(done)

    def test_step_until_last_step(self):
        state = self.env.reset()
        action = np.zeros((NUM_STOCKS + 1))
        for _ in range(NUM_STEPS):
            state, reward, done, _ = self.env.step(action)
        self.assertTrue((state[NUM_STOCKS: 2 * NUM_STOCKS] == 0).all())
        self.assertAlmostEqual(state[-1], 0)
        self.assertLessEqual(reward, CONSTANT)
        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()
