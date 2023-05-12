import numpy as np
import unittest

from consts import NUM_STOCKS, NUM_STEPS, NUM_PATHS, COV, MU, SPOT_INIT, PPY, SEED
from stock_dynamics import geometric_brownian_motion

class StockDynamicsTest(unittest.TestCase):
    def setUp(self):
        self.np_random = np.random.default_rng(seed=SEED)

    def test_geometric_brownian_motion_shape(self):
        stock_prices = geometric_brownian_motion(NUM_STOCKS, NUM_STEPS, NUM_PATHS, COV, MU, SPOT_INIT, PPY, self.np_random)
        self.assertEqual(stock_prices.shape, (NUM_PATHS, NUM_STOCKS, NUM_STEPS + 1))

    def test_zero_spot_init(self):
        stock_prices = geometric_brownian_motion(NUM_STOCKS, NUM_STEPS, NUM_PATHS, COV, MU, 0, PPY, self.np_random)
        self.assertTrue((stock_prices == 0).all())

    def test_zero_cov(self):
        stock_prices = geometric_brownian_motion(NUM_STOCKS, NUM_STEPS, NUM_PATHS, 0, MU, SPOT_INIT, PPY, self.np_random)
        # check if the first path is identical to the second one (deterministic)
        self.assertTrue((stock_prices[0] == stock_prices[1]).all())
        # check if the value is as excepted in the deterministic case
        dt = 1 / PPY
        for step in range(NUM_STEPS):    
            self.assertAlmostEqual(stock_prices[0, 0, step], stock_prices[0, 0, 0] * np.exp(dt * step * MU))

if __name__ == "__main__":
    unittest.main()
