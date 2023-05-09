import numpy as np
from typing import Optional

from gym import core, spaces
from stock_dynamics import geometric_brownian_motion

class BasketOptionHedging(core.Env):
    """
    This class models basket call option using a cash-flow formulation

    1. The State is specified using a flat array of length = 2*num_stocks + 2. The state is defined to be of the concat-
        enation of the following subsets
        State = [S, H, C, T] 
        S = The spot_price for each stock (len = num_stocks)
        H = The holding of each stock (len = num_stocks)
        C = The current cash-holding (len = 1)
        T = time to maturity (len = 1)

    2. The Action (A) is defined to be a flat-array of length = num_stocks + 1. It consists of A = [H, C]
        H = The readjustment of stock holding of each stock for the current step
        C = The cash-holding. Note that this is only set at the initial step, for intermediary steps it is adjusted
        automatically in order to preserve the self-financing condition. i.e, the amount needed for readjusting the
        stock holding is drawn from the cash-holding. We have added time to maturity, T, so that the actor-network
        "knows" what are the initial and intermediate steps are.

    3. The reward is formulated to be 0 for all steps except the terminal step (pay-off). For the terminal step,
        the reward is negative of the square of the sum of liquidation of stock-holdings, cash-account and the 
        final payoff of the option
    """
    
    def __init__(
        self,
        num_steps=10,
        num_stocks=10,
        spot_init=1.0,
        strike=1.0,
        covariance=0.01,
        mu=0.0,
        ppy=252,
        dtype=np.float32,
        price_max=2.0,
        price_min=0.0,
        cash_max=10.0,
        cash_min=-10.0,
        spot_holding_max=2.0,
        spot_holding_min=-2.0,
        option_price=1.5,
        multi=1000.0,
        constant=1.0,
        seed=1024,
    ):
        """
        :param num_steps: total number of time steps.
        :param num_stocks: number of stocks considered
        :param spot_init: spot price
        :param strike: option strike price
        :param covariance: stock covariance
        :param mu: annual drift of assets
        :param ppy: point per year. 1/ppy is the unit of the num_steps
        :param dtype: data type
        :param price_max: a clipping threshold for maximum price of observation
        :param price_min: a clipping threshold for minimum price of observation
        :param spot_holding_max: a clipping threshold for maximum holding of observation and action
        :param spot_holding_min: a clipping threshold for minimum holding of observation and action
        :cash_max: a clipping threshold for maximum cash of observation
        :cash_min: a clipping threshold for minimum cash of observation
        :param multi: multiplier for reward
        :param constant: like the advantage in policy gradient algorithm. Add to the reward so it is really rewarded.
        :param seed: random seed
        """
        self.num_steps = num_steps
        self.num_stocks = num_stocks
        self.spot_init = spot_init
        self.strike = strike
        self.covariance = covariance
        self.mu = mu
        self.ppy = ppy
        self.dtype = dtype
        self.price_max = price_max
        self.price_min = price_min
        self.cash_max = cash_max
        self.cash_min = cash_min
        self.spot_holding_max = spot_holding_max
        self.spot_holding_min = spot_holding_min
        self.option_price = option_price
        self.multi = multi
        self.constant = constant
        self.np_random = np.random.default_rng(seed)

        obs_low = np.zeros((2 * num_stocks + 2))
        obs_low[:num_stocks] = price_min
        obs_low[num_stocks:2 * num_stocks] = spot_holding_min
        obs_low[-2] = cash_min
        obs_low[-1] = 0
        obs_high = np.zeros((2 * num_stocks + 2))
        obs_high[:num_stocks] = price_max
        obs_high[num_stocks:2 * num_stocks] = spot_holding_max
        obs_high[-2] = cash_max
        obs_high[-1] = num_steps / ppy

        action_low = np.zeros((num_stocks + 1))
        action_low[:num_stocks] = spot_holding_min
        action_low[-1] = cash_min
        action_high = np.zeros((num_stocks + 1))
        action_high[:num_stocks] = spot_holding_max
        action_high[-1] = cash_max

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(2 * num_stocks + 2,), dtype=dtype)
        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(num_stocks + 1,), dtype=dtype)

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        When end of episode is reached, you are responsible for calling :meth:`reset` to reset this environment's state.
        Accepts an action and returns either a tuple `(observation, reward, done, info)`.
        Args:
            action (numpy array): [H, C] of shape (num_stocks + 1,)
        Returns:
            observation (numpy array): S, of shape (2 * num_stocks + 2,).
            reward (float): The amount of reward returned as a result of taking the action.
            done (bool): A boolean value for if the episode has ended, in which case further :meth:`step` calls will return undefined results.
                A done signal may be emitted for the following reason: a certain timelimit was exceeded.
            info (dictionary): `info` contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                It also can contain information that distinguishes truncation and termination, however this is deprecated in favour
                of returning two booleans, and will be removed in a future version.
        """
        
        H = action[:self.num_stocks]
        prev_H = self.state[self.num_stocks: 2 * self.num_stocks]
        prev_S = self.state[:self.num_stocks]

        instantaneous_cash_flow = np.sum((prev_H - H) * prev_S)
        self.state[-2] += instantaneous_cash_flow
        
        self.state[:self.num_stocks] = self.stock_price[0, :, self.timestep + 1]
        self.state[self.num_stocks:2 * self.num_stocks] = H
        self.state[-1] -= 1. / self.ppy

        self.timestep += 1

        reward = 0
        done = False
        if self.timestep >= self.num_steps:
            reward = -self.multi * self.state[-2] * self.state[-2] + self.constant
            done = True

        return self.state, reward, done, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None):
        
        self.timestep = 0
        self.state = np.zeros(2 * self.num_stocks + 2)
        self.state[:self.num_stocks] = self.spot_init
        self.state[-1] = self.num_steps / self.ppy  # time to maturity

        self.stock_price = geometric_brownian_motion(self.num_stocks, self.num_steps, 1, self.covariance, self.mu, \
                                                     self.spot_init, self.ppy, self.np_random)
        
    def render(self):
        pass

    def close(self):
        pass