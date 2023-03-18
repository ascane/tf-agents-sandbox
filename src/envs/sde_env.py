import numpy as np
from typing import Optional

from gym import core, spaces

EPSILON = 1e-5
INF = 1e10

class SimpleSdeEnv(core.Env):
    # dS = mu * S_t * dt + sigma * S_t * W_t
    # for all i, W_{t,i} ~ N(0, 1)

    dt = 0.1
    n = 4
    mu = np.random.uniform(low=0.0, high=1.0, size=(n, 1))
    sigma = np.random.uniform(low=0.0, high=1.0, size=(n, 1))
    timestep_limit = 100

    def __init__(self):
        self.S = np.zeros(shape=(self.n, 1))
        self.timestep = 0
        self.action_space = spaces.Box(low=EPSILON, high=1.0, shape=(self.n,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-INF, high=INF, shape=(self.n,), dtype=np.float32)
        pass

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        When end of episode is reached, you are responsible for calling :meth:`reset` to reset this environment's state.
        Accepts an action and returns either a tuple `(observation, reward, done, info)`.
        Args:
            action (numpy array): an action w, of shape (n,)
        Returns:
            observation (numpy array): S, of shape (n,).
                This may, for instance, be a numpy array containing the positions and velocities of certain objects.
            reward (float): The amount of reward returned as a result of taking the action.
            done (bool): A boolean value for if the episode has ended, in which case further :meth:`step` calls will return undefined results.
                A done signal may be emitted for the following reason: a certain timelimit was exceeded.
            info (dictionary): `info` contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                It also can contain information that distinguishes truncation and termination, however this is deprecated in favour
                of returning two booleans, and will be removed in a future version.
        """
        action = np.array(action).reshape((-1, 1)) # shape: (n,1)
        action /= np.sum(action)  # normalise action
        assert np.abs(np.sum(action) - 1) < EPSILON

        W = np.random.normal(size=(self.n, 1))
        self.S += np.matmul(np.diag(np.squeeze(self.mu * self.dt)), self.S) + np.matmul(np.diag(np.squeeze(self.sigma * W)), self.S)
        reward = -np.var(action * self.S - self.S / self.n)
        self.timestep += 1
        info = {"timestep": self.timestep, "S": self.S, "W": W, "mu": self.mu, "sigma": self.sigma, "timestep_limit": self.timestep_limit}
        return self.S.reshape((-1)), reward, self.timestep >= self.timestep_limit, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None):
        """Resets the environment to an initial state and returns the initial observation.
        This method can reset the environment's random number generator(s) if ``seed`` is an integer or
        if the environment has not yet initialized a random number generator.
        If the environment already has a random number generator and :meth:`reset` is called with ``seed=None``,
        the RNG should not be reset. Moreover, :meth:`reset` should (in the typical use case) be called with an
        integer seed right after initialization and then never again.
        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG.
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action.
            options (optional dict): Additional information to specify how the environment is reset (optional,
                depending on the specific environment)
        Returns:
            observation (object): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        super().reset(seed=seed)
        self.S = np.random.uniform(low=0.0, high=1.0, size=(self.n, 1))
        self.timestep = 0
        return self.S.reshape((-1))

    def render(self):
        pass

    def close(self):
        pass
