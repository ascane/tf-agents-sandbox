import numpy as np

def geometric_brownian_motion(num_stocks, num_steps, num_paths, cov, mu, spot_init, ppy, np_random):
    """
    Args:
        num_stocks: number of correlated stocks
        num_steps: number of time steps
        num_paths: number of simulations
        cov: covariance matrix. If float value (identity * value) is used as covariance
        mu: drift vector. If float value constant vector of the said value is used
        spot_init: vector of initial prices: If float all stocks start with this price
        ppy: point per year. 1/ppy is the unit of the num_steps
        np_random: numpy random generator

    Returns:
        array with dimensions:  [num_paths * num_stocks * num_steps+1]
    """
    
    if isinstance(cov, int) or isinstance(cov, float):
        cov = cov * np.identity(num_stocks)
    if isinstance(mu, int) or isinstance(mu, float):
        mu = mu * np.ones(num_stocks)
    if isinstance(spot_init, int) or isinstance(spot_init, float):
        spot_init = spot_init * np.ones(num_stocks)

    multivar_norm = np_random.multivariate_normal(np.zeros(num_stocks), cov, size=(num_paths, num_steps), method='cholesky')

    S = np.zeros((num_paths, num_stocks, num_steps + 1))
    S[:, :, 0] = spot_init

    dt = 1.0 / ppy

    # Ito's lemma
    for i_step in range(1, num_steps + 1):
        S[:, :, i_step] = S[:, :, i_step - 1] * np.exp(
            (np.sqrt(dt) * multivar_norm[:, i_step - 1, :]) + dt * (mu - 0.5 * np.diag(cov))
        )

    return S

if __name__ == "__main__":

    np_random = np.random.default_rng(seed=None)
    S = geometric_brownian_motion(4, 10, 3, 1.0, 1.0, 1.0, 252, np_random)
    print(S)
