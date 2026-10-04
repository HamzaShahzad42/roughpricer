import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def mc_call(S, K, T, r, sigma, n_paths=200_000, seed=0):
    rng = np.random.default_rng(seed)                 # seeded, so results are reproducible
    Z = rng.standard_normal(n_paths)                  # one normal draw per path
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)  # terminal stock price
    payoff = np.maximum(ST - K, 0.0)
    price = np.exp(-r * T) * payoff.mean()            # discounted average payoff
    stderr = np.exp(-r * T) * payoff.std() / np.sqrt(n_paths)  # Monte Carlo error bar
    return price, stderr

def implied_vol(price, S, K, T, r) -> float:
    f = lambda sig: bs_call(S, K, T, r, sig) - price   # zero at the right sigma
    return brentq(f, 1e-6, 5.0) # type: ignore

if __name__ == "__main__":
    print(bs_call(100, 100, 1, 0, 0.2))
    print(mc_call(100,100,1,0,0.2))
    print(implied_vol(bs_call(100, 100, 1, 0, 0.2), 100, 100, 1, 0))