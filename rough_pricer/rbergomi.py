import numpy as np
from scipy.special import hyp2f1
from .black_scholes import implied_vol

def cov_vv(s, t, H):
    """Covariance of the rough process W~ at times s and t."""
    s, t = min(s, t), max(s, t)
    if s == t:
        return t**(2*H)                                   # variance: t^(2H)
    return (2*H / (H + 0.5)) * s**(H + 0.5) * t**(H - 0.5) * hyp2f1(0.5 - H, 1, H + 1.5, s / t)

def cov_vw(s, t, H):
    """Covariance of W~ at time s with the Brownian motion W at time t."""
    m = min(s, t)
    return np.sqrt(2*H) / (H + 0.5) * (s**(H + 0.5) - (s - m)**(H + 0.5))

def build_cholesky(T, n, H):
    t = np.linspace(T / n, T, n)                  # grid t_1..t_n (skip t=0, where all values are 0)
    C = np.zeros((2*n, 2*n))
    for i in range(n):
        for j in range(n):
            C[i, j] = cov_vv(t[i], t[j], H)                     # rough process with itself
            C[n + i, n + j] = min(t[i], t[j])                   # Brownian motion with itself
            C[i, n + j] = C[n + j, i] = cov_vw(t[i], t[j], H)   # cross terms (symmetric)
    return t, np.linalg.cholesky(C)

def simulate_variance(T, n, H, eta, xi, n_paths, seed=0):
    t, L = build_cholesky(T, n, H)
    rng = np.random.default_rng(seed)
    X = L @ rng.standard_normal((2*n, n_paths))    # correlated Gaussians, shape (2n, n_paths)
    Wt, W = X[:n], X[n:]                           # split into rough process and Brownian motion
    v = xi * np.exp(eta * Wt - 0.5 * eta**2 * t[:, None]**(2*H))   # variance paths
    return t, v, Wt, W

def simulate_stock(t, v, W, xi, rho, S0=100.0, seed=1):
    n, n_paths = v.shape
    dt = t[0]                                                 # uniform grid, so dt = t_1
    v_left = np.vstack([np.full((1, n_paths), xi), v[:-1]])   # variance at the START of each step
    dW = np.diff(W, axis=0, prepend=0)                        # increments of the vol-driving BM
    dW_perp = np.random.default_rng(seed).standard_normal((n, n_paths)) * np.sqrt(dt)
    dZ = rho * dW + np.sqrt(1 - rho**2) * dW_perp             # stock's BM, correlated with W at level rho
    log_S = np.log(S0) + np.cumsum(np.sqrt(v_left) * dZ - 0.5 * v_left * dt, axis=0)
    return np.exp(log_S)                                      # stock paths, shape (n, n_paths)

def atm_skew(T, H=0.1, eta=1.9, xi=0.04, rho=-0.9, n=50, n_paths=50_000, S0=100.0):
    t, v, Wt, W = simulate_variance(T, n, H, eta, xi, n_paths)   # fresh grid of n steps on [0, T]
    ST = simulate_stock(t, v, W, xi, rho, S0)[-1]
    d = 0.05 * np.sqrt(T)                                      # strike spacing scales with maturity
    K_lo, K_hi = S0 * np.exp(-d), S0 * np.exp(d)
    iv_lo = implied_vol(np.maximum(ST - K_lo, 0).mean(), S0, K_lo, T, 0.0)
    iv_hi = implied_vol(np.maximum(ST - K_hi, 0).mean(), S0, K_hi, T, 0.0)
    return (iv_lo - iv_hi) / (2 * d)                           # slope of implied vol vs. log-strike

