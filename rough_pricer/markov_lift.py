import numpy as np
from scipy.special import gamma

def lift_weights(N, H, T, dt, fast=10):
    """Approximate the kernel t^(H-1/2) by sum_i c_i * exp(-g_i * t)."""
    a = 0.5 - H
    g_min, g_max = 0.1 / T, fast / dt                   # slowest and fastest speeds that matter
    inner = np.geomspace(g_min, g_max, N) if N > 1 else np.array([g_max])
    edges = np.concatenate([[0.0], inner])            # N bins: [0, g_min], then geometric up to g_max
    lo, hi = edges[:-1], edges[1:]
    c = (hi**a - lo**a) / gamma(a + 1)                # weight = total kernel mass in the bin
    g = a / (a + 1) * (hi**(a + 1) - lo**(a + 1)) / (hi**a - lo**a)   # speed = the bin's average speed
    return c, g

def step_factor(g, dt):
    """Matrix square root of the covariance of one step's (dI_1..dI_N, dW)."""
    G = g[:, None] + g[None, :]
    C = np.empty((len(g) + 1, len(g) + 1))
    C[:-1, :-1] = (1 - np.exp(-G * dt)) / G               # Cov(dI_i, dI_j)
    C[:-1, -1] = C[-1, :-1] = (1 - np.exp(-g * dt)) / g   # Cov(dI_i, dW)
    C[-1, -1] = dt                                        # Var(dW)
    w, V = np.linalg.eigh(C)
    return V * np.sqrt(np.clip(w, 0, None))               # robust square root: L @ L.T = C

def lifted_var(c, g, H, t):
    """Exact variance of the lifted rough process at time t (needed for the -1/2 correction)."""
    G = g[:, None] + g[None, :]
    return 2 * H * c @ ((1 - np.exp(-G * t)) / G) @ c

def simulate_lifted(T, n, N, H, eta, xi, rho, n_paths, S0=100.0, seed=0, fast=10):
    dt = T / n
    c, g = lift_weights(N, H, T, dt, fast)
    L, decay = step_factor(g, dt), np.exp(-g * dt)[:, None]
    rng = np.random.default_rng(seed)
    Y, v = np.zeros((N, n_paths)), np.full(n_paths, xi)
    logS = np.full(n_paths, np.log(S0))
    for k in range(1, n + 1):
        X = L @ rng.standard_normal((N + 1, n_paths))     # this step's (dI_1..dI_N, dW)
        dZ = rho * X[-1] + np.sqrt(1 - rho**2) * rng.standard_normal(n_paths) * np.sqrt(dt)
        logS += np.sqrt(v) * dZ - 0.5 * v * dt            # stock moves with START-of-step variance
        Y = decay * Y + X[:-1]                            # each factor: decay, then add new noise
        Wt = np.sqrt(2 * H) * (c @ Y)                     # lifted rough process
        v = xi * np.exp(eta * Wt - 0.5 * eta**2 * lifted_var(c, g, H, k * dt))
    return np.exp(logS)                                   # terminal stock prices only
