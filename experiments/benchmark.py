import time
import numpy as np
from rough_pricer.rbergomi import build_cholesky, simulate_stock
from rough_pricer.markov_lift import simulate_lifted
import matplotlib.pyplot as plt
from pathlib import Path
Path("figures").mkdir(exist_ok=True)

H, eta, xi, rho, T, paths, N = 0.1, 1.9, 0.04, -0.9, 1.0, 10_000, 40
rows = []
for n in [100, 250, 500, 1000, 2000]:
    t0 = time.perf_counter()
    tt, L = build_cholesky(T, n, H)                                  # one-time setup
    t1 = time.perf_counter()
    X = L @ np.random.default_rng(0).standard_normal((2*n, paths))   # whole path at once
    v = xi * np.exp(eta * X[:n] - 0.5 * eta**2 * tt[:, None]**(2*H))
    simulate_stock(tt, v, X[n:], xi, rho)
    t2 = time.perf_counter()
    simulate_lifted(T, n, N, H, eta, xi, rho, paths, fast=1e5)       # step by step, N factors
    t3 = time.perf_counter()
    print(f"n={n:5d} | exact: setup {t1-t0:6.2f}s, sim {t2-t1:6.2f}s, path memory {X.nbytes/1e6:6.0f} MB"
          f" | lifted: {t3-t2:6.2f}s, state memory {(N+1)*paths*8/1e6:4.1f} MB")
    rows.append((n, t1 - t0, t2 - t1, t3 - t2))

n_steps, setup, sim, lifted = np.array(rows).T
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(n_steps, setup + sim, "o-", label="exact: setup + simulation")
ax.loglog(n_steps, sim, "o--", label="exact: simulation only (setup reused)")
ax.loglog(n_steps, lifted, "s-", label="Markovian lift (N = 40)")
ax.set_xlabel("time steps n"); ax.set_ylabel("run time (s), 10,000 paths")
ax.set_title("Cost scaling: exact vs. Markovian lift"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("figures/benchmark.png", dpi=150)