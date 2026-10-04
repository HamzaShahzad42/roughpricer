# Lifted-vs-exact implied-vol bias, averaged over independent seeds, with per-strike standard errors.
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rough_pricer.black_scholes import implied_vol
from rough_pricer.rbergomi import simulate_variance, simulate_stock
from rough_pricer.markov_lift import lift_weights, lifted_var, simulate_lifted

H, ETA, XI, RHO, T, n, paths, seeds = 0.1, 1.9, 0.04, -0.9, 1.0, 100, 50_000, 8
strikes = np.arange(80, 125, 5)
configs = [(10, 20), (1e3, 40), (1e5, 40), (1e5, 80)]
smile = lambda ST: 100 * np.array([implied_vol(np.maximum(ST - K, 0).mean(), 100, K, T, 0.0) for K in strikes])

exact, lifted = [], {cfg: [] for cfg in configs}
for s in range(seeds):                                   # each seed = one independent run of every model
    t, v, _, W = simulate_variance(T, n, H, ETA, XI, paths, seed=2 * s)
    exact.append(smile(simulate_stock(t, v, W, XI, RHO, seed=2 * s + 1)[-1]))
    for fast, N in configs:
        lifted[(fast, N)].append(smile(simulate_lifted(T, n, N, H, ETA, XI, RHO, paths, seed=100 + s, fast=fast)))

exact = np.array(exact)
print("per-run noise of exact smile (SD, vol pts):", exact.std(0, ddof=1).round(2))
Path("figures").mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 4))
for (fast, N), runs in lifted.items():
    runs = np.array(runs)
    bias = runs.mean(0) - exact.mean(0)                                         # average over seeds
    se = np.sqrt(runs.var(0, ddof=1) / seeds + exact.var(0, ddof=1) / seeds)    # standard error of that average
    c, g = lift_weights(N, H, T, T / n, fast)
    print(f"fast={fast:g}/dt N={N}: var {lifted_var(c, g, H, T):.3f} | bias {bias.round(2)} | SE {se.round(2)}")
    ax.errorbar(strikes, bias, yerr=2 * se, marker="o", capsize=3, label=f"N = {N}, fastest speed = {fast:g}/Δt")
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("strike (spot = 100, T = 1)"); ax.set_ylabel("lifted - exact implied vol (vol pts)")
ax.set_title(f"Markovian lift bias ({seeds} seeds x {paths:,} paths, bars = 2 SE)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("figures/lift_accuracy.png", dpi=150)
