import numpy as np
from rough_pricer.rbergomi import cov_vv, cov_vw, simulate_variance, simulate_stock, atm_skew
from rough_pricer.black_scholes import implied_vol
import matplotlib.pyplot as plt
from pathlib import Path
Path("figures").mkdir(exist_ok=True)


t, v, Wt, W = simulate_variance(1, 100, 0.1, 1.9, 0.04, 20_000)
print(cov_vv(0.3, 0.7, 0.5), cov_vw(0.3, 0.7, 0.5))   # check 1: both should be 0.3
print(np.var(Wt[49]), 0.5**0.2)                       # check 2: simulated vs. theory at t = 0.5
print(v.mean(axis=1)[[9, 49, 99]])   
S = simulate_stock(t, v, W, xi=0.04, rho=-0.9)
print(S[-1].mean())                                   # martingale check: should be close to 100

for K in range(80, 125, 5):
    price = np.maximum(S[-1] - K, 0).mean()           # no discounting needed, since r = 0
    print(K, round(implied_vol(price, 100, K, 1.0, 0.0), 4))
Ts = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])

skews = np.array([atm_skew(T) for T in Ts])
slope, intercept = np.polyfit(np.log(Ts), np.log(skews), 1)       # straight-line fit on log-log scale
print(skews)
print("fitted slope:", round(slope, 3), "| theory (H - 0.5):", -0.4)

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(Ts, skews, "o", label="simulated ATM skew (exact rBergomi)")
ax.loglog(Ts, np.exp(intercept) * Ts**slope, "-", label=f"fitted slope {slope:.3f}")
ax.loglog(Ts, skews[0] * (Ts / Ts[0])**(-0.4), "--", label="theory: slope H - ½ = -0.40")
ax.set_xlabel("maturity T (years)"); ax.set_ylabel("ATM skew |∂σ/∂log K|")
ax.set_title("Rough volatility: ATM skew power law"); ax.legend()
fig.tight_layout(); fig.savefig("figures/skew_powerlaw.png", dpi=150)