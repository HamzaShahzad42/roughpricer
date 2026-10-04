import numpy as np
from scipy.integrate import quad
from rough_pricer.black_scholes import bs_call, mc_call, implied_vol
from rough_pricer.rbergomi import cov_vv, cov_vw, simulate_variance, simulate_stock
from rough_pricer.markov_lift import lift_weights, lifted_var, simulate_lifted

H, ETA, XI, RHO = 0.1, 1.9, 0.04, -0.9

def test_mc_matches_black_scholes():
    price, se = mc_call(100, 100, 1, 0, 0.2)
    assert abs(price - bs_call(100, 100, 1, 0, 0.2)) < 3 * se       # within 3 standard errors

def test_implied_vol_round_trip():
    assert abs(implied_vol(bs_call(100, 110, 0.5, 0, 0.25), 100, 110, 0.5, 0) - 0.25) < 1e-8

def test_rough_cov_reduces_to_brownian_at_half():
    assert abs(cov_vv(0.3, 0.7, 0.5) - 0.3) < 1e-12 and abs(cov_vw(0.3, 0.7, 0.5) - 0.3) < 1e-12

def test_rough_cov_matches_numerical_integral():
    for s, t in [(0.3, 0.7), (0.99, 1.0)]:                             # includes neighbouring grid points
        integral, _ = quad(lambda u: (t - u)**(H - 0.5), 0, s, weight="alg", wvar=(0, H - 0.5))
        assert abs(cov_vv(s, t, H) - 2 * H * integral) < 1e-8

def test_exact_simulator_is_martingale():
    t, v, _, W = simulate_variance(1.0, 50, H, ETA, XI, 20_000)
    assert abs(simulate_stock(t, v, W, XI, RHO)[-1].mean() - 100) < 0.6

def test_lifted_simulator_is_martingale():
    assert abs(simulate_lifted(1.0, 50, 20, H, ETA, XI, RHO, 20_000, fast=1e5).mean() - 100) < 0.6

def test_lift_kernel_accuracy():
    c, g = lift_weights(20, H, 1.0, 0.01)
    t = np.linspace(0.01, 1.0, 100)
    approx = (c[:, None] * np.exp(-g[:, None] * t)).sum(axis=0)
    assert np.max(np.abs(approx / t**(H - 0.5) - 1)) < 0.01           # within 1% on [dt, T]

def test_lifted_variance_approaches_exact():
    c, g = lift_weights(40, H, 1.0, 0.01, fast=1e5)
    assert 0.95 < lifted_var(c, g, H, 1.0) < 1.0                        # exact model gives 1.0

def test_simulator_uses_fast_parameter():                               # would have caught the 'fast' bug
    a = simulate_lifted(1.0, 20, 10, H, ETA, XI, RHO, 1_000, fast=10)
    b = simulate_lifted(1.0, 20, 10, H, ETA, XI, RHO, 1_000, fast=1e5)
    assert not np.allclose(a, b)