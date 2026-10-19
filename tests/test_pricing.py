"""Sanity checks for the pricing models."""

import numpy as np
import pytest

from src.pricing import black_scholes, heston_price, heston_price_mc

HESTON = dict(kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, v0=0.04)


def test_black_scholes_reference_value():
    # Classic textbook value (Hull): S=K=100, T=1, r=5%, sigma=20% -> call = 10.4506
    assert black_scholes(100, 100, 1.0, 0.05, 0.20, "call") == pytest.approx(10.4506, abs=1e-4)


def test_black_scholes_vectorized_matches_scalar():
    S, K = np.array([100.0, 100.0]), np.array([95.0, 105.0])
    types = np.array(["call", "put"])
    vec = black_scholes(S, K, 0.5, 0.03, 0.25, types)
    assert vec[0] == pytest.approx(black_scholes(100, 95, 0.5, 0.03, 0.25, "call"))
    assert vec[1] == pytest.approx(black_scholes(100, 105, 0.5, 0.03, 0.25, "put"))


@pytest.mark.parametrize("K", [90, 100, 110])
def test_put_call_parity_black_scholes(K):
    S, T, r = 100, 0.5, 0.04
    c = black_scholes(S, K, T, r, 0.2, "call")
    p = black_scholes(S, K, T, r, 0.2, "put")
    assert c - p == pytest.approx(S - K * np.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("K", [90, 100, 110])
def test_put_call_parity_heston(K):
    S, T, r = 100, 0.5, 0.04
    c = heston_price(S, K, T, r, **HESTON, option_type="call")
    p = heston_price(S, K, T, r, **HESTON, option_type="put")
    assert c - p == pytest.approx(S - K * np.exp(-r * T), abs=1e-8)


@pytest.mark.parametrize("K", [90, 100, 110])
def test_heston_converges_to_black_scholes_without_vol_of_vol(K):
    # With almost no vol-of-vol and v0 = theta = sigma^2, Heston collapses to Black-Scholes.
    sigma = 0.2
    h = heston_price(100, K, 1.0, 0.03, kappa=1.0, theta=sigma**2, sigma_v=1e-3, rho=0.0, v0=sigma**2)
    assert h == pytest.approx(black_scholes(100, K, 1.0, 0.03, sigma, "call"), abs=1e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_monte_carlo_matches_analytical(option_type):
    S, K, T, r = 100, 100, 0.5, 0.03
    analytical = heston_price(S, K, T, r, **HESTON, option_type=option_type)
    mc, se = heston_price_mc(S, K, T, r, **HESTON, option_type=option_type,
                             n_simulations=100_000, n_steps=200, seed=1, return_std_error=True)
    # Allow 3 standard errors plus a small discretisation bias from the Euler scheme.
    assert abs(mc - analytical) < 3 * se + 0.05


def test_monte_carlo_put_call_parity_holds_exactly():
    # The martingale correction makes simulated prices satisfy put-call parity exactly.
    S, K, T, r = 100, 105, 0.25, 0.03
    c = heston_price_mc(S, K, T, r, **HESTON, option_type="call", seed=3)
    p = heston_price_mc(S, K, T, r, **HESTON, option_type="put", seed=3)
    assert c - p == pytest.approx(S - K * np.exp(-r * T), abs=1e-8)


def test_monte_carlo_errors_average_out_with_one_seed_per_option():
    # Regression test: with ONE shared seed, every option inherits the same sampling error
    # and the whole dataset is biased. With one seed per option, errors are independent.
    S, T, r = 100, 0.25, 0.03
    diffs = [heston_price_mc(S, K, T, r, **HESTON, option_type=opt, n_simulations=1000, seed=i)
             - heston_price(S, K, T, r, **HESTON, option_type=opt)
             for i, (K, opt) in enumerate((K, o) for K in range(90, 111) for o in ("call", "put"))]
    assert abs(np.mean(diffs)) < 0.05


def test_monte_carlo_is_reproducible():
    a = heston_price_mc(100, 100, 0.5, 0.03, **HESTON, seed=7)
    b = heston_price_mc(100, 100, 0.5, 0.03, **HESTON, seed=7)
    assert a == b
