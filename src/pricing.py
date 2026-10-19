"""Option pricing models: Black-Scholes, Heston (semi-analytical) and Heston (Monte Carlo).

The formulas are the same ones used in the thesis. Changes from the original script:
- Black-Scholes also accepts NumPy arrays, so the whole dataset is priced at once.
- The Heston Monte Carlo uses a seeded random generator, so results are reproducible.
- Errors in the numerical integration are caught explicitly instead of with a bare `except`.
"""

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.stats import norm
import warnings


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------
def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes price for European calls/puts.

    Works with scalars or NumPy arrays (option_type can be an array of 'call'/'put').
    When T <= 0 or sigma <= 0 the intrinsic value is returned.
    """
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    is_call = np.char.lower(np.asarray(option_type, dtype=str)) == "call"

    valid = (T > 0) & (sigma > 0)
    T_safe = np.where(valid, T, 1.0)
    sigma_safe = np.where(valid, sigma, 1.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * np.sqrt(T_safe))
    d2 = d1 - sigma_safe * np.sqrt(T_safe)
    disc_K = K * np.exp(-r * T_safe)

    call = S * norm.cdf(d1) - disc_K * norm.cdf(d2)
    put = disc_K * norm.cdf(-d2) - S * norm.cdf(-d1)
    price = np.where(is_call, call, put)

    intrinsic = np.where(is_call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    price = np.where(valid, price, intrinsic)
    return price.item() if price.ndim == 0 else price


# ---------------------------------------------------------------------------
# Heston (semi-analytical, characteristic function + numerical integration)
# ---------------------------------------------------------------------------
def heston_price(S0, K, T, r, kappa, theta, sigma_v, rho, v0, option_type="call"):
    """Heston (1993) price via the characteristic function of ln(S_T).

    Uses the 'little Heston trap' formulation (Albrecher et al., 2007), which is
    numerically stable for long maturities. Puts are obtained by put-call parity.
    Returns NaN if the integration fails.
    """

    def char_func(phi):
        u = -0.5
        b = kappa - rho * sigma_v * phi * 1j
        a = kappa * theta
        d = np.sqrt(b**2 - sigma_v**2 * (2 * u * phi * 1j - phi**2))
        g = (b - d) / (b + d)
        C = (a / sigma_v**2) * ((b - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
        D = ((b - d) / sigma_v**2) * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
        return np.exp(C + D * v0 + 1j * phi * np.log(S0 * np.exp(r * T)))

    def integrand1(phi):
        return np.real(np.exp(-1j * phi * np.log(K)) * char_func(phi - 1j) / (1j * phi * char_func(-1j)))

    def integrand2(phi):
        return np.real(np.exp(-1j * phi * np.log(K)) * char_func(phi) / (1j * phi))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            P1, _ = quad(integrand1, 0, 100, limit=200)
            P2, _ = quad(integrand2, 0, 100, limit=200)
    except (ValueError, ZeroDivisionError, FloatingPointError, OverflowError):
        return np.nan

    call_price = S0 * (0.5 + P1 / np.pi) - K * np.exp(-r * T) * (0.5 + P2 / np.pi)
    if option_type.lower() == "call":
        return call_price
    return call_price - S0 + K * np.exp(-r * T)


# ---------------------------------------------------------------------------
# Heston (Monte Carlo, Euler scheme with full truncation)
# ---------------------------------------------------------------------------
def heston_price_mc(S0, K, T, r, kappa, theta, sigma_v, rho, v0, option_type="call",
                    n_simulations=1000, n_steps=100, seed=42, return_std_error=False):
    """Heston price by Monte Carlo simulation.

    Correlated shocks are built with a Cholesky decomposition, as in the thesis.
    A fixed `seed` makes the result reproducible. When pricing many options, give each
    one a different seed (the pipeline uses seed + row number): if all options share
    the same paths, the sampling error is identical across them and does not average
    out in aggregate statistics. Two standard variance-reduction techniques are used:
    - antithetic variates: half of the paths use the shocks with the sign flipped;
    - martingale correction: terminal prices are rescaled so that their sample mean
      equals the forward S0 * exp(rT), as it must under the risk-neutral measure.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    half = rng.standard_normal((n_steps, (n_simulations + 1) // 2, 2))
    rand = np.concatenate([half, -half], axis=1)[:, :n_simulations]
    chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    rand_corr = rand @ chol.T
    z_s, z_v = rand_corr[:, :, 0], rand_corr[:, :, 1]

    log_s = np.full(n_simulations, np.log(S0))
    v = np.full(n_simulations, v0)
    for t in range(n_steps):
        v_pos = np.maximum(v, 0.0)
        log_s = log_s + (r - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * z_s[t]
        v = np.maximum(0.0, v + kappa * (theta - v_pos) * dt + sigma_v * np.sqrt(v_pos * dt) * z_v[t])

    s_T = np.exp(log_s)
    s_T *= S0 * np.exp(r * T) / s_T.mean()   # martingale correction
    if option_type.lower() == "call":
        payoffs = np.maximum(s_T - K, 0.0)
    else:
        payoffs = np.maximum(K - s_T, 0.0)

    disc = np.exp(-r * T)
    price = disc * payoffs.mean()
    if return_std_error:
        # Standard error from antithetic pairs (each pair average is one independent draw).
        n_pairs = n_simulations // 2
        pairs = 0.5 * (payoffs[:n_pairs] + payoffs[(n_simulations + 1) // 2:][:n_pairs])
        return price, disc * pairs.std(ddof=1) / np.sqrt(n_pairs)
    return price
