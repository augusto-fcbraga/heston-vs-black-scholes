"""Heston calibration with SLSQP under the Feller condition.

Changes compared with the thesis script
---------------------------------------
- The thesis calibrated once, on the first day of the sample (early 2020), and used
  those parameters for all three years. `calibrate_by_period` recalibrates on the
  first trading day of every period (monthly by default), warm-starting from the
  previous solution, so the parameters can follow regime changes
  (COVID crash, 2021 recovery, 2022 rate hikes).
- Each calibration reports which parameters ended up on their bounds, because a
  solution pinned to the bounds usually means the optimizer was constrained rather
  than finding a genuine minimum.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .pricing import heston_price

PARAM_NAMES = ["kappa", "theta", "sigma_v", "rho"]
DEFAULT_INITIAL = [2.0, 0.05, 0.5, -0.7]
DEFAULT_BOUNDS = [(1e-3, 10.0), (1e-3, 1.0), (1e-3, 2.0), (-0.99, 0.99)]


def heston_calibration_error(params, market_data, S0, r, v0):
    """Sum of squared pricing errors (USD) between Heston and market mid prices."""
    kappa, theta, sigma_v, rho = params
    model = np.array([
        heston_price(S0, k, t, r, kappa, theta, sigma_v, rho, v0, typ)
        for k, t, typ in zip(market_data["Strike"], market_data["T"], market_data["Type"])
    ])
    err = model - market_data["Market_Price"].to_numpy()
    if np.any(~np.isfinite(err)):
        return 1e12
    return float(np.sum(err**2))


def calibrate_heston(day_data, initial=DEFAULT_INITIAL, bounds=DEFAULT_BOUNDS):
    """Calibrate kappa, theta, sigma_v and rho on the options of a single quote date.

    S0, r and v0 come from that day's data; v0 = (VIX / 100)^2, as in the thesis.
    """
    S0 = day_data["Underlying_Price"].iloc[0]
    r = day_data["Risk_Free_Rate"].iloc[0]
    v0 = (day_data["VIX_Price"].iloc[0] / 100) ** 2
    feller = {"type": "ineq", "fun": lambda x: 2 * x[0] * x[1] - x[2] ** 2}

    res = minimize(heston_calibration_error, initial, args=(day_data, S0, r, v0),
                   method="SLSQP", bounds=bounds, constraints=[feller])

    params = dict(zip(PARAM_NAMES, res.x))
    at_bounds = [name for name, val, (lo, hi) in zip(PARAM_NAMES, res.x, bounds)
                 if np.isclose(val, lo, rtol=1e-3, atol=1e-4) or np.isclose(val, hi, rtol=1e-3, atol=1e-4)]
    feller_gap = 2 * params["kappa"] * params["theta"] - params["sigma_v"] ** 2
    return {
        **params,
        "sse": res.fun,
        "rmse": np.sqrt(res.fun / len(day_data)),
        "n_options": len(day_data),
        "success": bool(res.success),
        "at_bounds": ",".join(at_bounds),
        "feller_binding": bool(feller_gap < 1e-3),
    }


def calibrate_by_period(df, freq="M", max_options=150, seed=42):
    """Recalibrate on the first quote date of each period.

    `max_options` limits how many options are used per calibration (random sample),
    which keeps runtime reasonable; set it to None to use all options of the day.
    Returns one row per period with the calibrated parameters and diagnostics.
    """
    df = df.copy()
    df["Period"] = df["QUOTE_DATE"].dt.to_period(freq)

    rows, initial = [], DEFAULT_INITIAL
    for period, group in df.groupby("Period"):
        first_day = group["QUOTE_DATE"].min()
        day = group[group["QUOTE_DATE"] == first_day]
        if max_options and len(day) > max_options:
            day = day.sample(max_options, random_state=seed)

        result = calibrate_heston(day, initial=initial)
        rows.append({"Period": str(period), "Calibration_Date": first_day, **result})
        initial = [result[p] for p in PARAM_NAMES]  # warm start for the next period
        print(f"  {period}: kappa={result['kappa']:.3f} theta={result['theta']:.4f} "
              f"sigma_v={result['sigma_v']:.3f} rho={result['rho']:.3f} "
              f"RMSE=${result['rmse']:.3f} bounds=[{result['at_bounds']}]")
    return pd.DataFrame(rows)


def attach_params(df, params_table, freq="M"):
    """Add the calibrated parameters of each option's period as columns."""
    df = df.copy()
    df["Period"] = df["QUOTE_DATE"].dt.to_period(freq).astype(str)
    cols = ["Period"] + PARAM_NAMES
    return df.merge(params_table[cols], on="Period", how="left")
