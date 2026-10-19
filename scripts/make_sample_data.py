"""Generate a small synthetic dataset in the same format as the Kaggle SPY option chains.

Prices come from a Heston model with KNOWN parameters, so the pipeline can be tested
end to end without downloading anything, and the calibration can be checked against
the true parameters.

    python -m scripts.make_sample_data
    python run_pipeline.py --options-csv data/sample/sample_options.csv \
        --market-csv data/sample/sample_market.csv --out-dir results/sample
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.pricing import heston_price

TRUE_PARAMS = dict(kappa=3.0, theta=0.05, sigma_v=0.5, rho=-0.7)
OUT = Path("data/sample")


def main(seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", "2021-04-30")
    all_days = pd.bdate_range("2020-12-01", "2021-08-31")

    # Underlying path, variance level and T-Bill for every business day (market data file).
    spy = 370 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(all_days))))
    v0 = np.clip(0.045 + 0.01 * np.sin(np.arange(len(all_days)) / 15), 0.02, None)
    market = pd.DataFrame({"VIX": 100 * np.sqrt(v0), "SPY_Close": spy, "DGS3MO": 0.08},
                          index=pd.Index(all_days, name="Date"))

    rows = []
    for date in dates:
        S = market.loc[date, "SPY_Close"]
        prev = market.index[market.index.get_loc(date) - 1]
        v = (market.loc[prev, "VIX"] / 100) ** 2   # the pipeline uses D-1 VIX as v0
        r = np.log1p(market.loc[prev, "DGS3MO"] / 100)
        for dte in (14, 30, 60, 120):
            T = dte / 365
            for K in np.arange(round(S * 0.9 / 5) * 5, S * 1.1, 5):
                c = heston_price(S, K, T, r, **TRUE_PARAMS, v0=v, option_type="call")
                p = heston_price(S, K, T, r, **TRUE_PARAMS, v0=v, option_type="put")
                # Small pricing noise and a realistic spread.
                c_mid, p_mid = c * (1 + rng.normal(0, 0.01)), p * (1 + rng.normal(0, 0.01))
                c_sp, p_sp = max(0.02, 0.03 * c_mid), max(0.02, 0.03 * p_mid)
                rows.append({
                    " [QUOTE_DATE]": date.strftime(" %Y-%m-%d"),
                    " [UNDERLYING_LAST]": round(S, 2),
                    " [EXPIRE_DATE]": (date + pd.Timedelta(days=dte)).strftime(" %Y-%m-%d"),
                    " [DTE]": dte,
                    " [C_BID]": round(c_mid - c_sp / 2, 2), " [C_ASK]": round(c_mid + c_sp / 2, 2),
                    " [C_VOLUME]": int(rng.integers(0, 3000)),
                    " [STRIKE]": K,
                    " [P_BID]": round(p_mid - p_sp / 2, 2), " [P_ASK]": round(p_mid + p_sp / 2, 2),
                    " [P_VOLUME]": int(rng.integers(0, 3000)),
                })

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "sample_options.csv", index=False)
    market.to_csv(OUT / "sample_market.csv")
    print(f"Wrote {len(rows):,} rows to {OUT / 'sample_options.csv'}. True parameters: {TRUE_PARAMS}")


if __name__ == "__main__":
    main()
