"""End-to-end pipeline: data -> Black-Scholes -> Heston calibration -> Heston prices -> errors -> backtest.

Usage
-----
    python run_pipeline.py --options-csv data/spy_2020_2022.csv

Data step and analysis in SQL (SQLite) instead of pandas:
    python run_pipeline.py --options-csv data/spy_2020_2022.csv --engine sql

Quick test on a fraction of the data:
    python run_pipeline.py --options-csv data/spy_2020_2022.csv --sample-frac 0.02 --mc-sims 500

All outputs are written to results/.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import db
from src.analysis import (error_by_moneyness, error_metrics, plot_error_by_moneyness, plot_heston_params)
from src.backtest import run_backtest
from src.calibration import PARAM_NAMES, attach_params, calibrate_by_period
from src.data import FilterConfig, load_market_data, prepare_dataset
from src.pricing import black_scholes, heston_price, heston_price_mc

tqdm.pandas()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--options-csv", required=True, help="Raw SPY option-chain CSV")
    p.add_argument("--market-csv", default="data/market_data.csv",
                   help="Cache for VIX, T-Bill and SPY closes (downloaded if missing)")
    p.add_argument("--filtered-cache", default="data/options_filtered.pkl",
                   help="Cache of the filtered options (delete it if you change the filters)")
    p.add_argument("--engine", choices=["pandas", "sql"], default="pandas",
                   help="Run the data step and the analysis in pandas or in SQL (queries in sql/)")
    p.add_argument("--db", default="data/options.db",
                   help="SQLite database (built from --options-csv on the first run with --engine sql)")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--calib-freq", default="M", help="Recalibration frequency (pandas period: M, Q, Y)")
    p.add_argument("--calib-options", type=int, default=150, help="Options sampled per calibration")
    p.add_argument("--mc-sims", type=int, default=1000)
    p.add_argument("--mc-steps", type=int, default=100)
    p.add_argument("--skip-mc", action="store_true", help="Skip the Monte Carlo pricing (slowest step)")
    p.add_argument("--sample-frac", type=float, default=1.0, help="Use a random fraction of the options")
    p.add_argument("--threshold", type=float, default=0.20, help="Backtest mispricing threshold")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--from-priced", action="store_true",
                   help="Skip steps 2-6 and reuse priced_options.csv and heston_params.csv from --out-dir")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("--- 1. Market data (VIX, 3M T-Bill, SPY) ---")
    market = load_market_data("2019-12-01", "2023-07-31", cache_path=args.market_csv)

    if args.from_priced:
        print("--- Reusing priced options and calibrated parameters from a previous run ---")
        df = pd.read_csv(out / "priced_options.csv", parse_dates=["QUOTE_DATE", "MKT_DATE", "EXPIRE_DATE"])
        params = pd.read_csv(out / "heston_params.csv", parse_dates=["Calibration_Date"])
        evaluate(df, params, market, args, out)
        return

    print("--- 2. Option data: one row per option, filters, D-1 market data ---")
    conn = None
    if args.engine == "sql":
        if Path(args.db).exists():
            conn = db.connect(args.db)
        else:
            print(f"Building the SQLite database {args.db} from {args.options_csv} (first run only)...")
            conn = db.build_database(args.db, args.options_csv, market)
        df = db.prepare_dataset_sql(conn, FilterConfig())
    else:
        df = prepare_dataset(args.options_csv, market, FilterConfig(), cache_path=args.filtered_cache)
    if args.sample_frac < 1:
        df = df.sample(frac=args.sample_frac, random_state=args.seed).sort_values("QUOTE_DATE").reset_index(drop=True)
    print(f"Options in the final dataset: {len(df):,} "
          f"(calls: {(df['Type'] == 'call').sum():,} | puts: {(df['Type'] == 'put').sum():,})")

    print("--- 3. Black-Scholes (sigma = VIX) ---")
    df["BS_Price"] = black_scholes(df["Underlying_Price"], df["Strike"], df["T"],
                                   df["Risk_Free_Rate"], df["VIX_Price"] / 100, df["Type"])

    print(f"--- 4. Heston calibration (frequency: {args.calib_freq}) ---")
    params = calibrate_by_period(df, freq=args.calib_freq, max_options=args.calib_options, seed=args.seed)
    params.to_csv(out / "heston_params.csv", index=False)
    df = attach_params(df, params, freq=args.calib_freq)

    def heston_inputs(row):
        return dict(S0=row["Underlying_Price"], K=row["Strike"], T=row["T"], r=row["Risk_Free_Rate"],
                    v0=(row["VIX_Price"] / 100) ** 2, option_type=row["Type"],
                    **{p: row[p] for p in PARAM_NAMES})

    print("--- 5. Heston (analytical) ---")
    df["Heston_Price"] = df.progress_apply(lambda row: heston_price(**heston_inputs(row)), axis=1)

    if not args.skip_mc:
        print(f"--- 6. Heston (Monte Carlo, {args.mc_sims} paths) ---")
        # Each option gets its own reproducible seed, so sampling errors are independent
        # across options and average out in the aggregate statistics.
        df["Heston_MC_Price"] = df.progress_apply(
            lambda row: heston_price_mc(**heston_inputs(row), n_simulations=args.mc_sims,
                                        n_steps=args.mc_steps, seed=args.seed + int(row.name)), axis=1)

    df.to_csv(out / "priced_options.csv", index=False)
    evaluate(df, params, market, args, out)
    if conn is not None:
        evaluate_sql(conn, df, args, out)


def evaluate(df, params, market, args, out):
    """Steps 7-8: pricing errors, charts and backtest."""
    print("--- 7. Pricing errors ---")
    metrics = error_metrics(df)
    metrics.to_csv(out / "error_metrics.csv", index=False)
    print(metrics.round(4).to_string(index=False))

    binned = error_by_moneyness(df)
    binned.to_csv(out / "error_by_moneyness.csv", index=False)
    plot_error_by_moneyness(binned, out / "error_by_moneyness.png")
    plot_heston_params(params, out / "heston_params.png")

    print(f"--- 8. Backtest (buy at ask when ask < model x {1 - args.threshold:.2f}, hold to expiry) ---")
    model_cols = [c for c in ["BS_Price", "Heston_Price", "Heston_MC_Price"] if c in df.columns]
    summary, trades = run_backtest(df, model_cols, market["SPY_Close"], threshold=args.threshold)
    summary.to_csv(out / "backtest_summary.csv")
    with pd.option_context("display.float_format", "{:,.2f}".format):
        print(summary.to_string())

    print(f"\nDone. Results saved to {out.resolve()}")


def evaluate_sql(conn, df, args, out):
    """The same analysis written in SQL (sql/04-06), saved next to the pandas results."""
    print("--- 9. Analysis in SQL ---")
    db.write_priced_options(conn, df)
    models = [c for c in ["BS_Price", "Heston_Price", "Heston_MC_Price"] if c in df.columns]

    errors = pd.concat([db.run_sql_file(conn, "04_error_by_bucket.sql", model=m).assign(model=m) for m in models])
    errors.to_csv(out / "sql_error_by_bucket.csv", index=False)
    print(errors.round(4).to_string(index=False))

    summaries = []
    for m in models:
        db.run_sql_file(conn, "05_trades.sql", params={"threshold": args.threshold}, model=m)
        summaries.append(db.run_sql_file(conn, "06_backtest_summary.sql").assign(model=m))
    summary = pd.concat(summaries).set_index("model").T
    summary.to_csv(out / "sql_backtest_summary.csv")
    with pd.option_context("display.float_format", "{:,.2f}".format):
        print(summary.to_string())


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
