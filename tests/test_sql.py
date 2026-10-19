"""Each SQL query must return the same result as the Python (pandas) implementation.

The checks run on a small synthetic dataset built here, with edge cases on purpose:
blank volumes, zero bids, options right on the filter limits and quote dates after holidays.

A query file that still contains the word TODO is an unfinished exercise and its test is skipped.
Run only these tests with:  python -m pytest tests/test_sql.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src import db
from src.backtest import generate_trades, summarize
from src.data import FilterConfig, prepare_dataset

RNG = np.random.default_rng(0)


def _require(name):
    if "TODO" in (db.SQL_DIR / name).read_text(encoding="utf-8"):
        pytest.skip(f"{name} is not finished yet (still contains TODO)")


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def _fmt(x):
    return "" if pd.isna(x) else f"{x:.6f}"


@pytest.fixture(scope="module")
def raw_csv(tmp_path_factory):
    days = pd.bdate_range("2021-01-04", "2021-03-31").drop(pd.Timestamp("2021-02-15"))  # Presidents' Day
    lines = ["﻿[QUOTE_DATE], [UNDERLYING_LAST], [EXPIRE_DATE], [DTE], [C_BID], [C_ASK], [C_VOLUME],"
             " [STRIKE], [P_BID], [P_ASK], [P_VOLUME]"]
    for d in days:
        S = 380 + RNG.normal(0, 5)
        for dte in (5, 7, 30, 90, 180, 200):
            for K in np.arange(330, 431, 5):
                c_mid, p_mid = max(S - K, 0) + RNG.uniform(0.2, 8), max(K - S, 0) + RNG.uniform(0.2, 8)
                c_sp, p_sp = RNG.uniform(0.01, 0.25) * c_mid, RNG.uniform(0.01, 0.25) * p_mid
                c_vol = RNG.choice([np.nan, 0, 50, 101, 500])           # blanks and limit values
                p_vol = RNG.choice([np.nan, 0, 50, 101, 500])
                c_bid = 0.0 if RNG.random() < 0.03 else c_mid - c_sp / 2  # some zero bids
                lines.append(", ".join([
                    f" {d:%Y-%m-%d}", _fmt(S), f" {d + pd.Timedelta(days=dte):%Y-%m-%d}", _fmt(dte),
                    _fmt(c_bid), _fmt(c_mid + c_sp / 2), _fmt(c_vol), _fmt(K),
                    _fmt(p_mid - p_sp / 2), _fmt(p_mid + p_sp / 2), _fmt(p_vol)]))
    path = tmp_path_factory.mktemp("raw") / "raw.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def market():
    dates = pd.date_range("2020-12-01", "2021-12-31", freq="D")
    m = pd.DataFrame({"VIX": RNG.uniform(15, 30, len(dates)),
                      "DGS3MO": RNG.uniform(0.02, 0.1, len(dates)),
                      "SPY_Close": 380 + np.cumsum(RNG.normal(0, 3, len(dates)))},
                     index=pd.Index(dates, name="Date"))
    m.loc[m.index.dayofweek >= 5, ["VIX", "SPY_Close"]] = np.nan   # no prices on weekends
    m.loc[m.index.dayofweek >= 5, "DGS3MO"] = np.nan
    m.loc["2021-02-15", ["VIX", "SPY_Close"]] = np.nan             # US holiday: no VIX/SPY...
    m.loc["2021-01-18", "DGS3MO"] = np.nan                         # ...and a FRED gap on another day
    return m


@pytest.fixture(scope="module")
def conn(raw_csv, market, tmp_path_factory):
    return db.build_database(tmp_path_factory.mktemp("db") / "test.db", raw_csv, market)


@pytest.fixture(scope="module")
def priced(raw_csv, market):
    df = prepare_dataset(raw_csv, market)
    noise = lambda: RNG.uniform(0.6, 1.4, len(df))
    return df.assign(BS_Price=df["Market_Price"] * noise(), Heston_Price=df["Market_Price"] * noise())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_02_03_sql_data_step_matches_pandas(conn, raw_csv, market):
    _require("02_options_long.sql")
    _require("03_attach_market_data.sql")
    expected = prepare_dataset(raw_csv, market)
    got = db.prepare_dataset_sql(conn)
    assert len(got) == len(expected), "different number of options after the filters"
    assert (got["Type"].value_counts() == expected["Type"].value_counts()).all()
    pd.testing.assert_frame_equal(got, expected, check_dtype=False, check_exact=False, rtol=1e-9)


def _bucket(df):
    m = df["Moneyness"]
    itm = ((df["Type"] == "call") & (m < 0.97)) | ((df["Type"] == "put") & (m > 1.03))
    return np.where(m.between(0.97, 1.03), "ATM", np.where(itm, "ITM", "OTM"))


@pytest.mark.parametrize("model", ["BS_Price", "Heston_Price"])
def test_04_error_by_bucket_matches_pandas(conn, priced, model):
    _require("04_error_by_bucket.sql")
    db.write_priced_options(conn, priced)
    got = db.run_sql_file(conn, "04_error_by_bucket.sql", model=model)

    p = priced.assign(bucket=_bucket(priced), err=priced["Market_Price"] - priced[model])
    exp = (p.groupby(["Type", "bucket"])["err"]
           .agg(n="size", rmse=lambda e: np.sqrt(np.mean(e**2)), mae=lambda e: np.mean(np.abs(e)), mean_error="mean")
           .reset_index().rename(columns={"Type": "type"}))
    pd.testing.assert_frame_equal(got, exp, check_dtype=False, check_exact=False, rtol=1e-9)


@pytest.mark.parametrize("model", ["BS_Price", "Heston_Price"])
def test_05_06_sql_backtest_matches_pandas(conn, priced, market, model):
    _require("05_trades.sql")
    _require("06_backtest_summary.sql")
    db.write_priced_options(conn, priced)
    db.run_sql_file(conn, "05_trades.sql", params={"threshold": 0.2}, model=model)
    got_trades = pd.read_sql_query("SELECT * FROM trades ORDER BY type, strike, expire_date", conn)

    exp_trades = generate_trades(priced, model, market["SPY_Close"], threshold=0.2)
    exp_trades = exp_trades.sort_values(["Type", "Strike", "EXPIRE_DATE"]).reset_index(drop=True)
    assert len(got_trades) == len(exp_trades), "different number of trades"
    np.testing.assert_allclose(got_trades["pnl"], exp_trades["PnL"], rtol=1e-9)

    got = db.run_sql_file(conn, "06_backtest_summary.sql").iloc[0]
    exp = summarize(exp_trades)
    assert got["trades"] == exp["Trades"]
    assert got["total_pnl"] == pytest.approx(exp["Total_PnL"])
    assert got["win_rate_pct"] == pytest.approx(exp["Win_Rate_%"])
    assert got["profit_factor"] == pytest.approx(exp["Profit_Factor"])
    assert got["roi_on_total_premium_pct"] == pytest.approx(exp["ROI_on_Total_Premium_% (thesis definition)"])
