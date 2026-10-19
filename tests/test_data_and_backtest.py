"""Regression tests for the fixes made to the thesis pipeline."""

import numpy as np
import pandas as pd

from src.backtest import generate_trades, peak_capital, summarize
from src.data import FilterConfig, apply_filters, attach_market_data, to_long_format


def raw_row(**overrides):
    row = dict(QUOTE_DATE=pd.Timestamp("2021-03-01"), EXPIRE_DATE=pd.Timestamp("2021-04-16"),
               DTE=46, UNDERLYING_LAST=380.0, STRIKE=380.0,
               C_BID=10.0, C_ASK=10.4, C_VOLUME=500,
               P_BID=12.0, P_ASK=12.6, P_VOLUME=800)
    row.update(overrides)
    return row


def test_puts_use_their_own_prices():
    """The thesis bug: puts inherited the call mid whenever the call had volume."""
    df = to_long_format(pd.DataFrame([raw_row()]))
    put = df[df["Type"] == "put"].iloc[0]
    call = df[df["Type"] == "call"].iloc[0]
    assert put["Market_Price"] == (12.0 + 12.6) / 2
    assert put["Spread"] == 12.6 - 12.0
    assert call["Market_Price"] == (10.0 + 10.4) / 2


def test_liquidity_filter_is_applied_per_option():
    # Liquid call, illiquid put on the same row: only the call should survive.
    df = apply_filters(to_long_format(pd.DataFrame([raw_row(P_VOLUME=50)])), FilterConfig())
    assert list(df["Type"]) == ["call"]


def test_blank_volume_on_one_leg_does_not_drop_the_other_leg(tmp_path):
    """Second thesis bug: rows with any blank numeric field were dropped entirely."""
    from src.data import load_raw_options
    csv = tmp_path / "raw.csv"
    csv.write_text(
        "﻿[QUOTE_DATE], [UNDERLYING_LAST], [EXPIRE_DATE], [DTE], [C_VOLUME], [C_BID], [C_ASK],"
        " [STRIKE], [P_BID], [P_ASK], [P_VOLUME]\n"
        " 2021-03-01, 380.0, 2021-04-16, 46.0, , 10.0, 10.4, 380.0, 12.0, 12.6, 800\n",
        encoding="utf-8",
    )
    df = apply_filters(to_long_format(load_raw_options(csv)), FilterConfig())
    assert list(df["Type"]) == ["put"]


def test_market_data_comes_from_previous_trading_day():
    options = to_long_format(pd.DataFrame([raw_row(QUOTE_DATE=pd.Timestamp("2021-03-01"))]))  # Monday
    market = pd.DataFrame({"VIX": [20.0, 25.0, 30.0], "DGS3MO": [0.05, 0.05, 0.05]},
                          index=pd.to_datetime(["2021-02-25", "2021-02-26", "2021-03-01"]))
    out = attach_market_data(options, market)
    assert (out["MKT_DATE"] == pd.Timestamp("2021-02-26")).all()   # Friday, not Monday
    assert (out["VIX_Price"] == 25.0).all()


def _signals_df():
    # The same cheap contract signals on three consecutive days.
    days = pd.to_datetime(["2021-03-01", "2021-03-02", "2021-03-03"])
    return pd.DataFrame({
        "QUOTE_DATE": days, "EXPIRE_DATE": pd.Timestamp("2021-03-19"), "Type": "call",
        "Strike": 380.0, "Ask": [5.0, 5.1, 5.2], "Market_Price": [4.9, 5.0, 5.1],
        "Model": [8.0, 8.0, 8.0],
    })


def test_first_signal_only_and_entry_at_ask():
    spy = pd.Series([390.0], index=pd.to_datetime(["2021-03-19"]))
    trades = generate_trades(_signals_df(), "Model", spy, threshold=0.2)
    assert len(trades) == 1
    assert trades.loc[0, "QUOTE_DATE"] == pd.Timestamp("2021-03-01")
    assert trades.loc[0, "Entry_Price"] == 5.0
    assert np.isclose(trades.loc[0, "PnL"], (10.0 - 5.0) * 100)


def test_peak_capital_counts_overlapping_positions():
    trades = pd.DataFrame({
        "QUOTE_DATE": pd.to_datetime(["2021-01-04", "2021-01-05", "2021-02-01"]),
        "EXPIRE_DATE": pd.to_datetime(["2021-01-29", "2021-01-29", "2021-02-26"]),
        "Cost": [100.0, 200.0, 50.0], "PnL": [10.0, -20.0, 5.0],
    })
    assert peak_capital(trades) == 300.0          # first two overlap; third opens after they expire
    assert summarize(trades)["Max_Drawdown"] == -10.0


def test_stale_cache_is_rebuilt(tmp_path):
    """A cache built with older cleaning code must not be reused."""
    from src import data
    csv = tmp_path / "raw.csv"
    csv.write_text(
        "[QUOTE_DATE], [UNDERLYING_LAST], [EXPIRE_DATE], [DTE], [C_VOLUME], [C_BID], [C_ASK],"
        " [STRIKE], [P_BID], [P_ASK], [P_VOLUME]\n"
        " 2021-03-01, 380.0, 2021-04-16, 46.0, 500, 10.0, 10.4, 380.0, 12.0, 12.6, 800\n", encoding="utf-8")
    cache = tmp_path / "cache.pkl"
    stale = pd.DataFrame({"x": [1]})
    stale.attrs["cache_key"] = {"data_version": data.DATA_VERSION - 1}
    stale.to_pickle(cache)
    df = data.load_filtered_options(csv, cache_path=cache)
    assert len(df) == 2                                    # rebuilt from the CSV, not the stale cache
    assert data.load_filtered_options(csv, cache_path=cache).equals(df)   # and the new cache is reused
