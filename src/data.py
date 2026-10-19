"""Data loading and cleaning.

Main fix compared with the thesis script
----------------------------------------
Each row of the raw dataset holds BOTH the call and the put for the same strike
and expiry. The original script computed one `Market_Price` per row
(the call mid whenever the call had any volume) *before* splitting calls and puts,
so most puts inherited the call's price and spread.

Here the raw data is first reshaped into one row per option (call OR put), and
price, spread and liquidity filters are computed with each option's own bid/ask.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_NUMERIC_COLS = ["UNDERLYING_LAST", "DTE", "STRIKE",
                    "C_BID", "C_ASK", "C_VOLUME", "P_BID", "P_ASK", "P_VOLUME"]


@dataclass
class FilterConfig:
    """Filters used in the thesis (Section 3)."""
    min_volume: int = 100          # contracts traded on the day (strictly greater than)
    min_premium: float = 0.50      # USD, mid price
    max_rel_spread: float = 0.15   # (ask - bid) / mid
    min_dte: int = 7
    max_dte: int = 180
    min_moneyness: float = 0.9     # K / S
    max_moneyness: float = 1.1


# ---------------------------------------------------------------------------
# Raw options
# ---------------------------------------------------------------------------
def _clean_name(col):
    return col.strip().lstrip("﻿").replace("[", "").replace("]", "")


RAW_COLS = RAW_NUMERIC_COLS + ["QUOTE_DATE", "EXPIRE_DATE"]


def _clean_raw_chunk(df, keep_blank_volumes=False):
    df.columns = [_clean_name(c) for c in df.columns]
    for col in RAW_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"].astype(str).str.strip())
    df["EXPIRE_DATE"] = pd.to_datetime(df["EXPIRE_DATE"].astype(str).str.strip())
    # A blank volume means no contracts traded that day. The thesis script dropped the
    # whole row when ANY numeric field was blank, so a liquid put was discarded whenever
    # the call at the same strike had no trades (and vice versa). Missing bid/ask on one
    # leg is handled later by the per-option filters.
    if not keep_blank_volumes:
        df[["C_VOLUME", "P_VOLUME"]] = df[["C_VOLUME", "P_VOLUME"]].fillna(0)
    return df.dropna(subset=["QUOTE_DATE", "EXPIRE_DATE", "UNDERLYING_LAST", "DTE", "STRIKE"])


def iter_raw_options(path, chunksize=500_000, keep_blank_volumes=False):
    """Read the raw SPY option-chain CSV (Kaggle format: columns like ' [C_BID]') in chunks.

    Only the 11 columns used by the study are loaded, so the full 2020-2022 file
    (about 1.3 GB, 33 columns) can be processed on a laptop.
    """
    reader = pd.read_csv(path, usecols=lambda c: _clean_name(c) in RAW_COLS, dtype=str,
                         encoding="utf-8-sig", chunksize=chunksize)
    for chunk in reader:
        yield _clean_raw_chunk(chunk, keep_blank_volumes)


def load_raw_options(path):
    """Read the whole raw CSV into memory (fine for small files; see iter_raw_options)."""
    return pd.concat(iter_raw_options(path), ignore_index=True)


def to_long_format(raw):
    """One row per option: calls and puts each keep their own bid, ask and volume."""
    base = ["QUOTE_DATE", "EXPIRE_DATE", "DTE", "UNDERLYING_LAST", "STRIKE"]
    legs = []
    for prefix, opt_type in (("C", "call"), ("P", "put")):
        leg = raw[base + [f"{prefix}_BID", f"{prefix}_ASK", f"{prefix}_VOLUME"]].copy()
        leg.columns = base + ["Bid", "Ask", "Volume"]
        leg["Type"] = opt_type
        legs.append(leg)

    df = pd.concat(legs, ignore_index=True)
    df = df.rename(columns={"UNDERLYING_LAST": "Underlying_Price", "STRIKE": "Strike"})
    df["Market_Price"] = (df["Bid"] + df["Ask"]) / 2
    df["Spread"] = df["Ask"] - df["Bid"]
    df["T"] = df["DTE"] / 365.0
    df["Moneyness"] = df["Strike"] / df["Underlying_Price"]
    return df


def apply_filters(df, cfg=FilterConfig()):
    """Liquidity, premium, spread, maturity and moneyness filters (per option)."""
    mask = (
        (df["Volume"] > cfg.min_volume)
        & (df["Market_Price"] >= cfg.min_premium)
        & (df["Bid"] > 0)
        & (df["Spread"] >= 0)
        & (df["Spread"] / df["Market_Price"] <= cfg.max_rel_spread)
        & df["DTE"].between(cfg.min_dte, cfg.max_dte)
        & df["Moneyness"].between(cfg.min_moneyness, cfg.max_moneyness)
    )
    return df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Market data: VIX, 3-month T-Bill and SPY close
# ---------------------------------------------------------------------------
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def download_market_data(start, end):
    """Download VIX and SPY closes (Yahoo Finance) and the 3M T-Bill (FRED)."""
    import yfinance as yf

    px = yf.download(["^VIX", "SPY"], start=start, end=end, progress=False, auto_adjust=False)["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    px = px.rename(columns={"^VIX": "VIX", "SPY": "SPY_Close"})

    tbill = pd.read_csv(FRED_URL.format(series="DGS3MO"), index_col=0, parse_dates=True, na_values=".")
    tbill.columns = ["DGS3MO"]

    market = px.join(tbill, how="outer").sort_index()
    market.index.name = "Date"
    return market.loc[start:end]


def load_market_data(start, end, cache_path=None):
    """Load market data from a local CSV cache, downloading it if the file does not exist."""
    if cache_path and Path(cache_path).exists():
        market = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
    else:
        market = download_market_data(start, end)
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            market.to_csv(cache_path)
    return market


def attach_market_data(df, market):
    """Attach the VIX and risk-free rate from the previous trading day (D-1).

    `merge_asof` with allow_exact_matches=False always takes the last available
    date strictly before the quote date, so holidays do not drop rows and
    there is no look-ahead.
    The T-Bill yield is converted to a continuously compounded rate: r = ln(1 + y).
    """
    mkt = market[["VIX", "DGS3MO"]].ffill().dropna().copy()
    mkt["VIX_Price"] = mkt["VIX"]
    mkt["Risk_Free_Rate"] = np.log1p(mkt["DGS3MO"] / 100)
    mkt = mkt[["VIX_Price", "Risk_Free_Rate"]].rename_axis("MKT_DATE").reset_index()

    # Same datetime resolution on both sides (pandas 2 and 3 can differ: ns vs us).
    df = df.assign(QUOTE_DATE=df["QUOTE_DATE"].astype("datetime64[ns]"))
    mkt["MKT_DATE"] = mkt["MKT_DATE"].astype("datetime64[ns]")
    out = pd.merge_asof(
        df.sort_values("QUOTE_DATE"), mkt.sort_values("MKT_DATE"),
        left_on="QUOTE_DATE", right_on="MKT_DATE",
        direction="backward", allow_exact_matches=False,
    )
    return out.dropna(subset=["VIX_Price", "Risk_Free_Rate"]).reset_index(drop=True)


# Bump this whenever the cleaning logic changes, so old caches are rebuilt automatically.
DATA_VERSION = 2


def _cache_key(cfg):
    return {"data_version": DATA_VERSION, **asdict(cfg)}


def load_filtered_options(options_csv, cfg=FilterConfig(), cache_path=None):
    """Raw CSV -> one row per option -> filters, with an optional pickle cache.

    Filters are applied chunk by chunk, so only the options that survive stay in memory.
    The cache stores the cleaning version and the filters it was built with; if either
    differs from the current code, the cache is ignored and rebuilt. (A stale cache once
    silently skipped one of the fixes listed in the README.)
    """
    if cache_path and Path(cache_path).exists():
        cached = pd.read_pickle(cache_path)
        if cached.attrs.get("cache_key") == _cache_key(cfg):
            print(f"Loading filtered options from cache: {cache_path}")
            return cached
        print(f"Cache {cache_path} was built with different code or filters; rebuilding it.")

    df = pd.concat([apply_filters(to_long_format(chunk), cfg) for chunk in iter_raw_options(options_csv)],
                   ignore_index=True)
    if cache_path:
        df.attrs["cache_key"] = _cache_key(cfg)
        df.to_pickle(cache_path)
    return df


def prepare_dataset(options_csv, market, cfg=FilterConfig(), cache_path=None):
    """Full data step: filtered options + market data from the previous trading day (D-1)."""
    df = attach_market_data(load_filtered_options(options_csv, cfg, cache_path), market)
    cols = ["QUOTE_DATE", "MKT_DATE", "EXPIRE_DATE", "Type", "Strike", "Underlying_Price",
            "DTE", "T", "Moneyness", "Bid", "Ask", "Market_Price", "Spread", "Volume",
            "VIX_Price", "Risk_Free_Rate"]
    return df[cols].sort_values(["QUOTE_DATE", "Type", "EXPIRE_DATE", "Strike"]).reset_index(drop=True)
