"""SQLite database for the project.

Python does the maths (pricing, calibration); SQL does the data work: storing,
filtering, joining and aggregating. The queries live in `sql/` and are run by
`run_sql_file`, so each one can be read, tested and explained on its own.
"""

import math
import os
import sqlite3
from pathlib import Path

import pandas as pd

from .data import FilterConfig, iter_raw_options

SQL_DIR = Path(os.environ.get("SQL_DIR", Path(__file__).resolve().parent.parent / "sql"))
MODEL_COLUMNS = {"BS_Price", "Heston_Price", "Heston_MC_Price"}   # allowed values for {model}


def connect(db_path):
    """Open the database. SQRT and LN are registered because some SQLite builds lack math functions."""
    conn = sqlite3.connect(db_path)
    conn.create_function("SQRT", 1, lambda x: math.sqrt(x) if x is not None and x >= 0 else None)
    conn.create_function("LN", 1, lambda x: math.log(x) if x is not None and x > 0 else None)
    return conn


def _date_text(series):
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d")


def run_sql_file(conn, name, params=None, model=None):
    """Run a file from sql/. If the last statement is a SELECT, return its result as a DataFrame.

    `params` are bound with SQLite named parameters (:threshold, :min_volume, ...).
    `{model}` is replaced by a model column name; only whitelisted names are accepted.
    """
    sql = (SQL_DIR / name).read_text(encoding="utf-8")
    if "{model}" in sql:
        if model not in MODEL_COLUMNS:
            raise ValueError(f"model must be one of {sorted(MODEL_COLUMNS)}")
        sql = sql.replace("{model}", model)

    statements = [_strip_comments(s) for s in sql.split(";")]
    *setup, last = [s for s in statements if s]
    for stmt in setup:
        conn.execute(stmt, params or {})
    conn.commit()
    if last.upper().startswith(("SELECT", "WITH")):
        return pd.read_sql_query(last, conn, params=params or {})
    conn.execute(last, params or {})
    conn.commit()
    return None


def _strip_comments(stmt):
    """Drop full-line `--` comments, so a header comment does not hide what a statement is."""
    return "\n".join(line for line in stmt.splitlines() if not line.strip().startswith("--")).strip()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def build_database(db_path, options_csv, market):
    """Create the schema and load the raw option chains and market data.

    The raw options are loaded as they are in the CSV (blank volumes stay NULL);
    cleaning happens in SQL, in sql/02_options_long.sql.
    """
    conn = connect(db_path)
    run_sql_file(conn, "01_schema.sql")

    for chunk in iter_raw_options(options_csv, keep_blank_volumes=True):
        out = chunk.rename(columns=str.lower)
        out["quote_date"] = _date_text(out["quote_date"])
        out["expire_date"] = _date_text(out["expire_date"])
        out.to_sql("raw_options", conn, if_exists="append", index=False)

    mkt = market.rename(columns={"VIX": "vix", "DGS3MO": "dgs3mo", "SPY_Close": "spy_close"})
    mkt = mkt[["vix", "dgs3mo", "spy_close"]].rename_axis("date").reset_index()
    mkt["date"] = _date_text(mkt["date"])
    mkt.to_sql("market_data", conn, if_exists="append", index=False)
    conn.commit()
    return conn


def write_priced_options(conn, df):
    """Store the priced options so errors and the backtest can be analysed in SQL."""
    cols = ["QUOTE_DATE", "EXPIRE_DATE", "Type", "Strike", "Underlying_Price", "DTE", "Moneyness",
            "Bid", "Ask", "Market_Price"] + [c for c in sorted(MODEL_COLUMNS) if c in df.columns]
    out = df[cols].copy()
    out.columns = [c.lower() for c in cols]
    out["quote_date"] = _date_text(out["quote_date"])
    out["expire_date"] = _date_text(out["expire_date"])
    out.to_sql("priced_options", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_priced_contract ON priced_options (type, strike, expire_date, quote_date)")
    conn.commit()


# ---------------------------------------------------------------------------
# Data step in SQL (same output as data.prepare_dataset)
# ---------------------------------------------------------------------------
def prepare_dataset_sql(conn, cfg=FilterConfig()):
    """Run the SQL data step and return the same columns as data.prepare_dataset."""
    params = {"min_volume": cfg.min_volume, "min_premium": cfg.min_premium,
              "max_rel_spread": cfg.max_rel_spread, "min_dte": cfg.min_dte, "max_dte": cfg.max_dte,
              "min_moneyness": cfg.min_moneyness, "max_moneyness": cfg.max_moneyness}
    run_sql_file(conn, "02_options_long.sql", params)
    df = run_sql_file(conn, "03_attach_market_data.sql")

    rename = {"quote_date": "QUOTE_DATE", "mkt_date": "MKT_DATE", "expire_date": "EXPIRE_DATE",
              "type": "Type", "strike": "Strike", "underlying_price": "Underlying_Price", "dte": "DTE",
              "t": "T", "moneyness": "Moneyness", "bid": "Bid", "ask": "Ask", "market_price": "Market_Price",
              "spread": "Spread", "volume": "Volume", "vix_price": "VIX_Price", "risk_free_rate": "Risk_Free_Rate"}
    df = df.rename(columns=rename)
    for col in ("QUOTE_DATE", "MKT_DATE", "EXPIRE_DATE"):
        df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
    return df[list(rename.values())].sort_values(["QUOTE_DATE", "Type", "EXPIRE_DATE", "Strike"]).reset_index(drop=True)
