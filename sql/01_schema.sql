-- 01_schema.sql
-- Tables for the raw option chains and the market data.
-- Dates are stored as ISO text ('YYYY-MM-DD'), so they sort and compare correctly.

DROP TABLE IF EXISTS raw_options;
DROP TABLE IF EXISTS market_data;
DROP TABLE IF EXISTS options_clean;

-- One row per (quote date, expiry, strike), holding BOTH the call and the put,
-- exactly as in the Kaggle file. Blank volumes are kept as NULL.
CREATE TABLE raw_options (
    quote_date      TEXT NOT NULL,
    expire_date     TEXT NOT NULL,
    dte             REAL NOT NULL,
    underlying_last REAL NOT NULL,
    strike          REAL NOT NULL,
    c_bid           REAL,
    c_ask           REAL,
    c_volume        REAL,
    p_bid           REAL,
    p_ask           REAL,
    p_volume        REAL
);

-- One row per calendar date with at least one series available.
-- vix and spy_close come from Yahoo Finance, dgs3mo (3-month T-Bill, % per year) from FRED.
CREATE TABLE market_data (
    date      TEXT PRIMARY KEY,
    vix       REAL,
    dgs3mo    REAL,
    spy_close REAL
);

CREATE INDEX idx_raw_quote_date ON raw_options (quote_date);
