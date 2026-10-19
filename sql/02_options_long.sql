-- 02_options_long.sql
-- Reshape raw_options (one row = call AND put) into options_clean (one row = one option)
-- and apply the study's filters. This is the fix for the thesis bug: every option uses
-- its OWN bid, ask and volume, and a blank volume counts as zero trades (COALESCE).
-- Parameters: :min_volume, :min_premium, :max_rel_spread, :min_dte, :max_dte,
--             :min_moneyness, :max_moneyness (bound from src/data.py FilterConfig).

DROP TABLE IF EXISTS options_clean;

CREATE TABLE options_clean AS
WITH legs AS (
    -- calls
    SELECT quote_date, expire_date, dte, underlying_last AS underlying_price, strike,
           'call' AS type, c_bid AS bid, c_ask AS ask, COALESCE(c_volume, 0) AS volume
    FROM raw_options

    UNION ALL

    -- puts
    SELECT quote_date, expire_date, dte, underlying_last AS underlying_price, strike,
           'put' AS type, p_bid AS bid, p_ask AS ask, COALESCE(p_volume, 0) AS volume
    FROM raw_options
),
priced AS (
    SELECT *,
           (bid + ask) / 2.0         AS market_price,
           ask - bid                 AS spread,
           dte / 365.0               AS t,
           strike / underlying_price AS moneyness
    FROM legs
)
SELECT *
FROM priced
WHERE volume > :min_volume
  AND market_price >= :min_premium
  AND bid > 0
  AND spread >= 0
  AND spread / market_price <= :max_rel_spread
  AND dte BETWEEN :min_dte AND :max_dte
  AND moneyness BETWEEN :min_moneyness AND :max_moneyness
