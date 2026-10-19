-- 05_trades.sql
-- Backtest trades for one model: buy at the ask when ask < {model} * (1 - :threshold),
-- only on the FIRST day each contract (type, strike, expiry) signals, and hold to expiry.
-- Settlement: SPY close on the expiry date, or the last close before it.

DROP TABLE IF EXISTS trades;

CREATE TABLE trades AS
WITH signals AS (
    SELECT quote_date, expire_date, type, strike, ask,
           ROW_NUMBER() OVER (PARTITION BY type, strike, expire_date ORDER BY quote_date) AS rn
    FROM priced_options
    WHERE ask < {model} * (1 - :threshold)
),
first_signal AS (
    SELECT * FROM signals WHERE rn = 1
),
settled AS (
    SELECT f.*,
           (SELECT m.spy_close FROM market_data m
             WHERE m.date <= f.expire_date AND m.spy_close IS NOT NULL
             ORDER BY m.date DESC LIMIT 1) AS spy_expiry
    FROM first_signal f
    WHERE f.expire_date <= (SELECT MAX(date) FROM market_data WHERE spy_close IS NOT NULL)
),
payoff AS (
    SELECT *, CASE WHEN type = 'call' THEN MAX(0, spy_expiry - strike)
                   ELSE MAX(0, strike - spy_expiry) END AS final_value
    FROM settled
)
SELECT quote_date, expire_date, type, strike, ask AS entry_price, spy_expiry, final_value,
       ask * 100 AS cost,
       (final_value - ask) * 100 AS pnl
FROM payoff
