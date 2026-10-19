-- 03_attach_market_data.sql
-- Add the VIX and the risk-free rate from the PREVIOUS trading day (D-1) to every option,
-- skipping blank values (US holidays, FRED gaps). Market data is looked up once per quote
-- date and joined back to the options. risk_free_rate = LN(1 + dgs3mo / 100).

WITH quote_days AS (
    SELECT DISTINCT quote_date FROM options_clean
),
d1 AS (
    SELECT q.quote_date,
           (SELECT MAX(m.date) FROM market_data m WHERE m.date < q.quote_date) AS mkt_date,
           (SELECT m.vix FROM market_data m WHERE m.date < q.quote_date AND m.vix IS NOT NULL ORDER BY m.date DESC LIMIT 1) AS vix_price,
           (SELECT m.dgs3mo FROM market_data m WHERE m.date < q.quote_date AND m.dgs3mo IS NOT NULL ORDER BY m.date DESC LIMIT 1) AS dgs3mo
    FROM quote_days q
)
SELECT o.quote_date, d1.mkt_date, o.expire_date, o.type, o.strike, o.underlying_price, o.dte, o.t,
       o.moneyness, o.bid, o.ask, o.market_price, o.spread, o.volume, d1.vix_price,
       LN(1 + d1.dgs3mo / 100.0) AS risk_free_rate
FROM options_clean o
JOIN d1 ON d1.quote_date = o.quote_date
WHERE d1.vix_price IS NOT NULL AND d1.dgs3mo IS NOT NULL
ORDER BY o.quote_date, o.type, o.expire_date, o.strike
