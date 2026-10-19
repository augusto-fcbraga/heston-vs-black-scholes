-- 04_error_by_bucket.sql
-- Pricing errors of one model ({model}: BS_Price, Heston_Price or Heston_MC_Price) by option
-- type and moneyness bucket (ITM / ATM / OTM). Error = market_price - model price.

WITH errors AS (
    SELECT type,
           CASE WHEN moneyness BETWEEN 0.97 AND 1.03 THEN 'ATM'
                WHEN (type = 'call' AND moneyness < 0.97) OR (type = 'put' AND moneyness > 1.03) THEN 'ITM'
                ELSE 'OTM' END AS bucket,
           market_price - {model} AS err
    FROM priced_options
    WHERE {model} IS NOT NULL
)
SELECT type, bucket, COUNT(*) AS n,
       SQRT(AVG(err * err)) AS rmse,
       AVG(ABS(err)) AS mae,
       AVG(err) AS mean_error
FROM errors
GROUP BY type, bucket
ORDER BY type, bucket
