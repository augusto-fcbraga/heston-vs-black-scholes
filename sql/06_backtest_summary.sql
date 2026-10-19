-- 06_backtest_summary.sql
-- Performance summary of the trades table created by 05_trades.sql.

SELECT COUNT(*) AS trades,
       SUM(cost) AS total_premium,
       SUM(pnl) AS total_pnl,
       AVG(pnl) AS avg_pnl_per_trade,
       100.0 * AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) AS win_rate_pct,
       SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) / -SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END) AS profit_factor,
       100.0 * SUM(pnl) / SUM(cost) AS roi_on_total_premium_pct
FROM trades
