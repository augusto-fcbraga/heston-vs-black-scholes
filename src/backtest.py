"""Mispricing backtest: buy options the model says are cheap and hold them to expiry.

Rule (same as the thesis): buy when entry price < model price * (1 - threshold).

Changes compared with the thesis script
---------------------------------------
1. First signal only: a contract (type, strike, expiry) is bought at most once, on
   the first day it triggers a signal. The thesis counted a new trade every day the
   same contract kept signalling, so the same position was bought many times.
2. Entry at the ask: the thesis bought at the mid price, which is not achievable.
3. P&L is booked on the expiry date (when it is actually realised). The thesis
   grouped P&L by entry date, which puts future information into the equity curve
   and understates the drawdown timing.
4. Capital: besides the thesis ROI (P&L / sum of all premiums), the backtest reports
   the peak capital tied up in open positions and the return on that capital.
"""

import numpy as np
import pandas as pd

CONTRACT_KEY = ["Type", "Strike", "EXPIRE_DATE"]


def settlement_prices(spy_close, expiries):
    """SPY close on each expiry date (last available close on or before that date)."""
    closes = spy_close.dropna().sort_index()
    closes.index = closes.index.astype("datetime64[ns]")
    exp = pd.DataFrame({"EXPIRE_DATE": pd.to_datetime(sorted(set(expiries))).astype("datetime64[ns]")})
    out = pd.merge_asof(exp, closes.rename("SPY_Expiry").rename_axis("Date").reset_index(),
                        left_on="EXPIRE_DATE", right_on="Date", direction="backward")
    # Expiries after the last available close cannot be settled.
    out.loc[out["EXPIRE_DATE"] > closes.index.max(), "SPY_Expiry"] = np.nan
    return out[["EXPIRE_DATE", "SPY_Expiry"]]


def generate_trades(df, model_col, spy_close, threshold=0.20, entry_col="Ask", first_signal_only=True):
    """Select trades and compute their payoff at expiry."""
    signals = df[df[entry_col] < df[model_col] * (1 - threshold)].copy()
    if signals.empty:
        return signals

    if first_signal_only:
        signals = signals.sort_values("QUOTE_DATE").drop_duplicates(CONTRACT_KEY, keep="first")

    signals["EXPIRE_DATE"] = signals["EXPIRE_DATE"].astype("datetime64[ns]")
    signals = signals.merge(settlement_prices(spy_close, signals["EXPIRE_DATE"]), on="EXPIRE_DATE", how="left")
    signals = signals.dropna(subset=["SPY_Expiry"])

    signals["Entry_Price"] = signals[entry_col]
    signals["Final_Value"] = np.where(
        signals["Type"] == "call",
        np.maximum(0.0, signals["SPY_Expiry"] - signals["Strike"]),
        np.maximum(0.0, signals["Strike"] - signals["SPY_Expiry"]),
    )
    signals["Cost"] = signals["Entry_Price"] * 100          # 1 contract = 100 shares
    signals["PnL"] = (signals["Final_Value"] - signals["Entry_Price"]) * 100
    return signals.reset_index(drop=True)


def peak_capital(trades):
    """Maximum premium tied up in open positions at any point in time."""
    flows = pd.concat([
        pd.Series(trades["Cost"].to_numpy(), index=trades["QUOTE_DATE"]),
        pd.Series(-trades["Cost"].to_numpy(), index=trades["EXPIRE_DATE"]),
    ])
    # On the same date, expiries are released before new positions are opened.
    flows = flows.groupby(level=0).sum().sort_index()
    return float(flows.cumsum().max())


def summarize(trades):
    """Performance metrics for a set of trades."""
    if trades.empty:
        return {"Trades": 0}

    pnl = trades["PnL"]
    gross_loss = abs(pnl[pnl < 0].sum())
    realised = trades.groupby("EXPIRE_DATE")["PnL"].sum().sort_index().cumsum()
    capital = peak_capital(trades)

    return {
        "Trades": len(trades),
        "Total_Premium": trades["Cost"].sum(),
        "Total_PnL": pnl.sum(),
        "Avg_PnL_per_Trade": pnl.mean(),
        "Win_Rate_%": 100 * (pnl > 0).mean(),
        "Profit_Factor": pnl[pnl > 0].sum() / gross_loss if gross_loss > 0 else np.inf,
        # Peak measured from zero, so losses on the very first trades also count as drawdown.
        "Max_Drawdown": float(min(0.0, (realised - realised.cummax().clip(lower=0)).min())),
        "Peak_Capital": capital,
        "Return_on_Peak_Capital_%": 100 * pnl.sum() / capital if capital > 0 else np.nan,
        "ROI_on_Total_Premium_% (thesis definition)": 100 * pnl.sum() / trades["Cost"].sum(),
    }


def run_backtest(df, model_cols, spy_close, threshold=0.20, entry_col="Ask", first_signal_only=True):
    """Run the backtest for several models and return (summary table, trades per model)."""
    summaries, all_trades = {}, {}
    for col in model_cols:
        trades = generate_trades(df, col, spy_close, threshold, entry_col, first_signal_only)
        all_trades[col] = trades
        summaries[col] = summarize(trades)
    return pd.DataFrame(summaries), all_trades
