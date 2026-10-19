"""Pricing-error statistics and charts."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = {"BS_Price": "Black-Scholes", "Heston_Price": "Heston (analytical)", "Heston_MC_Price": "Heston (Monte Carlo)"}


def available_models(df):
    return {col: name for col, name in MODELS.items() if col in df.columns}


def error_metrics(df):
    """RMSE and MAE (USD) by model, for all options and by option type."""
    rows = []
    for col, name in available_models(df).items():
        for subset_name, subset in [("all", df), ("call", df[df["Type"] == "call"]), ("put", df[df["Type"] == "put"])]:
            err = (subset["Market_Price"] - subset[col]).dropna()
            rows.append({"Model": name, "Options": subset_name, "N": len(err),
                         "RMSE": np.sqrt(np.mean(err**2)), "MAE": np.mean(np.abs(err)),
                         "Mean_Error": err.mean()})
    return pd.DataFrame(rows)


def error_by_moneyness(df, step=0.01):
    """Mean percentage error (market - model) / market by moneyness bucket and option type."""
    bins = np.round(np.arange(0.9, 1.1 + step / 2, step), 4)
    out = df.copy()
    out["Moneyness_Bin"] = pd.cut(out["Moneyness"], bins=bins, include_lowest=True)
    out = out.dropna(subset=["Moneyness_Bin"])
    out["Bin_Mid"] = out["Moneyness_Bin"].apply(lambda x: x.mid).astype(float)

    pct_cols = []
    for col, name in available_models(df).items():
        out[name] = (out["Market_Price"] - out[col]) / out["Market_Price"]
        pct_cols.append(name)
    return out.groupby(["Type", "Bin_Mid"], observed=True)[pct_cols].mean().reset_index()


def plot_error_by_moneyness(binned, path):
    """One panel for calls and one for puts (the thesis mixed both in a single line)."""
    models = [c for c in binned.columns if c not in ("Type", "Bin_Mid")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, opt_type in zip(axes, ["call", "put"]):
        part = binned[binned["Type"] == opt_type]
        for model, marker in zip(models, ["o", "x", "^"]):
            ax.plot(part["Bin_Mid"], part[model], marker=marker, lw=1.8, label=model)
        ax.axhline(0, color="black", ls="--", lw=1)
        ax.set_title(f"{opt_type.capitalize()}s")
        ax.set_xlabel("Moneyness (K/S)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean percentage error (market - model) / market")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heston_params(params_table, path):
    """Evolution of the calibrated Heston parameters over time."""
    names = ["kappa", "theta", "sigma_v", "rho"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    dates = pd.to_datetime(params_table["Calibration_Date"])
    for ax, name in zip(axes.ravel(), names):
        ax.plot(dates, params_table[name], marker="o", lw=1.5)
        ax.set_title(name)
        ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
