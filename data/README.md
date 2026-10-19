# Data

The raw option data is not included in this repository because of its size.

1. Download the SPY option chains for 2020–2022 from Kaggle (*SPY Option Chains Q1 2020 – Q4 2022*).
2. Combine the files into a single CSV and save it here as `spy_2020_2022.csv`.
3. Run `python run_pipeline.py --options-csv data/spy_2020_2022.csv`.

Expected columns (brackets and spaces in the headers are removed automatically):
`QUOTE_DATE, UNDERLYING_LAST, EXPIRE_DATE, DTE, C_BID, C_ASK, C_VOLUME, STRIKE, P_BID, P_ASK, P_VOLUME`.

Market data (VIX, SPY closes and the 3-month T-Bill) is downloaded automatically on the first run
and cached in `data/market_data.csv`.

`data/sample/` can be regenerated at any time with `python -m scripts.make_sample_data`; it contains
synthetic data priced with known Heston parameters, used to test the pipeline offline.
