import datetime as dt

import pandas as pd
import yfinance as yf

from finmas.cache_config import cache


@cache.memoize(expire=dt.timedelta(days=1).total_seconds())
def get_price_data(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    df: pd.DataFrame = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    return df
