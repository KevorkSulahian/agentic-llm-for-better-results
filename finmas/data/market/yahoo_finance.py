import datetime as dt

import pandas as pd
import yfinance as yf

from finmas.cache_config import cache


@cache.memoize(expire=dt.timedelta(days=1).total_seconds())
def get_price_data(ticker: str, **kwargs) -> pd.DataFrame:
    """
    Get historical price data for a given stock ticker.

    Args:
        ticker: The stock ticker symbol.
        start: The start date for the historical data.
        end: The end date for the historical data.
    """

    df: pd.DataFrame = yf.download(
        ticker,
        interval="1d",
        auto_adjust=True,
        threads=True,
        progress=False,
        **kwargs,
    )

    if df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    assert isinstance(df.index, pd.DatetimeIndex)
    df.index = df.index.tz_localize(None)
    return df
