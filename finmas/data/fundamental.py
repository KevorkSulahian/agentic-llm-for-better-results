import datetime as dt
import os

import pandas as pd
from alpha_vantage.fundamentaldata import FundamentalData

from finmas.cache_config import cache


@cache.memoize(expire=dt.timedelta(days=1).total_seconds())
def get_income_statement(ticker: str, freq: str) -> pd.DataFrame:
    if os.getenv("ALPHAVANTAGE_API_KEY") is None:
        return pd.DataFrame()
    fundamentals = FundamentalData(output_format="pandas")

    if freq == "Quarterly":
        data_func = fundamentals.get_income_statement_quarterly
    else:
        data_func = fundamentals.get_income_statement_annual

    try:
        income_statement: pd.DataFrame = data_func(ticker)[0]
    except Exception as e:
        print(e)
        return pd.DataFrame()

    return income_statement
