import datetime as dt
from typing import Type

import numpy as np
import pandas as pd
from crewai_tools import BaseTool
from pydantic import BaseModel, Field
from ta.momentum import rsi
from ta.trend import sma_indicator
from ta.volatility import bollinger_pband

from finmas.constants import defaults
from finmas.data.market.yahoo_finance import get_price_data
from finmas.logger import get_logger
from finmas.utils.common import extract_cols_from_df, get_text_content_file

logger = get_logger(__name__)

NUM_PERIODS = defaults["technical_analysis_periods"]

TA_COLS_MAP = {
    "close": "Close price",
    "sma_50": "SMA 50 week",
    "sma_20": "SMA 20 week",
    "sma_trend": "SMA Trend",
    "rsi_14": "RSI 14",
    "bb_pband": "Bollinger Band Percentage %",
}


class TechnicalAnalysisInput(BaseModel):
    """Input schema for StockFundamentalsTool."""

    ticker: str = Field(..., description="The stock ticker.")


class TechnicalAnalysisTool(BaseTool):
    name: str = "Technical Analysis Tool"
    description: str = (
        "Use this tool to get essential technical indicators for a given stock ticker."
    )
    args_schema: Type[BaseModel] = TechnicalAnalysisInput
    end_date: dt.date

    def __init__(self, end_date: dt.date | None = None, **kwargs):
        end_date = end_date or dt.date.today()
        super().__init__(end_date=end_date, **kwargs)

    def _run(self, ticker: str) -> str:
        """Function that returns essential technical indicators and price for a given ticker in a Markdown table format."""
        start = (self.end_date - pd.DateOffset(years=5)).date()
        df: pd.DataFrame = get_technical_indicators(ticker, start=start, end=self.end_date)
        df = df.tail(NUM_PERIODS)
        df = df.dropna(axis=0, how="any")
        assert isinstance(df.index, pd.DatetimeIndex)
        df.index = df.index.strftime("%Y-%m-%d")
        df.index.name = "Date"

        ta_df = extract_cols_from_df(df, TA_COLS_MAP)

        tabulate_config = dict(
            headers="keys",
            tablefmt="github",
            floatfmt=",.2f",
            stralign="right",
        )

        ta_table_context = (
            f"## {ticker} - Technical Indicators\n\n"
            f"Date of the latest technical analysis data is: {df.index[-1]}\n\n"
            "This table shows some technical indicators for the given stock ticker "
            f"over the last {NUM_PERIODS} weeks. SMA = Simple Moving Average represent the trend. "
            "RSI = Relative Strength Index and Bollinger Band represent the momentum.\n\n"
            "When RSI is above 70, the stock is considered overbought. When RSI is below 30, the stock is considered oversold.\n\n"
            "When the Bollinger Band Percentage is above 100, the stock is considered overbought. "
            "When the Bollinger Band Percentage is below 0, the stock is considered oversold.\n\n"
        )

        table_output = ta_table_context + ta_df.to_markdown(**tabulate_config)

        if defaults["save_text_content"]:
            file_path = get_text_content_file(ticker, data_type="market_data", suffix="ta")
            file_path.write_text(table_output)
            logger.info(f"Technical analysis data saved to {file_path}")

        return table_output


# @cache.memoize(expire=dt.timedelta(days=1).total_seconds())
def get_technical_indicators(ticker: str, **kwargs) -> pd.DataFrame:
    """
    Get technical indicators for a given stock ticker.

    Args:
        ticker: The stock ticker symbol.
        period: The period for the technical analysis data.
        Any additional keyword arguments are passed to the get_price_data function.

    Returns:
        DataFrame with technical indicators SMA, RSI, and Bollinger Band.
    """
    df = get_price_data(ticker, **kwargs)

    df = df.drop(columns=["volume"])

    # Resample to weekly frequency with the Monday as the first day of the week
    df = df.resample("W-FRI").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
    )

    df["sma_20"] = sma_indicator(df["close"], window=20)
    df["sma_50"] = sma_indicator(df["close"], window=50)
    df["sma_trend"] = np.where(
        (df["close"] > df["sma_50"]) & (df["close"] > df["sma_50"]),
        "Up",
        np.where(
            (df["close"] < df["sma_20"]) & (df["close"] < df["sma_50"]),
            "Down",
            "Neutral",
        ),
    )
    df["rsi_14"] = rsi(df["close"], window=14)
    df["bb_pband"] = bollinger_pband(df["close"], window=20) * 100
    df["ret_1w"] = df["close"].pct_change(periods=1)

    return df
