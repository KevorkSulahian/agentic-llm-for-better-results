from typing import Type

import pandas as pd
from crewai_tools import BaseTool
from pydantic import BaseModel, Field

from finmas.constants import defaults
from finmas.data.market.alpha_vantage import get_fundamental_data, get_income_statement_df
from finmas.data.market.yahoo_finance import get_price_data
from finmas.logger import get_logger
from finmas.utils.common import extract_cols_from_df, get_text_content_file

logger = get_logger(__name__)

BASIC_COLS_MAP = {
    "totalRevenue": "Total Revenue",
    "netIncome": "Net Income",
    "netProfitMargin": "Net Profit Margin (%)",
    "close": "Stock Price",
    "basic_eps": "Basic Quarterly Earnings Per Share",
    "D/E": "Debt to Equity",
    "basic_P/E_ttm": "Basic TTM Price to Earnings Ratio",
    "P/S_ttm": "Price to Sales Ratio TTM",
}

QOQ_COLS_MAP = {
    "totalRevenue_qoq": "Total Revenue QoQ (%)",
    "totalRevenue_ttm_qoq": "Total Revenue TTM QoQ (%)",
    "netIncome_qoq": "Net Income QoQ (%)",
    "netIncome_ttm_qoq": "Net Income TTM QoQ (%)",
    "netProfitMargin_qoq": "Net Profit Margin QoQ (%)",
    "netProfitMargin_ttm_qoq": "Net Profit Margin TTM QoQ (%)",
}

YOY_COLS_MAP = {
    "totalRevenue_ttm": "Total Revenue TTM",
    "totalRevenue_ttm_yoy": "Total Revenue TTM YoY (%)",
    "netIncome_ttm": "Net Income TTM",
    "netIncome_ttm_yoy": "Net Income TTM YoY (%)",
    "netProfitMargin_ttm": "Net Profit Margin TTM (%)",
    "netProfitMargin_ttm_yoy": "Net Profit Margin TTM YoY (%)",
    "basic_eps_ttm": "Basic TTM Earnings Per Share",
    "basic_eps_ttm_yoy": "Basic TTM Earnings Per Share YoY (%)",
}


NUM_QUARTERS = 8


def format_value(value):
    """
    Helper function to format values in a DataFrame.
    Used to prevent .00 being added to integers.
    """
    if isinstance(value, float) and value != int(value):
        return f"{value:,.2f}"
    elif isinstance(value, (int, float)):
        return f"{int(value):,}"
    return value


class StockFundamentalsInput(BaseModel):
    """Input schema for StockFundamentalsTool."""

    ticker: str = Field(..., description="The stock ticker.")


class StockFundamentalsTool(BaseTool):
    name: str = "Stock Fundamentals Tool"
    description: str = "Use this tool to get essential fundamental data for a given stock ticker."
    args_schema: Type[BaseModel] = StockFundamentalsInput

    def _run(self, ticker: str) -> str:
        """Function that returns essential fundamental data for a given ticker in a Markdown table format."""
        include_qoq = defaults["fundamentals_tool"]["include_qoq"]

        df: pd.DataFrame = get_ticker_essentials(ticker)
        if df.empty:
            return f"No data found for {ticker}."

        df = df.tail(NUM_QUARTERS)
        df = df.dropna(axis=0, how="any")
        assert isinstance(df.index, pd.DatetimeIndex)
        df.index = df.index.strftime("%Y-%m-%d")
        df.index.name = "Date"

        df = df.map(format_value)

        basic_df = extract_cols_from_df(df, BASIC_COLS_MAP)
        yoy_df = extract_cols_from_df(df, YOY_COLS_MAP)

        tabulate_config = dict(
            headers="keys",
            tablefmt="github",
            floatfmt=",.2f",
            stralign="right",
        )

        basic_table_context = (
            f"## {ticker} - Fundamentals\n\n"
            f"The date of the latest quarter is: {df.index[-1]}\n\n"
            "This table shows some essential fundamental data for the given stock ticker "
            f"over the last {NUM_QUARTERS} quarters. TTM means Trailing Twelve Months.\n\n"
        )

        yoy_table_context = (
            "### Year over Year Growth for Trailing Twelve Months\n\n"
            "This table shows the year over year growth rates and the trailing twelve months "
            "for total revenue, net income, net profit margin, and basic EPS for the given "
            f"stock ticker over the last {NUM_QUARTERS} quarters.\n\n"
        )

        table_output = (
            basic_table_context
            + basic_df.to_markdown(**tabulate_config)
            + "\n\n"
            + yoy_table_context
            + yoy_df.to_markdown(**tabulate_config)
        )
        if include_qoq:
            qoq_df = extract_cols_from_df(df, QOQ_COLS_MAP) * 100
            qoq_table_context = (
                "### Quarter over Quarter Growth\n\n"
                "This table shows the quarter over quarter growth rates for total revenue, "
                "net income, net profit margin, and basic EPS for the given stock ticker "
                f"over the last {NUM_QUARTERS} quarters.\n\n"
            )
            table_output += "\n\n" + qoq_table_context + qoq_df.to_markdown(**tabulate_config)

        if defaults["save_text_content"]:
            file_path = get_text_content_file(
                ticker, data_type="market_data", suffix="fundamentals"
            )
            file_path.write_text(table_output)
            logger.info(f"Fundamental data saved to {file_path}")

        return table_output


def get_ticker_essentials(ticker: str) -> pd.DataFrame:
    """Gets essential data for a given ticker.

    Price data is fetched from Yahoo Finance and fundamental data is fetched from Alpha Vantage.
    Income statement and Balance sheet data are used to calculate additional metrics.

    Args:
        ticker: The stock ticker

    Returns:
        DataFrame with metrics such as EPS, P/E, P/S, D/E, growth rates.
        Twelve months trailing columns are included.
    """

    # Price data
    price_df = get_price_data(ticker, period="5y")
    price_df.index = price_df.index.tz_localize(tz=None)

    # Income statement
    income_df = get_income_statement_df(ticker, "Quarterly")
    if income_df.empty:
        return pd.DataFrame()

    income_df.sort_index(inplace=True)
    df = income_df[
        ["totalRevenue", "grossProfit", "operatingExpenses", "netIncome", "netProfitMargin"]
    ].copy()

    df["close"] = price_df.reindex(df.index, method="ffill")["close"]

    # Balance sheet
    balance_df = get_fundamental_data(ticker, type="balance", freq="Quarterly")
    balance_df.sort_index(inplace=True)

    income_df = income_df.reindex(balance_df.index, method="ffill")

    df["basic_eps"] = income_df["netIncome"] / balance_df["commonStockSharesOutstanding"]

    # Trailing 12 months
    df["totalRevenue_ttm"] = df["totalRevenue"].rolling(4).sum()
    df["netIncome_ttm"] = df["netIncome"].rolling(4).sum()
    df["netProfitMargin_ttm"] = df["netIncome_ttm"] / df["totalRevenue_ttm"]
    df["basic_eps_ttm"] = df["basic_eps"].rolling(4).sum()
    df["basic_P/E_ttm"] = df["close"] / df["basic_eps_ttm"]
    df["P/S_ttm"] = df["close"] / (
        df["totalRevenue_ttm"] / balance_df["commonStockSharesOutstanding"]
    )

    # Debt to Equity
    df["D/E"] = balance_df["totalLiabilities"] / balance_df["totalShareholderEquity"]

    # Year over year growth for Trailing 12 months
    df["totalRevenue_ttm_yoy"] = df["totalRevenue_ttm"].pct_change(4) * 100
    df["netIncome_ttm_yoy"] = df["netIncome_ttm"].pct_change(4) * 100
    df["netProfitMargin_ttm_yoy"] = df["netProfitMargin_ttm"].pct_change(4) * 100
    df["basic_eps_ttm_yoy"] = df["basic_eps_ttm"].pct_change(4) * 100

    # Quarter over quarter growth
    df["totalRevenue_qoq"] = df["totalRevenue"].pct_change() * 100
    df["totalRevenue_ttm_qoq"] = df["totalRevenue_ttm"].pct_change() * 100
    df["netIncome_qoq"] = df["netIncome"].pct_change() * 100
    df["netIncome_ttm_qoq"] = df["netIncome_ttm"].pct_change() * 100
    df["netProfitMargin_qoq"] = df["netProfitMargin"].pct_change() * 100
    df["netProfitMargin_ttm_qoq"] = df["netProfitMargin_ttm"].pct_change() * 100
    df["basic_eps_qoq"] = df["basic_eps"].pct_change() * 100

    # Adjust netProfitMargin
    df["netProfitMargin"] *= 100
    df["netProfitMargin_ttm"] *= 100

    return df
