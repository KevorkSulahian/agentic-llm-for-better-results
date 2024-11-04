from typing import Type

from crewai_tools import BaseTool
from pydantic import BaseModel, Field

from finmas.data.market.alpha_vantage import get_fundamental_data, get_income_statement_df
from finmas.data.market.yahoo_finance import get_price_data

BASIC_COLS_MAP = {
    "totalRevenue": "Total Revenue",
    "netIncome": "Net Income",
    "netProfitMargin": "Net Profit Margin (%)",
    "close": "Stock Price",
    "basic_eps": "Basic EPS",
    "D/E": "Debt to Equity",
    "totalRevenue_ttm": "Total Revenue TTM",
    "netIncome_ttm": "Net Income TTM",
}

QOQ_COLS_MAP = {
    "totalRevenue_qoq": "Total Revenue QoQ",
    "netIncome_qoq": "Net Income QoQ",
    "netProfitMargin_qoq": "Net Profit Margin QoQ",
}


class StockFundamentalsInput(BaseModel):
    """Input schema for StockFundamentalsTool."""

    ticker: str = Field(..., description="The stock ticker.")


class StockFundamentalsTool(BaseTool):
    name: str = "Stock Fundamentals Tool"
    description: str = "Use this tool to get essential fundamental data for a given stock ticker."
    args_schema: Type[BaseModel] = StockFundamentalsInput

    def _run(self, ticker: str) -> str:
        """Function that returns essential fundamental data for a given ticker in a Markdown table format."""
        df = get_ticker_essentials(ticker)
        df = df.dropna(axis=0, how="any")
        df.index = df.index.strftime("%Y-%m-%d")
        df.index.name = "Date"
        df["netProfitMargin"] = df["netProfitMargin"] * 100

        basic_df = df[list(BASIC_COLS_MAP.keys())].copy()
        basic_df.rename(columns=BASIC_COLS_MAP, inplace=True)
        qoq_df = df[list(QOQ_COLS_MAP.keys())].copy()
        qoq_df.rename(columns=QOQ_COLS_MAP, inplace=True)

        tabulate_config = dict(
            headers="keys",
            tablefmt="github",
            floatfmt=".2f",
        )

        table_output = (
            basic_df.to_markdown(**tabulate_config) + "\n\n" + qoq_df.to_markdown(**tabulate_config)
        )

        return table_output


def get_ticker_essentials(ticker: str):
    """Gets essential data for a given ticker.

    - Price data
    - Income statement
    - Balance sheet
    """
    NUM_QUARTERS = 8

    # Price data
    price_df = get_price_data(ticker, period="5y")
    price_df.index = price_df.index.tz_localize(tz=None)

    # Income statement
    income_df = get_income_statement_df(ticker, "Quarterly")
    income_df.sort_index(inplace=True)
    income_df = income_df.tail(NUM_QUARTERS)
    df = income_df[["totalRevenue", "netIncome", "netProfitMargin"]].copy()

    df["close"] = price_df.reindex(df.index, method="ffill")["close"]

    # Balance sheet
    balance_df = get_fundamental_data(ticker, type="balance", freq="Quarterly")
    balance_df.sort_index(inplace=True)
    balance_df = balance_df.tail(NUM_QUARTERS)

    df["basic_eps"] = income_df["netIncome"] / balance_df["commonStockSharesOutstanding"]

    # df["P/E"] = df["close"] / df["eps"]
    # df["P/S"] = df["close"] / (df["totalRevenue"] / balance_df["commonStockSharesOutstanding"])

    # Trailing 12 months
    df["totalRevenue_ttm"] = df["totalRevenue"].rolling(4).sum()
    df["netIncome_ttm"] = df["netIncome"].rolling(4).sum()

    # Debt to Equity
    df["D/E"] = balance_df["totalLiabilities"] / balance_df["totalShareholderEquity"]

    # Quarter over quarter growth
    df["totalRevenue_qoq"] = df["totalRevenue"].pct_change()
    df["netIncome_qoq"] = df["netIncome"].pct_change()
    df["netProfitMargin_qoq"] = df["netProfitMargin"].pct_change()
    df["basic_eps_qoq"] = df["basic_eps"].pct_change()

    return df
