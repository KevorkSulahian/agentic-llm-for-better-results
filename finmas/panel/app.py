import datetime as dt
import os

import hvplot.pandas  # noqa: F401
import pandas as pd
import panel as pn
import plotly.graph_objects as go
import yfinance as yf
from alpha_vantage.fundamentaldata import FundamentalData
from dotenv import find_dotenv, load_dotenv
from panel.viewable import Viewable

from finmas.panel.constants import INCOME_STATEMENT_COLS, defaults
from finmas.panel.formatting import income_statement_config, ohlcv_config

hvplot.extension("plotly")
pn.extension("tabulator", "plotly", template="fast")

load_dotenv(find_dotenv())


class FinMAnalysis(pn.viewable.Viewer):
    ticker_select = pn.widgets.Select(name="Ticker", width=100)
    start_picker = pn.widgets.DatetimeInput(
        name="Start",
        format="%Y-%m-%d",
        value=dt.date.fromisoformat(defaults["start_date"]),
        width=100,
    )
    end_picker = pn.widgets.DatetimeInput(
        name="End",
        format="%Y-%m-%d",
        value=dt.date.today(),
        width=100,
    )
    freq_select = pn.widgets.Select(
        name="Fundamental Freq",
        options=["quarterly", "annual"],
        value=defaults["fundamental_freq"],
        width=100,
    )
    update_counter = pn.widgets.IntInput(value=0)

    fetch_data_btn = pn.widgets.Button(name="Fetch data", button_type="primary")
    # ohlcv_tbl = None
    income_statement_tbl = None

    def __init__(self, **params) -> None:
        super().__init__(**params)

        self.ticker_select.value = defaults["tickerid"]
        self.ticker_select.options = defaults["tickerids"]

        self.fetch_data_btn.on_click(self.fetch_data)
        self.fetch_data(None)

    def fetch_data(self, event) -> None:
        with self.fetch_data_btn.param.update(loading=True):
            self.fetch_price_data(event)
            self.fetch_fundamental_data(event)
            self.update_counter.value += 1

    @pn.depends("update_counter")
    def _data_alert_box(self) -> pn.pane.Alert:
        message = f"Data fetched for {self.ticker_select.value}"
        alert_type = "success"
        if os.getenv("ALPHAVANTAGE_API_KEY") is None:
            message = "Set ALPHAVANTAGE_API_KEY in the .env file"
            alert_type = "danger"
        return pn.pane.Alert(message, alert_type=alert_type, margin=0)

    def fetch_price_data(self, event) -> None:
        df: pd.DataFrame = yf.download(
            self.ticker_select.value,
            start=self.start_picker.value,
            end=self.end_picker.value,
            interval="1d",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
        df.reset_index(inplace=True)
        df.columns = df.columns.str.lower()

        if event is None:
            self.ohlcv_tbl = pn.widgets.Tabulator(df, **ohlcv_config)
        elif isinstance(self.ohlcv_tbl, pn.widgets.Tabulator):
            self.ohlcv_tbl.value = df

    def fetch_fundamental_data(self, event) -> None:
        """Fetch fundamental data"""
        if os.getenv("ALPHAVANTAGE_API_KEY") is None:
            return
        fundamentals = FundamentalData(output_format="pandas")

        if self.freq_select.value == "quarterly":
            data_func = fundamentals.get_income_statement_quarterly
        else:
            data_func = fundamentals.get_income_statement_annual

        income_statement: pd.DataFrame = data_func(self.ticker_select.value)[0]

        income_statement.set_index("fiscalDateEnding", inplace=True)
        income_statement.index = pd.to_datetime(income_statement.index)
        income_statement.sort_index(inplace=True, ascending=False)

        income_statement = income_statement[INCOME_STATEMENT_COLS]
        income_statement = income_statement.astype(int)

        # Add column for profit margin
        income_statement["netProfitMargin"] = (
            income_statement["netIncome"] / income_statement["totalRevenue"]
        )

        if event is None:
            self.income_statement_tbl = pn.widgets.Tabulator(
                income_statement, **income_statement_config
            )
        elif isinstance(self.income_statement_tbl, pn.widgets.Tabulator):
            self.income_statement_tbl.value = income_statement.copy()

    def get_ta_plot(self, *args, **kwargs) -> go.Figure:
        """Plot Technical analysis"""
        if self.ohlcv_tbl is None or self.ohlcv_tbl.value.empty:
            return None
        df = self.ohlcv_tbl.value
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df["date"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                )
            ]
        )
        fig.update_layout(
            autosize=True,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Date",
            yaxis=dict(title="Price (USD)", side="right"),
        )
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
            type="date",
        )
        return fig

    def get_income_statement_plot(self, *args, **kwargs) -> go.Figure:
        """Plot income statement table"""
        if self.income_statement_tbl is None or self.income_statement_tbl.value.empty:
            return None
        df = self.income_statement_tbl.value

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df.index, y=df["totalRevenue"], mode="lines", name="Total Revenue")
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["operatingExpenses"], mode="lines", name="Operating Expenses"
            )
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["grossProfit"], mode="lines", name="Gross Profit")
        )
        fig.add_trace(go.Scatter(x=df.index, y=df["netIncome"], mode="lines", name="Net Income"))

        fig.update_layout(
            autosize=True,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="",
            yaxis=dict(title="USD", side="right"),
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=0),
        )
        return fig

    def __panel__(self) -> Viewable:
        return pn.Row(
            pn.Column(
                pn.WidgetBox(
                    self.ticker_select,
                    pn.Row(self.start_picker, self.end_picker),
                    self.freq_select,
                    self.fetch_data_btn,
                ),
                self._data_alert_box,
                width=300,
            ),
            pn.Column(
                pn.Card(
                    pn.pane.Plotly(pn.bind(self.get_ta_plot, update_counter=self.update_counter)),
                    width=800,
                    height=600,
                    margin=10,
                    title="Technical Analysis",
                ),
                self.ohlcv_tbl,
            ),
            pn.Column(
                pn.Card(
                    pn.pane.Plotly(
                        pn.bind(self.get_income_statement_plot, update_counter=self.update_counter)
                    ),
                    width=800,
                    height=400,
                    margin=10,
                    title="Income Statement",
                ),
                self.income_statement_tbl,
            ),
        )


if pn.state.served:
    FinMAnalysis().servable(title="FinMAS")
