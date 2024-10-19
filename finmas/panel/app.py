import datetime as dt
import os
import time

import hvplot.pandas  # noqa: F401
import pandas as pd
import panel as pn
import plotly.graph_objects as go
import yfinance as yf
from alpha_vantage.fundamentaldata import FundamentalData
from dotenv import find_dotenv, load_dotenv
from panel.viewable import Viewable

from finmas.news import NewsFetcher
from finmas.panel.constants import INCOME_STATEMENT_COLS, defaults
from finmas.panel.formatting import (
    income_statement_config,
    llm_models_config,
    news_config,
    ohlcv_config,
)
from finmas.utils import get_groq_models, to_datetime

hvplot.extension("plotly")
pn.extension(
    "tabulator",
    "plotly",
    template="fast",
    css_files=["https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css"],
)

load_dotenv(find_dotenv())


class FinMAnalysis(pn.viewable.Viewer):
    llm_provider = pn.widgets.Select(
        name="LLM Provider",
        options=["groq"],
        width=100,
    )
    llm_model = pn.widgets.Select(name="LLM Model", width=200, disabled=True)
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
        options=["Quarterly", "Annual"],
        value=defaults["fundamental_freq"],
        width=100,
    )
    news_source = pn.widgets.Select(name="News Source", width=100)
    news_start = pn.widgets.DatetimeInput(
        name="Start",
        format="%Y-%m-%d",
        value=dt.date.today() - dt.timedelta(days=14),
        width=100,
    )
    news_end = pn.widgets.DatetimeInput(
        name="End",
        format="%Y-%m-%d",
        value=dt.date.today(),
        width=100,
    )
    update_counter = pn.widgets.IntInput(value=0)

    fetch_data_btn = pn.widgets.Button(name="Fetch data", button_type="primary")
    income_statement_tbl = None
    time_elapsed: dict[str, float] = {}

    llm_models_tbl = None
    news_tbl = None

    def __init__(self, **params) -> None:
        super().__init__(**params)
        # Models
        self.llm_model.options = [defaults["llm_model"]]
        self.llm_model.value = defaults["llm_model"]
        self.llm_provider.value = defaults["llm_provider"]
        self.update_llm_models_tbl()

        # Ticker
        self.ticker_select.value = defaults["tickerid"]
        self.ticker_select.options = defaults["tickerids"]

        # News
        self.news_source.options = [defaults["news_source"]]
        self.news_source.value = defaults["news_source"]

        self.fetch_data_btn.on_click(self.fetch_data)
        self.fetch_data(None)

    @pn.depends("llm_provider.value", watch=True)
    def update_llm_models_tbl(self):
        """
        Updates the LLM models table if the LLM provider have changed.
        Initializes the table with the Tabulator widget.
        The current LLM model is selected in the table.
        """
        if self.llm_provider.value == "groq":
            df = get_groq_models()

        if self.llm_models_tbl is None:
            # Initialize the models table
            selection = df.index[df["id"] == self.llm_model.value].tolist()
            self.llm_models_tbl = pn.widgets.Tabulator(
                df,
                on_click=self.handle_llm_models_tbl_click,
                selection=selection,
                **llm_models_config,
            )
        else:
            self.llm_models_tbl.value = df
        self.llm_model.options = df["id"].tolist()

    def handle_llm_models_tbl_click(self, event):
        """Callback for when a row in LLM models table is clicked"""
        llm_model_id = self.llm_models_tbl.value.iloc[event.row]["id"]
        if self.llm_model.value != llm_model_id:
            self.llm_model.value = llm_model_id

    def fetch_data(self, event) -> None:
        """Fetches data for the app"""
        with self.fetch_data_btn.param.update(loading=True):
            start = time.time()
            self.fetch_price_data(event)
            self.fetch_fundamental_data(event)
            self.fetch_news(event)
            self.time_elapsed["fetch_data"] = round(time.time() - start, 1)
            self.update_counter.value += 1

    def _data_alert_box(self, *args, **kwargs) -> pn.pane.Alert:
        message = f"Data fetched for {self.ticker_select.value}"
        alert_type = "success"
        if os.getenv("ALPHAVANTAGE_API_KEY") is None:
            message = "Set ALPHAVANTAGE_API_KEY in the .env file"
            alert_type = "danger"
        message += f". Spent {self.time_elapsed.get('fetch_data', 0)}s"
        return pn.pane.Alert(message, alert_type=alert_type)

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

        if self.freq_select.value == "Quarterly":
            data_func = fundamentals.get_income_statement_quarterly
        else:
            data_func = fundamentals.get_income_statement_annual

        try:
            income_statement: pd.DataFrame = data_func(self.ticker_select.value)[0]
        except Exception as e:
            print(e)
            return

        income_statement.set_index("fiscalDateEnding", inplace=True)
        income_statement.index = pd.to_datetime(income_statement.index)
        income_statement.sort_index(inplace=True, ascending=False)

        income_statement = income_statement[INCOME_STATEMENT_COLS]
        income_statement = income_statement.astype(int)

        income_statement["netProfitMargin"] = (
            income_statement["netIncome"] / income_statement["totalRevenue"]
        )

        if event is None:
            self.income_statement_tbl = pn.widgets.Tabulator(
                income_statement, **income_statement_config
            )
        elif isinstance(self.income_statement_tbl, pn.widgets.Tabulator):
            self.income_statement_tbl.value = income_statement.copy()

    def fetch_news(self, event) -> None:
        """Fetch news from the chosen news provider and the chosen ticker"""
        news_fetcher = NewsFetcher()
        records = news_fetcher.get_news(
            ticker=self.ticker_select.value,
            source=self.news_source.value,
            start=to_datetime(self.news_start.value),
            end=to_datetime(self.news_end.value),
        )

        df = pd.DataFrame.from_records(records)
        df["published"] = df["published"].dt.strftime("%Y-%m-%d")

        if self.news_tbl is None:
            self.news_tbl = pn.widgets.Tabulator(
                df, row_content=self.get_news_content, **news_config
            )
        elif isinstance(self.news_tbl, pn.widgets.Tabulator):
            self.news_tbl.value = df

    @pn.cache
    def get_news_content(self, row: pd.Series) -> pn.pane.HTML:
        """Get the news content as HTML"""
        return pn.Row(
            pn.pane.HTML(row["content"], sizing_mode="stretch_width"),
            max_width=600,
            sizing_mode="stretch_width",
        )

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
                    self.llm_provider,
                    self.llm_model,
                    pn.Row(self.ticker_select, self.freq_select),
                    pn.Row(self.start_picker, self.end_picker),
                    self.news_source,
                    pn.Row(self.news_start, self.news_end),
                    self.fetch_data_btn,
                ),
                pn.bind(self._data_alert_box, update_counter=self.update_counter),
                width=300,
            ),
            pn.Column(
                pn.Tabs(
                    (
                        "Analysis",
                        pn.Column(
                            pn.Row(
                                pn.Column(
                                    pn.Card(
                                        pn.pane.Plotly(
                                            pn.bind(
                                                self.get_ta_plot, update_counter=self.update_counter
                                            )
                                        ),
                                        width=800,
                                        height=400,
                                        margin=10,
                                        title="Technical Analysis",
                                    ),
                                    self.ohlcv_tbl,
                                ),
                                pn.Column(
                                    pn.Card(
                                        pn.pane.Plotly(
                                            pn.bind(
                                                self.get_income_statement_plot,
                                                update_counter=self.update_counter,
                                            )
                                        ),
                                        width=800,
                                        height=400,
                                        margin=10,
                                        title="Income Statement",
                                    ),
                                    self.income_statement_tbl,
                                ),
                            ),
                            pn.Row(self.news_tbl),
                        ),
                    ),
                    ("Models", pn.Column(self.llm_models_tbl)),
                    ("Crews", pn.Column(pn.pane.Markdown("## Crew Configuration"))),
                    # TODO: Include a Markdown file that explains the app and with link to the github repo and the authors.
                    ("About", pn.Column(pn.pane.Markdown("## About"))),
                )
            ),
        )


if pn.state.served:
    FinMAnalysis().servable(title="FinMAS")
