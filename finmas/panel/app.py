import datetime as dt
import os
import time
import sys

import hvplot.pandas  # noqa: F401
import pandas as pd
import panel as pn
import plotly.graph_objects as go
import yfinance as yf
from alpha_vantage.fundamentaldata import FundamentalData
from dotenv import find_dotenv, load_dotenv
from panel.viewable import Viewable

# Add the root directory of the project to the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)  # Necessary for windows users.

from finmas.news import get_news_fetcher
from finmas.constants import INCOME_STATEMENT_COLS, defaults
from finmas.panel.formatting import (
    income_statement_config,
    llm_models_config,
    news_config,
    ohlcv_config,
)
from finmas.utils import to_datetime, get_valid_models

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
        options=defaults["llm_providers"],
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
    include_fundamental_data = pn.widgets.Checkbox(name="Include Fundamental Data", value=False)
    include_news = pn.widgets.Checkbox(name="Include News", value=False)

    update_counter = pn.widgets.IntInput(value=0)

    fetch_data_btn = pn.widgets.Button(name="Fetch data", button_type="primary")
    time_elapsed: dict[str, float] = {}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        # Models
        self.llm_model.options = [defaults["llm_model"]]
        self.llm_model.value = defaults["llm_model"]
        self.llm_provider.value = defaults["llm_provider"]
        self.llm_provider.param.watch(self.update_llm_models_tbl, "value")
        self.update_llm_models_tbl(None)

        # Ticker
        self.ticker_select.value = defaults["tickerid"]
        self.ticker_select.options = defaults["tickerids"]

        # News
        self.news_source.options = [defaults["news_source"]]
        self.news_source.value = defaults["news_source"]

        self.fetch_data_btn.on_click(self.fetch_data)
        self.fetch_data(None)

        # About tab
        with open("finmas/panel/about.md", mode="r", encoding="utf-8") as f:
            about = f.read()

        self.about_md = pn.pane.Markdown(about)

    def update_llm_models_tbl(self, event):
        """
        Updates the LLM models table if the LLM provider have changed.
        Initializes the table with the Tabulator widget.
        The current LLM model is selected in the table.
        """
        df = get_valid_models(self.llm_provider.value)

        if event is None:
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
        """Main handler for fetching data for the app"""
        with self.fetch_data_btn.param.update(loading=True):
            start = time.time()
            self.fetch_price_data(event)

            if self.include_fundamental_data.value:
                self.fetch_fundamental_data(event)
            else:
                self.income_statement_tbl = None

            if self.include_news.value:
                self.fetch_news(event)
            else:
                self.news_tbl = None

            self.time_elapsed["fetch_data"] = round(time.time() - start, 1)
            self.update_counter.value += 1  # This trigges updates of plots and tables widgets

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

        if getattr(self, "ohlcv_tbl", None) is None:
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
        # income_statement = income_statement.astype(int) # Too small
        income_statement = income_statement.astype("Int64")

        income_statement["netProfitMargin"] = (
            income_statement["netIncome"] / income_statement["totalRevenue"]
        )

        if getattr(self, "income_statement_tbl", None) is None:
            self.income_statement_tbl = pn.widgets.Tabulator(  # type: ignore
                income_statement, **income_statement_config
            )
        elif isinstance(self.income_statement_tbl, pn.widgets.Tabulator):
            self.income_statement_tbl.value = income_statement.copy()

    def fetch_news(self, event) -> None:
        """Fetch news from the selected news provider and the selected ticker"""
        news_fetcher = get_news_fetcher(self.news_source.value)
        records = news_fetcher.get_news(
            ticker=self.ticker_select.value,
            start=to_datetime(self.news_start.value),
            end=to_datetime(self.news_end.value),
        )

        df = pd.DataFrame.from_records(records)
        df["published"] = df["published"].dt.strftime("%Y-%m-%d")

        if getattr(self, "news_tbl", None) is None:
            self.news_tbl = pn.widgets.Tabulator(  # type: ignore
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

    def get_income_statement_tbl(self, *args, **kwargs):
        """Get the income statement table"""
        if getattr(self, "income_statement_tbl", None) is None:
            return pn.pane.Markdown("No Fundamental data")
        return self.income_statement_tbl

    def get_news_tbl(self, *args, **kwargs):
        """Get the news table"""
        if getattr(self, "news_tbl", None) is None:
            return pn.pane.Markdown("No News data")
        return self.news_tbl

    def get_ta_plot(self, *args, **kwargs) -> go.Figure:
        """Get the plot for Technical analysis"""
        if self.ohlcv_tbl is None or self.ohlcv_tbl.value.empty:
            return pn.pane.Markdown("No Technical Analysis data")
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
        return pn.pane.Plotly(fig)

    def get_income_statement_plot(self, *args, **kwargs) -> go.Figure:
        """Plot income statement table"""
        if (
            getattr(self, "income_statement_tbl", None) is None
            or self.income_statement_tbl.value.empty  # type: ignore
        ):
            return pn.pane.Markdown("No Fundamental data")
        assert isinstance(self.income_statement_tbl, pn.widgets.Tabulator)
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
        return pn.pane.Plotly(fig)

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
                    self.include_fundamental_data,
                    self.include_news,
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
                                        pn.bind(
                                            self.get_ta_plot, update_counter=self.update_counter
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
                                        pn.bind(
                                            self.get_income_statement_plot,
                                            update_counter=self.update_counter,
                                        ),
                                        width=800,
                                        height=400,
                                        margin=10,
                                        title="Income Statement",
                                    ),
                                    pn.bind(
                                        self.get_income_statement_tbl,
                                        update_counter=self.update_counter,
                                    ),
                                ),
                            ),
                            pn.Row(pn.bind(self.get_news_tbl, update_counter=self.update_counter)),
                        ),
                    ),
                    ("Models", pn.Column(self.llm_models_tbl)),
                    ("Crews", pn.Column(pn.pane.Markdown("## Crew Configuration"))),
                    ("About", pn.Column(self.about_md)),
                )
            ),
        )


if pn.state.served:
    FinMAnalysis().servable(title="FinMAS")
