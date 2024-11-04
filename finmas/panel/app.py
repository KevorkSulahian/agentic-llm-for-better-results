import datetime as dt
import os
import time
from pathlib import Path
from typing import Union

import hvplot.pandas  # noqa: F401
import pandas as pd
import panel as pn
import plotly.graph_objects as go
from dotenv import find_dotenv, load_dotenv
from panel.viewable import Viewable

from finmas.constants import INCOME_STATEMENT_COLS, defaults
from finmas.crews.news.crew import NewsAnalysisCrew
from finmas.crews.sec.crew import SECFilingCrew
from finmas.crews.sec_mda_risk_factors.crew import SECFilingSectionsCrew
from finmas.crews.utils import (
    CrewRunMetrics,
    get_usage_metrics_as_string,
    get_yaml_config_as_markdown,
    save_crew_output,
)
from finmas.data import get_income_statement, get_price_data
from finmas.data.news import get_news_fetcher
from finmas.data.sec.filings import filings_to_df, get_sec_filings
from finmas.panel.formatting import (
    income_statement_config,
    llm_models_config,
    news_config,
    ohlcv_config,
    sec_filings_config,
    tickers_config,
)
from finmas.utils.common import format_time_spent, get_tickers_df, get_valid_models, to_datetime

hvplot.extension("plotly")
pn.extension(
    "tabulator",
    "plotly",
    template="fast",
    css_files=["https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css"],
)

load_dotenv(find_dotenv())


class FinMAS(pn.viewable.Viewer):
    llm_provider = pn.widgets.Select(
        name="LLM Provider",
        options=defaults["llm_providers"],
        width=100,
    )
    llm_model = pn.widgets.Select(name="LLM Model", width=200, disabled=True)
    embedding_model_news = pn.widgets.Select(
        name="Embedding Model News",
        value=defaults["hf_embedding_model_news"],
        options=defaults["hf_embedding_models"],
        width=250,
    )
    embedding_model_sec = pn.widgets.Select(
        name="Embedding Model SEC",
        value=defaults["hf_embedding_model_sec"],
        options=defaults["hf_embedding_models"],
        width=250,
    )
    ticker_select = pn.widgets.Select(name="Ticker", disabled=True, width=100)
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
    news_source = pn.widgets.Select(
        name="News Source",
        value=defaults["news_source"],
        options=defaults["news_sources"],
        width=100,
    )
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
    only_sp500_tickers = pn.widgets.Checkbox(
        name="SP500 Tickers", value=defaults["only_sp500_tickers"]
    )

    update_counter = pn.widgets.IntInput(value=0)

    fetch_data_btn = pn.widgets.Button(name="Fetch data", button_type="primary")
    time_elapsed: dict[str, float] = {}

    crew_select = pn.widgets.Select(name="Crew", width=100)
    llm_temperature = pn.widgets.FloatInput(
        name="LLM Temp.", value=defaults["llm_temperature"], start=0.0, end=1.0, width=100
    )
    llm_max_tokens = pn.widgets.IntInput(
        name="Max tokens", value=defaults["llm_max_tokens"], width=100
    )
    similarity_top_k = pn.widgets.IntInput(
        name="Similarity Top K", value=defaults["similarity_top_k"], width=100
    )
    compress_sec_filing = pn.widgets.Checkbox(
        name="Compress SEC Filing with keywords search", value=False, align="center"
    )
    filing_types = pn.widgets.MultiSelect(
        name="Filing Types",
        value=defaults["sec_filing_types_selected"],
        options=defaults["sec_filing_types"],
        width=200,
    )
    kickoff_crew_btn = pn.widgets.Button(name="Kickoff Crew", button_type="primary", align="center")
    crew_agents_config_md = pn.pane.Markdown("Agents", sizing_mode="stretch_width")
    crew_tasks_config_md = pn.pane.Markdown("Tasks", sizing_mode="stretch_width")
    crew_usage_metrics = pn.pane.Markdown("")
    crew_output_status = pn.pane.Alert("No Crew output generated yet.", alert_type="warning")
    crew_output = pn.pane.Markdown("", sizing_mode="stretch_width")

    def __init__(self, ticker: str | None = None, **params) -> None:
        super().__init__(**params)
        # Models
        self.llm_provider.value = defaults["llm_provider"]
        self.llm_provider.param.watch(self.handle_llm_provider_change, "value")
        self.handle_llm_provider_change(None)

        # Ticker
        self.ticker_select.value = ticker or defaults["tickerid"]
        self.update_tickers_tbl(None)
        self.only_sp500_tickers.param.watch(self.update_tickers_tbl, "value")

        self.fetch_data_btn.on_click(self.fetch_data)
        self.fetch_data(None)

        # Crews
        self.crew_select.param.watch(self.update_crew_config_markdown, "value")
        self.crew_select.options = defaults["crews"]
        self.crew_select.value = defaults["crew"]

        self.kickoff_crew_btn.on_click(self.generate_crew_output)

        # About tab
        with open("finmas/panel/about.md", mode="r", encoding="utf-8") as f:
            about = f.read()

        self.about_md = pn.pane.Markdown(about)

    def update_crew_config_markdown(self, event):
        """Update the crew configuration markdown"""
        config_path = Path(__file__).parent.parent / "crews" / self.crew_select.value / "config"
        self.crew_agents_config_md.object = get_yaml_config_as_markdown(config_path, "agents")
        self.crew_tasks_config_md.object = get_yaml_config_as_markdown(config_path, "tasks")

    def update_tickers_tbl(self, event):
        """Set the tickers table"""
        df = get_tickers_df(sp500=self.only_sp500_tickers.value)
        selection = df.index[df["ticker"] == self.ticker_select.value].tolist()
        if getattr(self, "tickers_tbl", None) is None:
            self.tickers_tbl = pn.widgets.Tabulator(
                df, on_click=self.handle_tickers_tbl_click, selection=selection, **tickers_config
            )
        else:
            self.tickers_tbl.value = df
        self.ticker_select.options = df["ticker"].tolist()

    def handle_tickers_tbl_click(self, event):
        """Callback for when a row in the tickers table is clicked"""
        tickerid = self.tickers_tbl.value.iloc[event.row]["ticker"]
        if self.ticker_select.value != tickerid:
            self.ticker_select.value = tickerid

    def handle_llm_provider_change(self, event):
        """
        Updates the LLM models table if the LLM provider have changed.
        Initializes the table with the Tabulator widget.
        The current LLM model is selected in the table.
        """
        df = get_valid_models(self.llm_provider.value)
        models = df["id"].tolist()
        self.llm_model.options = models
        self.llm_model.value = defaults[f"{self.llm_provider.value}_llm_model"]
        selection = df.index[df["id"] == self.llm_model.value].tolist()

        if getattr(self, "llm_models_tbl", None) is None:
            # Initialize the models table
            self.llm_models_tbl = pn.widgets.Tabulator(
                df,
                on_click=self.handle_llm_models_tbl_click,
                selection=selection,
                **llm_models_config,
            )
        else:
            self.llm_models_tbl.value = df
            self.llm_models_tbl.selection = selection

        # Update embedding models options
        prefix = "openai" if self.llm_provider.value == "openai" else "hf"
        self.embedding_model_news.options = defaults[f"{prefix}_embedding_models"]
        self.embedding_model_sec.options = defaults[f"{prefix}_embedding_models"]
        self.embedding_model_news.value = defaults[f"{prefix}_embedding_model_news"]
        self.embedding_model_sec.value = defaults[f"{prefix}_embedding_model_sec"]

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

            self.fetch_sec_filings(event)

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
        """Fetch price data from Yahoo Finance"""
        df = get_price_data(
            ticker=self.ticker_select.value,
            start=self.start_picker.value,
            end=self.end_picker.value,
        )
        df.reset_index(inplace=True)
        if df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        df.columns = df.columns.str.lower()

        if getattr(self, "ohlcv_tbl", None) is None:
            self.ohlcv_tbl = pn.widgets.Tabulator(df, **ohlcv_config)
        elif isinstance(self.ohlcv_tbl, pn.widgets.Tabulator):
            self.ohlcv_tbl.value = df

    def fetch_fundamental_data(self, event) -> None:
        """Fetch fundamental data"""
        income_statement = get_income_statement(
            ticker=self.ticker_select.value, freq=self.freq_select.value
        )

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
        self.news_records = news_fetcher.get_news(
            ticker=self.ticker_select.value,
            start=to_datetime(self.news_start.value),
            end=to_datetime(self.news_end.value),
        )

        df = pd.DataFrame.from_records(self.news_records)
        df["published"] = df["published"].dt.strftime("%Y-%m-%d")

        if getattr(self, "news_tbl", None) is None:
            self.news_tbl = pn.widgets.Tabulator(  # type: ignore
                df, row_content=self.get_news_content, **news_config
            )
        elif isinstance(self.news_tbl, pn.widgets.Tabulator):
            self.news_tbl.value = df

    def fetch_sec_filings(self, event) -> None:
        """Fetch SEC filings for the selected ticker"""
        self.sec_filings = get_sec_filings(
            self.ticker_select.value, filing_types=self.filing_types.value
        )
        df = filings_to_df(self.sec_filings)
        if getattr(self, "sec_filings_tbl", None) is None:
            self.sec_filings_tbl = pn.widgets.Tabulator(
                df, on_click=self.handle_sec_filings_tbl_click, **sec_filings_config
            )
        elif isinstance(self.sec_filings_tbl, pn.widgets.Tabulator):
            self.sec_filings_tbl.value = df
        self.sec_filings_tbl.selection = [0]
        self.sec_filing = self.sec_filings.get(df.iloc[0]["accession_number"])

    def handle_sec_filings_tbl_click(self, event):
        """Callback for when a row in SEC filings table is clicked"""
        self.sec_filing = self.sec_filings.get(
            self.sec_filings_tbl.value.iloc[event.row]["accession_number"]
        )

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

    def get_sec_filings_tbl(self, *args, **kwargs):
        """Get the SEC filings table"""
        if getattr(self, "sec_filings_tbl", None) is None:
            return pn.pane.Markdown("No SEC Filings data")
        return self.sec_filings_tbl

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

    def generate_crew_output(self, event) -> None:
        """
        Constructs the crew and kicks off the crew with defined inputs.
        Displays the output in Markdown.
        """
        with self.kickoff_crew_btn.param.update(loading=True):
            crew: Union[NewsAnalysisCrew, SECFilingCrew, SECFilingSectionsCrew]
            start = time.time()
            model_config = dict(
                ticker=self.ticker_select.value,
                llm_provider=self.llm_provider.value,
                llm_model=self.llm_model.value,
                temperature=self.llm_temperature.value,
                max_tokens=self.llm_max_tokens.value,
                similarity_top_k=self.similarity_top_k.value,
            )

            self.crew_usage_metrics.object = (
                "Loading embedding model and creating vector store index"
            )
            try:
                if self.crew_select.value == "news":
                    if getattr(self, "news_records", None) is None:
                        self.crew_output.object = "Need to fetch news data first."
                        return
                    crew = NewsAnalysisCrew(
                        records=self.news_records,
                        embedding_model=self.embedding_model_news.value,
                        news_source=self.news_source.value,
                        news_start=self.news_start.value,
                        news_end=self.news_end.value,
                        **model_config,
                    )
                elif self.crew_select.value == "sec":
                    crew = SECFilingCrew(
                        embedding_model=self.embedding_model_sec.value,
                        filing=self.sec_filing,
                        compress_filing=self.compress_sec_filing.value,
                        **model_config,
                    )
                elif self.crew_select.value == "sec_mda_risk_factors":
                    crew = SECFilingSectionsCrew(
                        embedding_model=self.embedding_model_sec.value,
                        filing=self.sec_filing,
                        **model_config,
                    )
            except Exception as e:
                self.crew_output_status.object = f"Error when setting up the crew: {str(e)}"
                self.crew_output_status.alert_type = "danger"
                return

            index_creation_metrics_message = ""
            for attr in dir(crew):
                if attr.endswith("index_creation_metrics"):
                    index_creation_metrics_message += (
                        f"{attr.replace('_', ' ').title()}:  \n{getattr(crew, attr).markdown()}\n\n"
                    )

            self.crew_usage_metrics.object = index_creation_metrics_message + "Started crew..."

            inputs = {"ticker": self.ticker_select.value, "form": self.sec_filing.form}
            try:
                output = crew.crew().kickoff(inputs=inputs)
            except Exception as e:
                self.crew_output_status.object = (
                    "The crew failed with the following error:  \n" + str(e)
                )
                self.crew_output_status.alert_type = "danger"
                return

            # Display the results
            time_spent = time.time() - start
            usage_metrics_string = get_usage_metrics_as_string(
                output.token_usage, self.llm_model.value
            )
            self.crew_usage_metrics.object = (
                index_creation_metrics_message
                + "Crew usage metrics:  \n"
                + usage_metrics_string
                + f"Time spent: {format_time_spent(time_spent)}"
            )

            crew_run_metrics = CrewRunMetrics(
                config=crew.config, token_usage=output.token_usage, time_spent=time_spent
            )
            file_path = save_crew_output(crew_run_metrics, output.raw)

            self.crew_output_status.object = f"Output stored in {str(file_path)}"
            self.crew_output_status.alert_type = "success"
            self.crew_output.object = output.raw

    def __panel__(self) -> Viewable:
        return pn.Row(
            pn.Column(
                pn.WidgetBox(
                    self.llm_provider,
                    self.llm_model,
                    self.embedding_model_news,
                    self.embedding_model_sec,
                    pn.Row(self.ticker_select, self.freq_select),
                    pn.Row(self.start_picker, self.end_picker),
                    self.news_source,
                    pn.Row(self.news_start, self.news_end),
                    self.include_fundamental_data,
                    self.include_news,
                    self.fetch_data_btn,
                ),
                pn.bind(self._data_alert_box, update_counter=self.update_counter),
                pn.WidgetBox("## Config", self.only_sp500_tickers, self.filing_types),
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
                                        width=600,
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
                                        width=600,
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
                    (
                        "Tickers",
                        pn.Column(
                            pn.pane.Markdown(
                                "Select a ticker. "
                                "Use the filters to explore and find the desired ticker.",
                                margin=0,
                            ),
                            self.tickers_tbl,
                        ),
                    ),
                    ("Models", pn.Column(self.llm_models_tbl)),
                    (
                        "SEC Filings",
                        pn.Column(
                            pn.bind(self.get_sec_filings_tbl, update_counter=self.update_counter)
                        ),
                    ),
                    (
                        "Crew Analysis",
                        pn.Row(
                            pn.Column(
                                pn.WidgetBox(
                                    pn.Row(
                                        self.crew_select,
                                        self.llm_temperature,
                                        self.llm_max_tokens,
                                        self.kickoff_crew_btn,
                                    ),
                                    pn.Row(self.similarity_top_k, self.compress_sec_filing),
                                ),
                                self.crew_usage_metrics,
                                pn.Column(
                                    pn.Card(
                                        self.crew_agents_config_md,
                                        margin=5,
                                        title="Agents",
                                    ),
                                    pn.Card(
                                        self.crew_tasks_config_md,
                                        margin=5,
                                        title="Tasks",
                                    ),
                                    width=600,
                                ),
                            ),
                            pn.Column(self.crew_output_status, self.crew_output),
                        ),
                    ),
                    ("About", pn.Column(self.about_md)),
                )
            ),
        )


if pn.state.served:
    import argparse
    import sys

    args_list = sys.argv[1:]
    args = argparse.Namespace()
    args.ticker = args_list[0] if args_list else None
    FinMAS(ticker=args.ticker).servable(title="FinMAS - Financial Multi-Agent System")
