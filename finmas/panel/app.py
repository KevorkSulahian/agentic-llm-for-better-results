import os
import time
import traceback
from pathlib import Path
from typing import Union

import hvplot.pandas  # noqa: F401
import pandas as pd
import panel as pn
import plotly.graph_objects as go
from crewai import Crew
from dotenv import find_dotenv, load_dotenv
from panel.viewable import Viewable

from finmas.constants import DOCS_URL, defaults
from finmas.crews.combined.crew import CombinedCrew
from finmas.crews.market_data.crew import MarketDataCrew
from finmas.crews.news.crew import NewsAnalysisCrew
from finmas.crews.sec.crew import SECFilingCrew
from finmas.crews.sec_mda_risk_factors.crew import SECFilingSectionsCrew
from finmas.crews.utils import (
    CrewRunMetrics,
    get_usage_metrics_as_string,
    get_yaml_config_as_markdown,
    save_crew_output,
)
from finmas.data.market.fundamentals import NUM_QUARTERS, get_ticker_essentials
from finmas.data.market.technical_analysis import get_technical_indicators
from finmas.data.news import get_news_fetcher
from finmas.data.sec.filings import filings_to_df, get_sec_filings
from finmas.logger import get_logger
from finmas.panel.formatting import (
    INCOME_STATEMENT_COLS_MAP,
    embedding_models_config,
    income_statement_config,
    llm_models_config,
    news_config,
    sec_filings_config,
    ta_config,
    ta_styler,
    tickers_config,
)
from finmas.panel.plotting import get_income_statment_plot_figure, get_ta_plot_figure
from finmas.utils.common import (
    format_time_spent,
    get_embedding_models_df,
    get_llm_models_df,
    get_tickers_df,
    to_datetime,
)

hvplot.extension("plotly")
pn.extension(
    "tabulator",
    "plotly",
    template="fast",
    css_files=["https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css"],
)

load_dotenv(find_dotenv())

logger = get_logger(__name__)


class FinMAS(pn.viewable.Viewer):
    llm_selected = pn.widgets.StaticText(name="LLM", value="")
    llm_provider = pn.widgets.Select(
        name="LLM Provider",
        options=defaults["llm_providers"],
        width=100,
    )
    llm_model = pn.widgets.Select(name="LLM Model", width=200, disabled=True)
    embedding_model = pn.widgets.Select(
        name="Embedding Model for Text",
        value=defaults["hf_embedding_model"],
        options=defaults["hf_embedding_models"],
        width=250,
    )
    ticker_select = pn.widgets.StaticText(name="Ticker")
    news_source = pn.widgets.Select(
        name="News Source",
        value=defaults["news_source"],
        options=defaults["news_sources"],
        width=100,
    )
    news_start = pn.widgets.DatetimeInput(
        name="Start",
        format="%Y-%m-%d",
        value=defaults["news_start_date"],
        width=100,
    )
    news_end = pn.widgets.DatetimeInput(
        name="End",
        format="%Y-%m-%d",
        value=defaults["news_end_date"],
        width=100,
    )
    include_fundamental_data = pn.widgets.Checkbox(
        name="Include Fundamental Data", value=defaults["include_fundamental_data"]
    )
    include_news = pn.widgets.Checkbox(name="Include News", value=defaults["include_news"])
    only_sp500_tickers = pn.widgets.Checkbox(
        name="SP500 Tickers", value=defaults["only_sp500_tickers"]
    )

    update_counter = pn.widgets.IntInput(value=0)

    fetch_data_btn = pn.widgets.Button(name="Fetch data", button_type="primary")
    time_elapsed: dict[str, float] = {}

    crew_select = pn.widgets.Select(name="Crew", width=250)
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

    sec_filing_selected = pn.widgets.StaticText(name="SEC Filing", align=("center"))

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
        self.crew_select.param.watch(self.handle_crew_select_change, "value")
        self.crew_select.options = defaults["crews"]
        self.crew_select.value = defaults["crew"]

        self.kickoff_crew_btn.on_click(self.generate_crew_output)

        # Embedding models
        self.embedding_models_tbl = pn.widgets.Tabulator(
            get_embedding_models_df(), **embedding_models_config
        )

        # About tab
        with open("finmas/panel/about.md", mode="r", encoding="utf-8") as f:
            about = f.read()

        self.about_md = pn.pane.Markdown(about)

    def handle_crew_select_change(self, event):
        """Handle the change of the crew select"""
        self.similarity_top_k.disabled = False
        self.compress_sec_filing.disabled = True
        self.sec_filing_selected.visible = True

        if self.crew_select.value == "market_data":
            self.similarity_top_k.disabled = True
        if self.crew_select.value == "sec":
            self.compress_sec_filing.disabled = False
        if self.crew_select.value in ["news", "market_data"]:
            self.sec_filing_selected.visible = False

        # Update the crew configuration markdown
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

        if getattr(self, "llm_models_tbl", None) is None:
            # Initialize the models table
            df = get_llm_models_df(self.llm_provider.options)
            self.llm_models_tbl = pn.widgets.Tabulator(
                df,
                on_click=self.handle_llm_models_tbl_click,
                **llm_models_config,
            )
            self.llm_model.options = df.loc[
                df["provider"] == self.llm_provider.value, "id"
            ].tolist()
            self.llm_model.value = defaults[f"{self.llm_provider.value}_llm_model"]
            self.llm_selected.value = f"{self.llm_provider.value} / {self.llm_model.value}"

        # Update embedding models options
        prefix = "openai" if self.llm_provider.value == "openai" else "hf"
        self.embedding_model.options = defaults[f"{prefix}_embedding_models"]
        self.embedding_model.value = defaults[f"{prefix}_embedding_model"]

    def handle_llm_models_tbl_click(self, event):
        """Callback for when a row in LLM models table is clicked"""
        llm_model_id = self.llm_models_tbl.value.iloc[event.row]["id"]
        llm_provider = self.llm_models_tbl.value.iloc[event.row]["provider"]
        if self.llm_provider.value != llm_provider:
            self.llm_provider.value = llm_provider
        if self.llm_model.value != llm_model_id:
            self.llm_model.value = llm_model_id
        self.llm_selected.value = f"{self.llm_provider.value} / {self.llm_model.value}"

    def fetch_data(self, event) -> None:
        """Main handler for fetching data for the app"""
        with self.fetch_data_btn.param.update(loading=True):
            start = time.time()
            self.fetch_technical_analysis_data(event)

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
        if getattr(self, "news_records", None):
            message += f"\nFetched {len(self.news_records)} news articles."
        return pn.pane.Alert(message, alert_type=alert_type)

    def fetch_technical_analysis_data(self, event) -> None:
        """Fetch price data from Yahoo Finance"""
        df = get_technical_indicators(ticker=self.ticker_select.value, period="5y")
        df = df.reset_index(names="date")

        if getattr(self, "ta_tbl", None) is None:
            self.ta_tbl = pn.widgets.Tabulator(df.style.pipe(ta_styler), **ta_config)
        elif isinstance(self.ta_tbl, pn.widgets.Tabulator):
            self.ta_tbl.value = df

    def fetch_fundamental_data(self, event) -> None:
        """Fetch fundamental data"""
        df: pd.DataFrame = get_ticker_essentials(ticker=self.ticker_select.value)
        df = df.tail(NUM_QUARTERS * 2)
        df = df.dropna(axis=0, how="any")
        assert isinstance(df.index, pd.DatetimeIndex)
        df.index.name = "date"
        df = df[list(INCOME_STATEMENT_COLS_MAP.keys())].copy()
        df.sort_index(ascending=False, inplace=True)

        if getattr(self, "income_statement_tbl", None) is None:
            self.income_statement_tbl = pn.widgets.Tabulator(  # type: ignore
                df, **income_statement_config
            )
        elif isinstance(self.income_statement_tbl, pn.widgets.Tabulator):
            self.income_statement_tbl.value = df.copy()

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
        self.sec_filing_selected.value = (
            f"{self.sec_filing.form} - Report Date: {df.iloc[0]['reportDate']}"
        )

    def handle_sec_filings_tbl_click(self, event):
        """Callback for when a row in SEC filings table is clicked"""
        self.sec_filing = self.sec_filings.get(
            self.sec_filings_tbl.value.iloc[event.row]["accession_number"]
        )
        report_date = self.sec_filings_tbl.value.iloc[event.row]["reportDate"]
        self.sec_filing_selected.value = f"{self.sec_filing.form} - Report Date: {report_date}"

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
        if self.ta_tbl is None or self.ta_tbl.value.empty:
            return pn.pane.Markdown("No Technical Analysis data")
        df = self.ta_tbl.value

        return pn.pane.Plotly(get_ta_plot_figure(df))

    def get_ta_plot_title(self, *args, **kwargs) -> str:
        """Get the title for the Technical analysis plot"""
        return f"{self.ticker_select.value} - Technical Analysis (Weekly Timeframe)"

    def get_income_statement_plot(self, *args, **kwargs) -> go.Figure:
        """Plot income statement table"""
        if (
            getattr(self, "income_statement_tbl", None) is None
            or self.income_statement_tbl.value.empty  # type: ignore
        ):
            return pn.pane.Markdown("No Fundamental data")
        assert isinstance(self.income_statement_tbl, pn.widgets.Tabulator)
        df = self.income_statement_tbl.value.copy()
        df = df.sort_index()
        df = df.tail(NUM_QUARTERS)

        assert isinstance(df.index, pd.DatetimeIndex)
        df.index = df.index.to_period("Q").strftime("%YQ%q")

        return pn.pane.Plotly(get_income_statment_plot_figure(df))

    def generate_crew_output(self, event) -> None:
        """
        Constructs the crew and kicks off the crew with defined inputs.
        Displays the output in Markdown.
        """
        with self.kickoff_crew_btn.param.update(loading=True):
            if (
                self.crew_select.value in ["news", "combined"]
                and getattr(self, "news_records", None) is None
            ):
                self.crew_output.object = "Need to fetch news data first."
                return

            crew: Union[
                NewsAnalysisCrew, SECFilingCrew, SECFilingSectionsCrew, MarketDataCrew, CombinedCrew
            ]
            start = time.time()
            model_config = dict(
                ticker=self.ticker_select.value,
                llm_provider=self.llm_provider.value,
                llm_model=self.llm_model.value,
                temperature=self.llm_temperature.value,
                max_tokens=self.llm_max_tokens.value,
            )

            if self.crew_select.value != "market_data":
                self.crew_usage_metrics.object = (
                    "Loading embedding model and creating vector store index"
                )
            try:
                if self.crew_select.value == "news":
                    crew = NewsAnalysisCrew(
                        records=self.news_records,
                        embedding_model=self.embedding_model.value,
                        news_source=self.news_source.value,
                        news_start=self.news_start.value,
                        news_end=self.news_end.value,
                        similarity_top_k=self.similarity_top_k.value,
                        **model_config,
                    )
                elif self.crew_select.value == "sec":
                    crew = SECFilingCrew(
                        embedding_model=self.embedding_model.value,
                        filing=self.sec_filing,
                        compress_filing=self.compress_sec_filing.value,
                        similarity_top_k=self.similarity_top_k.value,
                        **model_config,
                    )
                elif self.crew_select.value == "sec_mda_risk_factors":
                    crew = SECFilingSectionsCrew(
                        embedding_model=self.embedding_model.value,
                        filing=self.sec_filing,
                        similarity_top_k=self.similarity_top_k.value,
                        **model_config,
                    )
                elif self.crew_select.value == "market_data":
                    crew = MarketDataCrew(**model_config)
                elif self.crew_select.value == "combined":
                    crew = CombinedCrew(
                        records=self.news_records,
                        embedding_model=self.embedding_model.value,
                        news_source=self.news_source.value,
                        news_start=self.news_start.value,
                        news_end=self.news_end.value,
                        filing=self.sec_filing,
                        similarity_top_k=self.similarity_top_k.value,
                        **model_config,
                    )
            except Exception as e:
                error_message = (
                    f"Error when setting up the crew: {str(e)}\n\n{traceback.format_exc()}"
                )
                logger.error(error_message)
                self.crew_output_status.object = error_message
                self.crew_output_status.alert_type = "danger"
                return

            index_creation_metrics_message = ""
            for attr in dir(crew):
                if attr.endswith("index_creation_metrics"):
                    index_creation_metrics_message += (
                        f"{attr.replace('_', ' ').title()}:  \n{getattr(crew, attr).markdown()}\n\n"
                    )

            self.crew_usage_metrics.object = index_creation_metrics_message + "Started crew..."

            inputs = {"ticker": self.ticker_select.value}
            if crew.name not in ["market_data", "news"]:
                inputs["form"] = self.sec_filing.form
            try:
                crew_ready: Crew = crew.crew()
                output = crew_ready.kickoff(inputs=inputs)
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
            file_path = save_crew_output(
                crew_run_metrics, output.raw, index_creation_metrics_message
            )

            self.crew_output_status.object = f"Output stored in {str(file_path)}"
            self.crew_output_status.alert_type = "success"
            self.crew_output.object = output.raw
            logger.info(f"Crew finished successfully and output stored in {str(file_path)}")

    def __panel__(self) -> Viewable:
        return pn.Row(
            pn.Column(
                pn.WidgetBox(
                    self.ticker_select,
                    self.llm_selected,
                    self.embedding_model,
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
                                        width=700,
                                        height=400,
                                        margin=10,
                                        title=pn.bind(
                                            self.get_ta_plot_title,
                                            update_counter=self.update_counter,
                                        ),
                                    ),
                                    self.ta_tbl,
                                ),
                                pn.Column(
                                    pn.Card(
                                        pn.bind(
                                            self.get_income_statement_plot,
                                            update_counter=self.update_counter,
                                        ),
                                        width=700,
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
                            pn.pane.HTML("<b>News Articles</b>"),
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
                    (
                        "Models",
                        pn.Row(
                            pn.Column(pn.pane.Markdown("### LLM Models"), self.llm_models_tbl),
                            pn.Column(
                                pn.pane.Markdown("### Embedding Models"), self.embedding_models_tbl
                            ),
                        ),
                    ),
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
                                    pn.Row(self.crew_select, self.sec_filing_selected),
                                    pn.Row(
                                        self.llm_temperature,
                                        self.llm_max_tokens,
                                        self.similarity_top_k,
                                        self.kickoff_crew_btn,
                                    ),
                                    pn.Row(self.compress_sec_filing),
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

    menu_html = f"""
        <a href="{DOCS_URL}" class="button-style">Docs</a>
        """
    app_menu = pn.Row(pn.pane.HTML(menu_html, align="end"))
    app_menu.servable(area="header")
