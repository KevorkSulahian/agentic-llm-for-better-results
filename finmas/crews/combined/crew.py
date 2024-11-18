import datetime as dt
import time

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import LlamaIndexTool
from edgar import Filing

from finmas.constants import agent_config, defaults
from finmas.crews.model_provider import get_crewai_llm_model
from finmas.crews.utils import CombinedCrewConfiguration, get_log_filename
from finmas.data.market import StockFundamentalsTool, TechnicalAnalysisTool
from finmas.data.news.query_engine import get_news_query_engine
from finmas.data.sec.query_engine import get_sec_query_engine
from finmas.logger import get_logger

logger = get_logger(__name__)


@CrewBase
class CombinedCrew:
    """Stock Analysis Crew that analyze a stock using:

    - Recent news
    - SEC filing
    - Fundamental data
    - Stock price data with technical indicators

    The final output is a recommendation on the stock.
    """

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    name = "combined"

    def __init__(
        self,
        *,
        ticker: str,
        records: list[dict],
        llm_provider: str,
        llm_model: str,
        embedding_model: str,
        news_source: str,
        news_start: dt.date,
        news_end: dt.date,
        filing: Filing,
        temperature: float = defaults["llm_temperature"],
        max_tokens: int = defaults["llm_max_tokens"],
        similarity_top_k: int = defaults["similarity_top_k"],
        async_execution: bool = True,
        price_end_date: dt.date | None = None,
    ):
        start = time.time()
        self.crewai_llm = get_crewai_llm_model(
            llm_provider, llm_model, temperature=temperature, max_tokens=max_tokens
        )
        # News
        self.news_query_engine, self.news_index_creation_metrics = get_news_query_engine(
            ticker,
            records,
            llm_provider,
            llm_model,
            embedding_model,
            temperature=temperature,
            max_tokens=max_tokens,
            similarity_top_k=similarity_top_k,
        )
        self.news_tool = LlamaIndexTool.from_query_engine(
            self.news_query_engine,
            name=f"News Query Tool for {ticker}",
            description=f"Use this tool to lookup the latest news for {ticker}",
        )

        # SEC Filing
        self.sec_query_engine, self.sec_index_creation_metrics = get_sec_query_engine(
            ticker,
            llm_provider,
            llm_model,
            embedding_model,
            filing=filing,
            method="section:mda",
            temperature=temperature,
            max_tokens=max_tokens,
            similarity_top_k=similarity_top_k,
        )
        self.sec_tool = LlamaIndexTool.from_query_engine(
            self.sec_query_engine,
            name=f"{filing.form} SEC Filing Query Tool for {ticker}",
            description=f"Use this tool to search and analyze the the {filing.form} SEC filing",
        )

        # Market Data
        self.stock_fundamentals_tool = StockFundamentalsTool()
        self.technical_analysis_tool = TechnicalAnalysisTool(end_date=price_end_date)

        self.config = CombinedCrewConfiguration(
            name=self.name,
            ticker=ticker,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_temperature=temperature,
            llm_max_tokens=max_tokens,
            similarity_top_k=similarity_top_k,
            embedding_model=embedding_model,
            news_source=news_source,
            news_start=news_start,
            news_end=news_end,
            form_type=filing.form,
            filing_date=filing.filing_date,
        )
        self.async_execution = async_execution
        super().__init__()
        logger.info(f"Combined Crew initialized in {round(time.time() - start, 2)}s")

    @agent
    def news_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["news_analyst"],  # type: ignore
            verbose=True,
            memory=True,
            llm=self.crewai_llm,
            tools=[self.news_tool],
            **agent_config,
        )

    @agent
    def sec_filing_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["sec_filing_analyst"],  # type: ignore
            verbose=True,
            memory=True,
            llm=self.crewai_llm,
            tools=[self.sec_tool],
            **agent_config,
        )

    @agent
    def fundamental_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["fundamental_analyst"],  # type: ignore
            verbose=True,
            memory=True,
            llm=self.crewai_llm,
            tools=[self.stock_fundamentals_tool],
            **agent_config,
        )

    @agent
    def technical_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["technical_analyst"],  # type: ignore
            verbose=True,
            memory=True,
            llm=self.crewai_llm,
            tools=[self.technical_analysis_tool],
            **agent_config,
        )

    @agent
    def stock_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config["stock_advisor"],  # type: ignore
            verbose=True,
            memory=True,
            llm=self.crewai_llm,
            **agent_config,
        )

    @task
    def news_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["news_analysis"],  # type: ignore
            async_execution=self.async_execution,
        )

    @task
    def sec_filing_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["sec_filing_analysis"],  # type: ignore
            async_execution=self.async_execution,
        )

    @task
    def fundamental_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["fundamental_analysis"],  # type: ignore
            async_execution=self.async_execution,
        )

    @task
    def technical_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["technical_analysis"],  # type: ignore
            async_execution=self.async_execution,
        )

    @task
    def stock_advisor_task(self) -> Task:
        return Task(
            config=self.tasks_config["stock_advisor_task"],  # type: ignore
            context=[
                self.news_analysis(),
                self.sec_filing_analysis(),
                self.fundamental_analysis(),
                self.technical_analysis(),
            ],
        )

    @crew
    def crew(self) -> Crew:
        """Creates Combined Analysis crew"""
        return Crew(
            agents=self.agents,  # type: ignore
            tasks=[
                self.news_analysis(),
                self.sec_filing_analysis(),
                self.fundamental_analysis(),
                self.technical_analysis(),
                self.stock_advisor_task(),
            ],
            cache=True,
            process=Process.sequential,
            verbose=True,
            planning=True,
            output_log_file=get_log_filename(self.name),
        )
