from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from finmas.constants import agent_config, defaults
from finmas.crews.model_provider import get_crewai_llm_model
from finmas.crews.utils import CrewConfiguration, get_log_filename
from finmas.data.market import StockFundamentalsTool, TechnicalAnalysisTool


@CrewBase
class MarketDataCrew:
    """Crew that analyzes a stock using:

    - Fundamental data
    - Stock price data with technical indicators

    The final output is a recommendation on the stock.
    """

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    name = "market_data"

    def __init__(
        self,
        ticker: str,
        llm_provider: str,
        llm_model: str,
        temperature: float = defaults["llm_temperature"],
        max_tokens: int = defaults["llm_max_tokens"],
    ):
        self.crewai_llm = get_crewai_llm_model(
            llm_provider, llm_model, temperature=temperature, max_tokens=max_tokens
        )
        self.stock_fundamentals_tool = StockFundamentalsTool()
        self.technical_analysis_tool = TechnicalAnalysisTool()

        self.config = CrewConfiguration(
            name=self.name,
            ticker=ticker,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_temperature=temperature,
            llm_max_tokens=max_tokens,
            similarity_top_k=None,
            embedding_model=None,
        )
        super().__init__()

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
    def fundamental_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["fundamental_analysis"],  # type: ignore
            async_execution=True,
        )

    @task
    def technical_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["technical_analysis"],  # type: ignore
            async_execution=True,
        )

    @task
    def stock_advisor_task(self) -> Task:
        return Task(
            config=self.tasks_config["stock_advisor_task"],  # type: ignore
            context=[self.fundamental_analysis(), self.technical_analysis()],
        )

    @crew
    def crew(self) -> Crew:
        """Creates Market Data Analysis crew"""
        return Crew(
            agents=self.agents,  # type: ignore
            tasks=[
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
