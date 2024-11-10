from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import LlamaIndexTool

from finmas.crews.model_provider import get_crewai_llm_model
from finmas.data.sec.query_engine import get_sec_query_engine
from edgar import Filing
from finmas.constants import defaults
from finmas.crews.utils import SECCrewConfiguration


@CrewBase
class SECFilingCrew:
    """SEC Filing Crew for analyzing SEC filings with LLM"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    name = "sec"

    def __init__(
        self,
        ticker: str,
        llm_provider: str,
        llm_model: str,
        embedding_model: str,
        filing: Filing,
        compress_filing: bool = False,
        temperature: float = defaults["llm_temperature"],
        max_tokens: int = defaults["llm_max_tokens"],
        similarity_top_k: int = defaults["similarity_top_k"],
    ):
        self.crewai_llm = get_crewai_llm_model(
            llm_provider, llm_model, temperature=temperature, max_tokens=max_tokens
        )
        self.sec_query_engine, self.index_creation_metrics = get_sec_query_engine(
            ticker,
            llm_provider,
            llm_model,
            embedding_model,
            filing=filing,
            method="semantic_search_keywords" if compress_filing else "full_text",
            temperature=temperature,
            max_tokens=max_tokens,
            similarity_top_k=similarity_top_k,
        )
        self.sec_tool = LlamaIndexTool.from_query_engine(
            self.sec_query_engine,
            name=f"{filing.form} SEC Filing Query Tool for {ticker}",
            description=f"Use this tool to search and analyze the the {filing.form} SEC filing",
        )
        self.config = SECCrewConfiguration(
            name="sec",
            ticker=ticker,
            llm_provider=llm_provider,
            llm_model=llm_model,
            embedding_model=embedding_model,
            llm_temperature=temperature,
            llm_max_tokens=max_tokens,
            similarity_top_k=similarity_top_k,
            form_type=filing.form,
            filing_date=filing.filing_date,
        )
        super().__init__()

    @agent
    def sec_filing_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config["sec_filing_analyzer"],  # type: ignore
            verbose=True,
            memory=True,
            llm=self.crewai_llm,
            tools=[self.sec_tool],
        )

    @agent
    def sec_filing_sentiment_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config["sec_filing_sentiment_analyzer"],  # type: ignore
            verbose=True,
            memory=True,
            llm=self.crewai_llm,
            tools=[self.sec_tool],
        )

    @agent
    def sec_filing_summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config["sec_filing_summarizer"],  # type: ignore
            verbose=True,
            memory=True,
            llm=self.crewai_llm,
        )

    @task
    def sec_filing_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["sec_filing_analysis_task"],  # type: ignore
            async_execution=True,
        )

    @task
    def sec_filing_sentiment_task(self) -> Task:
        return Task(
            config=self.tasks_config["sec_filing_sentiment_task"],  # type: ignore
            async_execution=True,
        )

    @task
    def sec_filing_summary_task(self) -> Task:
        return Task(
            config=self.tasks_config["sec_filing_summary_task"],  # type: ignore
            context=[self.sec_filing_analysis_task(), self.sec_filing_sentiment_task()],
        )

    @crew
    def crew(self) -> Crew:
        """Creates SEC Filing Analysis crew"""
        return Crew(
            agents=self.agents,  # type: ignore
            tasks=self.tasks,  # type: ignore
            cache=True,
            process=Process.sequential,
            verbose=True,
            planning=True,
            output_log_file="sec_crew.log",
        )
