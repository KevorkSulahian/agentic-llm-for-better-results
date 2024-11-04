from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import LlamaIndexTool

from finmas.crews.model_provider import get_crewai_llm_model
from finmas.data.news.query_engine import get_news_query_engine
from finmas.crews.utils import NewsCrewConfiguration
import datetime as dt
from finmas.constants import defaults


@CrewBase
class NewsAnalysisCrew:
    """News Analysis crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(
        self,
        ticker: str,
        records: list[dict],
        llm_provider: str,
        llm_model: str,
        embedding_model: str,
        news_source: str,
        news_start: dt.date,
        news_end: dt.date,
        temperature: float = defaults["llm_temperature"],
        max_tokens: int = defaults["llm_max_tokens"],
        similarity_top_k: int = defaults["similarity_top_k"],
    ):
        self.crewai_llm = get_crewai_llm_model(
            llm_provider, llm_model, temperature=temperature, max_tokens=max_tokens
        )
        self.news_query_engine, self.index_creation_metrics = get_news_query_engine(
            records,
            llm_provider,
            llm_model,
            embedding_model,
            temperature=temperature,
            max_tokens=max_tokens,
            similarity_top_k=similarity_top_k,
        )
        self.llama_index_news_tool = LlamaIndexTool.from_query_engine(
            self.news_query_engine,
            name="News Query Tool",
            description="Use this tool to lookup the latest news",
        )
        self.config = NewsCrewConfiguration(
            name="news",
            ticker=ticker,
            llm_provider=llm_provider,
            llm_model=llm_model,
            embedding_model=embedding_model,
            llm_temperature=temperature,
            llm_max_tokens=max_tokens,
            similarity_top_k=similarity_top_k,
            news_source=news_source,
            news_start=news_start,
            news_end=news_end,
        )
        super().__init__()

    @agent
    def news_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config["news_analyzer"],  # type: ignore
            verbose=True,
            memory=True,  # helpful for smaller llm in case they fail -> won't repeat the same thing twice
            llm=self.crewai_llm,
            tools=[self.llama_index_news_tool],
        )

    @agent
    def sentiment_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config["sentiment_analyzer"],  # type: ignore
            verbose=True,
            memory=True,
            llm=self.crewai_llm,
            tools=[self.llama_index_news_tool],
        )

    @agent
    def news_summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config["news_summarizer"],  # type: ignore
            verbose=True,
            memory=True,
            llm=self.crewai_llm,
        )

    @task
    def news_analyzer_task(self) -> Task:
        return Task(
            config=self.tasks_config["news_analyzer_task"],  # type: ignore
        )

    @task
    def sentiment_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["sentiment_analysis_task"],  # type: ignore
        )

    @task
    def news_summary_task(self) -> Task:
        return Task(
            config=self.tasks_config["news_summary_task"],  # type: ignore
        )

    @crew
    def crew(self) -> Crew:
        """Creates News Analysis crew"""
        return Crew(
            agents=self.agents,  # type: ignore
            tasks=self.tasks,  # type: ignore
            cache=True,
            process=Process.sequential,
            verbose=True,
            planning=True,
            output_log_file="news_crew.log",
        )
