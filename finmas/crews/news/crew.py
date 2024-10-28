from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import LlamaIndexTool

from finmas.crews.model_provider import get_crewai_llm_model
from finmas.crews.news.tools import get_news_query_engine


@CrewBase
class NewsAnalysisCrew:
    """News Analysis crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(
        self,
        records: list[dict],
        llm_provider: str,
        llm_model: str,
        embedding_model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        similarity_top_k: int | None = None,
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
