from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from finmas.crews.model_provider import get_crewai_llm_model
from finmas.crews.sec.tools import get_sec_tool


@CrewBase
class SECFilingCrew:
    """SEC Filing Crew for analyzing SEC filings with LLM"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(
        self,
        ticker: str,
        llm_provider: str,
        llm_model: str,
        embedding_model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self.crewai_llm = get_crewai_llm_model(
            llm_provider, llm_model, temperature=temperature, max_tokens=max_tokens
        )
        self.sec_tool = get_sec_tool(
            ticker,
            llm_provider,
            llm_model,
            embedding_model,
            temperature=temperature,
            max_tokens=max_tokens,
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
        )

    @task
    def sec_filing_sentiment_task(self) -> Task:
        return Task(
            config=self.tasks_config["sec_filing_sentiment_task"],  # type: ignore
        )

    @task
    def sec_filing_summary_task(self) -> Task:
        return Task(
            config=self.tasks_config["sec_filing_summary_task"],  # type: ignore
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
