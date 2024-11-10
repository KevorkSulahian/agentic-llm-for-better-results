import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import yaml
from crewai.types.usage_metrics import UsageMetrics

from finmas.constants import agent_config, defaults
from finmas.utils.common import format_time_spent


@dataclass
class IndexCreationMetrics:
    embedding_model: str
    time_spent: float
    num_nodes: int
    text_length: int
    chunk_size: int
    chunk_overlap: int
    total_embedding_token_count: int

    def markdown(self) -> str:
        output = (
            f"Embedding Model: {self.embedding_model}  \n"
            f"Time spent: {format_time_spent(self.time_spent)}  \n"
            f"Number of nodes: {self.num_nodes}  \n"
            f"Text length: {self.text_length}  \n"
            f"Chunk size: {self.chunk_size} tokens  \n"
            f"Chunk overlap: {self.chunk_overlap} tokens  \n"
            f"Total embedding token count: {self.total_embedding_token_count}  \n"
        )
        if self.embedding_model in defaults["embedding_model_cost"]:
            cost = (
                defaults["embedding_model_cost"][self.embedding_model]
                * self.total_embedding_token_count
            )
            cost_str = f"${cost:.15f}".rstrip("0")
            output += f"Estimated embedding model cost for total tokens: {cost_str}  \n"
        return output


@dataclass
class CrewConfiguration:
    name: str
    ticker: str
    llm_provider: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    similarity_top_k: int | None
    embedding_model: str | None

    def markdown(self, crew_description: bool = False) -> str:
        output = (
            "## Configuration:  \n\n"
            f"Crew Name: {self.name}  \n"
            f"Ticker: {self.ticker}  \n"
            f"LLM: {self.llm_provider} / {self.llm_model}  \n"
            f"Temperature: {self.llm_temperature} Max tokens: {self.llm_max_tokens}  \n"
            "Agent Configuration:  \n"
            f"Max iterations: {agent_config['max_iter']} Max requests per minute: {agent_config['max_rpm']}  \n"
        )
        if self.similarity_top_k and self.embedding_model:
            output += f"Embedding Model: {self.embedding_model} similarity_top_k: {self.similarity_top_k}  \n"
        if crew_description:
            config_path = Path(__file__).parent / self.name / "config"
            output += (
                "\n## Agents\n\n"
                + get_yaml_config_as_markdown(config_path, "agents")
                + "## Tasks\n\n"
                + get_yaml_config_as_markdown(config_path, "tasks")
            )
        return output


@dataclass
class NewsCrewConfiguration(CrewConfiguration):
    news_source: str
    news_start: dt.date
    news_end: dt.date

    def markdown(self, crew_description: bool = False) -> str:
        return (
            super().markdown(crew_description)
            + f"News Source: {self.news_source}\n"
            + f"Date range: {self.news_start} - {self.news_end}\n"
        )


@dataclass
class SECCrewConfiguration(CrewConfiguration):
    form_type: str


@dataclass
class CrewRunMetrics:
    config: CrewConfiguration
    token_usage: UsageMetrics
    time_spent: float

    def markdown(self, crew_description: bool = False) -> str:
        return (
            self.config.markdown(crew_description)
            + "\n"
            + "## Crew Run Metrics\n\n"
            + get_usage_metrics_as_string(self.token_usage, self.config.llm_model)
            + "\n"
            + f"Time spent: {format_time_spent(self.time_spent)}"
        )


def get_yaml_config_as_markdown(config_path: Path, config_file: str):
    """Returns a markdown representation of the yaml configuration file."""
    with open(config_path / f"{config_file}.yaml", "r") as c:
        config = yaml.safe_load(c)

    output = ""
    for key, value in config.items():
        output += f"### {key.replace('_', ' ').title()}\n\n"
        for field, specification in value.items():
            output += f"- **{field.replace('_', ' ').title()}**: {specification}"
        output += "\n"

    return output


def get_usage_metrics_as_string(usage_metrics: UsageMetrics, llm_model: str | None = None) -> str:
    """Returns a string representation of the usage metrics."""
    output = (
        f"Total tokens: {usage_metrics.total_tokens} "
        f"Prompt tokens: {usage_metrics.prompt_tokens}  \n"
        f"Successful Requests: {usage_metrics.successful_requests}  \n"
    )
    if llm_model and llm_model in defaults["llm_model_cost"]:
        cost = (
            defaults["llm_model_cost"][llm_model]["input_cost"] * usage_metrics.prompt_tokens
            + defaults["llm_model_cost"][llm_model]["output_cost"] * usage_metrics.completion_tokens
        )
        cost_str = f"{cost:.15f}".rstrip("0")
        output += f"Estimated LLM Model cost for total tokens: ${cost_str}  \n"
    return output


def save_crew_output(crew_run_metrics: CrewRunMetrics, output_content: str) -> Path:
    """Saves crew output together with metadata from the crew run.

    Args:
        crew_run_metrics: Metadata from the crew run.
        output_content: Crew output content.
    """
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    config = crew_run_metrics.config
    filename = f"{config.ticker}_{config.name}_analysis_{timestamp}.md"
    output_dir = Path(defaults["crew_output_dir"]) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / filename
    file_path.write_text(
        crew_run_metrics.markdown(crew_description=True)
        + "\n\n## Crew output:\n\n"
        + output_content,
        encoding="utf-8",
    )
    return file_path


def get_log_filename(crew_name: str):
    """
    Returns the log file path for the crew.
    If the folder does not exist, it will be created.
    """
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    file_path = f"{defaults['crew_logs_dir']}/{crew_name}/crew_{timestamp}.log"
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    Path(file_path).touch()
    return str(file_path)
