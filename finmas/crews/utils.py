import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import yaml
from crewai.types.usage_metrics import UsageMetrics

from finmas.constants import defaults
from finmas.utils.common import format_time_spent


@dataclass
class IndexCreationMetrics:
    time_spent: float
    num_nodes: int
    text_length: int
    chunk_size: int
    chunk_overlap: int

    def markdown(self) -> str:
        return (
            f"Time spent: {format_time_spent(self.time_spent)}  \n"
            f"Number of nodes: {self.num_nodes}  \n"
            f"Text length: {self.text_length}  \n"
            f"Chunk size: {self.chunk_size} tokens  \n"
            f"Chunk overlap: {self.chunk_overlap} tokens"
        )


@dataclass
class CrewConfiguration:
    name: str
    ticker: str
    llm_provider: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    similarity_top_k: int
    embedding_model: str

    def markdown(self) -> str:
        return (
            "Configuration:  \n\n"
            f"Crew Name: {self.name}  \n"
            f"LLM: {self.llm_provider} / {self.llm_model}  \n"
            f"Temperature: {self.llm_temperature} Max tokens: {self.llm_max_tokens}  \n"
            f"Embedding Model: {self.embedding_model} similarity_top_k: {self.similarity_top_k}  \n"
            f"Ticker: {self.ticker}  \n"
        )


@dataclass
class NewsCrewConfiguration(CrewConfiguration):
    news_source: str
    news_start: dt.date
    news_end: dt.date

    def markdown(self) -> str:
        return (
            super().markdown()
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

    def markdown(self) -> str:
        return (
            self.config.markdown()
            + "\n"
            + get_usage_metrics_as_string(self.token_usage)
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


def get_usage_metrics_as_string(usage_metrics: UsageMetrics) -> str:
    """Returns a string representation of the usage metrics."""
    output = (
        f"Total tokens: {usage_metrics.total_tokens} "
        f"Prompt tokens: {usage_metrics.prompt_tokens}  \n"
        f"Successful Requests: {usage_metrics.successful_requests} "
    )
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
        crew_run_metrics.markdown() + "\n\nCrew output:\n\n" + output_content, encoding="utf-8"
    )
    return file_path
