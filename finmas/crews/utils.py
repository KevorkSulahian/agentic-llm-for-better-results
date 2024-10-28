from dataclasses import dataclass
from pathlib import Path

import yaml

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


def get_yaml_config_as_markdown(config_path: Path, config_file: str):
    with open(config_path / f"{config_file}.yaml", "r") as c:
        config = yaml.safe_load(c)

    # output = f"## {config_file.title()}\n\n"
    output = ""
    for key, value in config.items():
        output += f"### {key.replace('_', ' ').title()}\n\n"
        for field, specification in value.items():
            output += f"- **{field.replace('_', ' ').title()}**: {specification}"
        output += "\n"

    return output


def get_usage_metrics_as_string(usage_metrics):
    output = (
        f"Total tokens: {usage_metrics.total_tokens} "
        f"Prompt tokens: {usage_metrics.prompt_tokens}\n"
        f"Successful Requests: {usage_metrics.successful_requests} "
    )
    return output
