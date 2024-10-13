import os

import click
import pandas as pd
import requests
import typer
from rich import print


def get_groq_models() -> pd.DataFrame:
    """Get a list of currenct active Groq models"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")

    url = "https://api.groq.com/openai/v1/models"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.get(url, headers=headers)

    df = pd.DataFrame(response.json()["data"])
    df["created"] = pd.to_datetime(df["created"], unit="s").dt.strftime("%Y-%m-%d")

    return (
        df.drop(columns=["object", "active", "public_apps"])
        .sort_values(by=["owned_by", "context_window", "id"])
        .reset_index(drop=True)
    )


def prompt_for_id(df: pd.DataFrame, default_id: str, id_column: str = "id") -> str:
    """Prompt the user to choose a row from a dataframe based on an id column.

    Args:
        df: The dataframe to choose from.
        default_id: The default id to pre-select.
        id_column: The column to use for the id. Defaults to "id".

    Returns:
        The chosen id.
    """
    print(df)

    row_choice = click.Choice(list(map(str, range(len(df)))))
    default_index = str(df.index[df[id_column] == default_id].tolist()[0])
    row_chosen = typer.prompt(
        f"Choose index (0-{len(df)-1})", default_index, show_choices=False, type=row_choice
    )
    id_chosen = df.iloc[int(row_chosen)][id_column]
    print(f"ID chosen: {id_chosen}")
    return id_chosen
