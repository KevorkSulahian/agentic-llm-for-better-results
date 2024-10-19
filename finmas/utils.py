import os

import pandas as pd
import requests
import datetime as dt


def to_datetime(date: dt.date):
    if isinstance(date, dt.date):
        return dt.datetime.combine(date, dt.time(0))
    raise ValueError("Input must be a datetime.date object")


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

    df = df[~df["id"].str.contains("whisper")]
    df = df[~df["id"].str.contains("vision")]
    df = df[["id", "context_window", "owned_by", "created"]]

    return df.sort_values(by=["owned_by", "context_window", "id"]).reset_index(drop=True)
