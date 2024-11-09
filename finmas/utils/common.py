import datetime as dt
import os
import re

import financedatabase as fd
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from finmas.cache_config import cache
from finmas.constants import MARKET_CAP_MAP, TICKER_COLS, defaults

HF_ACTIVE_MODELS_URL = (
    "https://huggingface.co/models?inference=warm&pipeline_tag=text-generation&sort=trending"
)


def extract_cols_from_df(df: pd.DataFrame, cols_map: dict[str, str]) -> pd.DataFrame:
    df = df[list(cols_map.keys())].copy()
    return df.rename(columns=cols_map)


def format_time_spent(seconds: float):
    """Format the time spent in a human-readable way"""
    minutes, seconds = divmod(seconds, 60)
    output = ""
    if minutes > 0:
        output += f"{int(minutes)}m "
    if seconds > 0:
        output += f"{int(seconds)}s"
    return output


def to_datetime(date: dt.date):
    if isinstance(date, dt.date):
        return dt.datetime.combine(date, dt.time(0))
    raise ValueError("Input must be a datetime.date object")


def get_environment_variable(key: str) -> str:
    """Get the value from the environment variables"""
    try:
        return os.environ[key]
    except KeyError:
        raise ValueError(
            f"{key} not found in environment variables. Please set {key} in the .env file."
        )


def get_valid_models(llm_provider: str) -> pd.DataFrame:
    """Get a list of valid models for the given LLM provider"""
    if llm_provider == "groq":
        return get_groq_models()
    elif llm_provider == "huggingface":
        return get_huggingface_models()
    elif llm_provider == "openai":
        return get_openai_models()
    else:
        raise ValueError(f"Invalid LLM provider: {llm_provider}")


@cache.memoize(expire=dt.timedelta(days=1).total_seconds())
def get_huggingface_models():
    """Get a DataFrame of the current active HuggingFace LLM models"""
    response = requests.get(HF_ACTIVE_MODELS_URL)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    model_cards = soup.find_all("article", class_="overview-card-wrapper")

    models = []
    for card in model_cards:
        model_text = re.sub(
            r"\s+",
            " ",
            re.sub(r"[^\x00-\x7F]+", "", card.text.replace("\n", "").replace("\t", " ")),
        ).strip()

        model_info = model_text.split(" ")
        models.append(
            {
                "id": model_info[0],
                "downloads": model_info[-2],
                "likes": model_info[-1],
            }
        )

    df = pd.DataFrame(models)
    df["context_window"] = np.nan
    df["owned_by"] = df["id"].str.split("/").str[0]
    df["created"] = np.nan

    return df


@cache.memoize(expire=dt.timedelta(days=1).total_seconds())
def get_groq_models() -> pd.DataFrame:
    """Get a DataFrame of the currenct active Groq models"""
    headers = {
        "Authorization": f"Bearer {get_environment_variable('GROQ_API_KEY')}",
        "Content-Type": "application/json",
    }

    response = requests.get("https://api.groq.com/openai/v1/models", headers=headers)

    df = pd.DataFrame(response.json()["data"])

    df["created"] = pd.to_datetime(df["created"], unit="s").dt.strftime("%Y-%m-%d")

    df = df[~df["id"].str.contains("whisper")]
    df = df[~df["id"].str.contains("vision")]
    df["downloads"] = np.nan
    df["likes"] = np.nan
    df = df[["id", "downloads", "likes", "context_window", "owned_by", "created"]]

    return df.sort_values(by=["owned_by", "context_window", "id"]).reset_index(drop=True)


def get_openai_models() -> pd.DataFrame:
    """Get a DataFrame of the currenct active OpenAI models"""
    df = pd.DataFrame(
        {
            "id": defaults["openai_models"],
            "downloads": np.nan,
            "likes": np.nan,
            "context_window": np.nan,
            "owned_by": "OpenAI",
            "created": np.nan,
        }
    )
    return df


@cache.memoize(expire=dt.timedelta(days=1).total_seconds())
def get_wikipedia_sp500_tickers() -> pd.DataFrame:
    """Gets a DataFrame of S&P500 tickers from Wikipedia"""
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df = df.set_index("symbol")
    return df


@cache.memoize(expire=dt.timedelta(days=1).total_seconds())
def get_tickers_df(sp500: bool = False):
    """Gets a DataFrame of S&P500 tickers with info from Wikipedia and FinanceDatabase"""
    equities_df = fd.Equities().select()
    if sp500:
        sp500_df = get_wikipedia_sp500_tickers()
        df = sp500_df.join(equities_df, how="inner")
    else:
        print("Using FinanceDatabase")
        # There are multiple duplicates in the FinanceDatabase. Keep the first.
        df = equities_df.drop_duplicates(subset=["name"])
        df = df.dropna(subset=["market", "market_cap"], how="any")
        df = df[df["market"].str.contains("NASDAQ|New York")]
        # Filter out tickers according to tickers_market_cap_exclude configuration
        df = df[~df["market_cap"].str.contains("|".join(defaults["tickers_market_cap_exclude"]))]

    df.index.name = "ticker"
    df = df[TICKER_COLS].reset_index()

    df["market_cap_sort"] = df["market_cap"].map(MARKET_CAP_MAP)
    df = df.sort_values(by=["market_cap_sort", "ticker"], ascending=[False, True])
    return df.drop(columns=["market_cap_sort"])
