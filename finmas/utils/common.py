import datetime as dt
import os
import re
import shutil
from pathlib import Path

import financedatabase as fd
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from finmas.cache_config import cache
from finmas.constants import MARKET_CAP_MAP, TICKER_COLS, defaults
from finmas.logger import get_logger

HF_ACTIVE_MODELS_URL = (
    "https://huggingface.co/models?inference=warm&pipeline_tag=text-generation&sort=trending"
)


logger = get_logger(__name__)


def get_vector_store_index_dir(ticker: str, data_type: str, subfolder: str | None = None) -> str:
    """
    Get the directory of the vector store index

    Args:
        ticker: The ticker of the stock
        data_type: The type of data (e.g. news, sec, etc.)
        subfolder: An optional subfolder inside the vector store index directory
    """
    index_dir = Path(f"{defaults['vector_store_index_dir']}/{ticker}/{data_type}")
    if subfolder:
        index_dir = index_dir / subfolder

    if Path(index_dir).exists():
        shutil.rmtree(index_dir)
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    return str(index_dir)


def get_text_content_file(ticker: str, data_type: str, suffix: str | None = None) -> Path:
    """
    Get the text content file for the given data type

    Args:
        data_type: The type of data (e.g. news, sec, fundamentals, etc.)
    """
    text_content_dir = Path(f"{defaults['text_content_dir']}/{data_type}")
    text_content_dir.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{ticker}_{suffix + '_' if suffix else ''}{timestamp}.md"
    return text_content_dir / filename


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


def date_to_str(date: dt.date):
    return date.strftime("%Y-%m-%d")


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


def get_llm_models_df(llm_providers: list[str]) -> pd.DataFrame:
    """Get a DataFrame of the current active LLM models from multiple providers"""
    dfs = []
    for provider in llm_providers:
        dfs.append(get_valid_models(provider))
    return pd.concat(dfs).reset_index(drop=True)


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
        models.append({"id": model_info[0]})

    df = pd.DataFrame(models)
    df["context_window"] = np.nan
    df["owned_by"] = df["id"].str.split("/").str[0]
    df["created"] = np.nan
    df.insert(0, "provider", "huggingface")

    return df


# @cache.memoize(expire=dt.timedelta(seconds=1).total_seconds())
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
    df = df[["id", "context_window", "owned_by", "created"]]
    df.insert(0, "provider", "groq")

    return df.sort_values(by=["owned_by", "context_window", "id"]).reset_index(drop=True)


def get_openai_models() -> pd.DataFrame:
    """Get a DataFrame of the currenct active OpenAI models"""
    df = pd.DataFrame(
        {
            "provider": "openai",
            "id": defaults["openai_models"],
            "context_window": np.nan,
            "owned_by": "OpenAI",
            "created": np.nan,
        }
    )
    return df


def get_embedding_models_df() -> pd.DataFrame:
    """Returns a DataFrame of the possible embedding models"""
    records = []
    for model in defaults["openai_embedding_models"]:
        records.append(
            {
                "provider": "openai",
                "id": model,
                "link": "https://platform.openai.com/docs/guides/embeddings",
            }
        )

    for model in defaults["hf_embedding_models"]:
        records.append(
            {
                "provider": "huggingface",
                "id": model,
                "link": f"https://huggingface.co/{model}",
            }
        )
    return pd.DataFrame(records)


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
        logger.info("Using FinanceDatabase")
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
