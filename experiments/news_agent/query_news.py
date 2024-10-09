import datetime as dt
import os
import time
from pathlib import Path

import feedparser
import pandas as pd
import typer
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
from constants import (
    HF_EMBEDDING_MODEL,
    HF_MODELS,
    OUTPUT_DIR,
    LlmNameEnum,
    LlmProviderEnum,
    NewsSourceEnum,
)
from dotenv import find_dotenv, load_dotenv
from llama_index.core import Document, Response, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from rich import print
from typing_extensions import Annotated

load_dotenv(find_dotenv())

HF_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
ALPACA_START_DATE = "2024-01-01"


def parse_news_to_documents(news: list[dict]):
    documents = []
    for item in news:
        text = (
            f"Title: {item['title']}\n"
            f"Published: {item['published'].date().isoformat()}\n"
            f"Summary: {item['summary']}"
        )
        doc = Document(text=text)
        documents.append(doc)

    return documents


def get_news(ticker: str, source: NewsSourceEnum) -> list[dict]:
    match source:
        case NewsSourceEnum.yahoo_finance:
            print(f"Fetching Yahoo Finance news for {ticker}")
            return get_yahoo_finance_news(ticker)
        case NewsSourceEnum.benzinga:
            print(f"Fetching Benzinga news for {ticker}")
            return get_benzinga_news(ticker)
        case _:
            raise ValueError("Invalid news source")


def get_yahoo_finance_news(ticker: str) -> list[dict]:
    rss_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    feed = feedparser.parse(rss_url)

    records = []
    for entry in feed.entries:
        record = dict(
            title=entry.title,
            link=entry.link,
            id=entry.id,
            published=dt.datetime.fromtimestamp(time.mktime(entry.published_parsed)),
            summary=entry.summary,
        )
        records.append(record)

    return records


def get_benzinga_news(ticker: str) -> list[dict]:
    assert os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_API_SECRET")

    client = NewsClient(
        api_key=os.getenv("ALPACA_API_KEY"), secret_key=os.getenv("ALPACA_API_SECRET")
    )

    request = NewsRequest(
        symbols=ticker,
        start=dt.datetime.fromisoformat(ALPACA_START_DATE),
        end=dt.datetime.now(),
        include_content=True,
    )
    news_set = client.get_news(request_params=request)

    records = []
    for news in news_set.data["news"]:
        record = dict(
            title=news.headline,
            link=news.url,
            id=news.id,
            published=news.updated_at,
            summary=news.summary,
            # TODO: Add content, symbols, author fields
        )
        records.append(record)

    return records


def parse_news_to_dataframe(news: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(news)
    return df


def get_embedding_model():
    cache_dir = Path("embeddings").absolute()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variable for Hugging Face to use our cache directory
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    return HuggingFaceEmbedding(
        model_name=HF_EMBEDDING_MODEL, device="cpu", cache_folder=str(cache_dir)
    )


def get_llama_index_llm(llm_name: LlmNameEnum, llm_provider: LlmProviderEnum):
    if llm_provider == LlmProviderEnum.huggingface:
        if llm_name.value not in HF_MODELS:
            raise ValueError(
                f"Invalid LLM name: {llm_name.value} Choose one of: {HF_MODELS.keys()}"
            )

        return HuggingFaceInferenceAPI(
            model_name=HF_MODELS[llm_name.value],
            temperature=0.7,
            max_tokens=100,
            token=os.getenv("HF_TOKEN"),
        )
    else:
        raise ValueError(f"Invalid LLM provider: {llm_provider.value}")


def main(
    ticker: Annotated[str, typer.Argument(help="Ticker")],
    source: Annotated[
        NewsSourceEnum, typer.Option(help="News Source")
    ] = NewsSourceEnum.yahoo_finance,
    llm_provider: Annotated[
        LlmProviderEnum, typer.Option(help="Model provider")
    ] = LlmProviderEnum.huggingface,
    llm_name: Annotated[
        LlmNameEnum,
        typer.Option(help="Model name"),
    ] = LlmNameEnum.llama3_2_1B_Instruct,
    output_dir: Annotated[str, typer.Option(help="Output directory for results")] = OUTPUT_DIR,
):
    """
    Fetching the latest news for a given ticker and chat with the news.

    Methodology:\n
    1. News snippets are fetched from the source.\n
    2. The news are converted into llama-index Documents.\n
    3. A embedding model is fetched from HuggingFace\n
    4. A VectorStoreIndex is created from the Documents and the embedding model.\n
    5. A llama-index LLM model is created.\n
    6. A query engine is created from the VectorStoreIndex with the LLM model.\n
    7. A loop lets you query the news using the query engine.
    """

    print(f"Fetching news for {ticker}")
    print("Configuration: ")
    print("Source: ", source.value, "LLM: ", llm_name.value, " LLM Provider: ", llm_provider.value)

    records = get_news(ticker, source)

    documents = parse_news_to_documents(records)

    embed_model = get_embedding_model()

    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    index.storage_context.persist(persist_dir="storage")

    llama_index_llm = get_llama_index_llm(llm_name, llm_provider)

    query_engine = index.as_query_engine(llm=llama_index_llm)
    print()

    news_df = parse_news_to_dataframe(records)
    with pd.option_context("display.max_colwidth", 100):
        news_df["published"] = news_df["published"].dt.strftime("%Y-%m-%d %H:%M")
        print(news_df[["title", "published"]])

    while True:
        query = input("Enter a query (or 'q' to quit): ")
        if query.lower() == "q":
            break
        response: Response = query_engine.query(query)

        print(response.response)
        answer = input("Do you want to see the source nodes?: [y/n]")
        if answer.lower() == "y":
            print(response.source_nodes)


if __name__ == "__main__":
    typer.run(main)
