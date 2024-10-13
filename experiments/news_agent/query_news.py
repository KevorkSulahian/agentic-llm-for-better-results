import os
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from constants import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_HF_MODEL,
    HF_EMBEDDING_MODEL,
    HF_MODELS,
    OUTPUT_DIR,
    LlmNameEnum,
    LlmProviderEnum,
    NewsSourceEnum,
)
from dotenv import find_dotenv, load_dotenv
from news_fetcher import NewsFetcher, parse_news_to_dataframe, parse_news_to_documents
from rich import print
from typing_extensions import Annotated
from utils import get_groq_models, prompt_for_id

load_dotenv(find_dotenv())

HF_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def get_embedding_model():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    cache_dir = Path("embeddings").absolute()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variable for Hugging Face to use our cache directory
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    return HuggingFaceEmbedding(
        model_name=HF_EMBEDDING_MODEL, device="cpu", cache_folder=str(cache_dir)
    )


def get_llama_index_llm(
    llm_provider: LlmProviderEnum,
    llm_name: LlmNameEnum | None = None,
    temperature: float = 0.7,
    max_tokens: int = 100,
):
    """Get a llama-index compatible LLM model"""
    if llm_provider == LlmProviderEnum.huggingface:
        if llm_name is None:
            llm_name = LlmNameEnum(DEFAULT_HF_MODEL)
            print(f"Using default LLM: {llm_name.value}")
        if llm_name.value not in HF_MODELS:
            raise ValueError(
                f"Invalid LLM name: {llm_name.value} Choose one of: {HF_MODELS.keys()}"
            )
        from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

        return HuggingFaceInferenceAPI(
            model_name=HF_MODELS[llm_name.value],
            temperature=temperature,
            max_tokens=max_tokens,
            token=os.getenv("HF_TOKEN"),
        )
    elif llm_provider == LlmProviderEnum.groq:
        models_df = get_groq_models()
        if llm_name is None:
            print("Available models:")
            model_id = prompt_for_id(models_df, default_id=DEFAULT_GROQ_MODEL)
        else:
            model_id = llm_name.value

        from llama_index.llms.groq import Groq

        return Groq(model=model_id, temperature=temperature, max_tokens=max_tokens)
    else:
        raise ValueError(f"Invalid LLM provider: {llm_provider.value}")


def main(
    ticker: Annotated[str, typer.Argument(help="Ticker")],
    source: Annotated[
        NewsSourceEnum, typer.Option(help="News Source")
    ] = NewsSourceEnum.yahoo_finance,
    llm_provider: Annotated[
        LlmProviderEnum,
        typer.Option(help="LLM provider", prompt="LLM provider", show_choices=True),
    ] = LlmProviderEnum.groq,
    llm_name: Annotated[
        Optional[LlmNameEnum],
        typer.Option(help="LLM name"),
    ] = None,
    temperature: Annotated[float, typer.Option(help="Temperature")] = 0.7,
    max_tokens: Annotated[int, typer.Option(help="Max tokens")] = 100,
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
    print(f"Source: {source.value} LLM Provider: {llm_provider.value}")
    print(f"LLM Model: {llm_name.value if llm_name else 'None'}")

    llama_index_llm = get_llama_index_llm(
        llm_provider, llm_name, temperature=temperature, max_tokens=max_tokens
    )

    news_fetcher = NewsFetcher(source=source)
    records = news_fetcher.get_news(ticker)
    news_df = parse_news_to_dataframe(records)
    with pd.option_context("display.max_colwidth", 100):
        news_df["published"] = news_df["published"].dt.strftime("%Y-%m-%d %H:%M")
        print(news_df[["title", "published"]])

    print("Loading embedding model, creating vector store index and loading LLM model")

    documents = parse_news_to_documents(records)

    embed_model = get_embedding_model()

    from llama_index.core import Response, VectorStoreIndex

    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    index.storage_context.persist(persist_dir="storage")

    query_engine = index.as_query_engine(llm=llama_index_llm)
    print("Model ready.")
    print()

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
