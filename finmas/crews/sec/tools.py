import os
from pathlib import Path

from crewai_tools import LlamaIndexTool
from finmas.constants import defaults
from finmas.crews.model_provider import get_llama_index_llm
from finmas.sec.sec_tool import SECTools
from finmas.sec.sec_parser import parse_sec_filing_to_documents


def get_embedding_model():
    """Get the embedding model for SEC filing analysis"""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    cache_dir = Path("embeddings").absolute()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variable for Hugging Face to use our cache directory
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)

    return HuggingFaceEmbedding(
        model_name=defaults["hf_embedding_model"], device="cpu", cache_folder=str(cache_dir)
    )


def get_sec_tool(ticker: str, llm_provider: str, llm_model: str):
    """
    Fetch SEC filings for a ticker, prepare the embedding model, and initialize the LLM for analysis.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
        llm_provider (str): LLM provider (e.g., 'huggingface', 'openai')
        llm_model (str): LLM model name

    Returns:
        LlamaIndexTool: A tool for querying SEC filings
    """
    sec_tool = SECTools()

    # Try 10-Q, otherwise fall back to 10-K
    filing_content = sec_tool.search_10q(ticker)
    filing_type = "10-Q"
    if "couldn't find any 10-Q filings" in filing_content:
        filing_content = sec_tool.search_10k(ticker)
        filing_type = "10-K"

    # Parse the filing content into a list of Document objects
    documents = parse_sec_filing_to_documents(filing_content, filing_type, ticker)

    embed_model = get_embedding_model()

    from llama_index.core import VectorStoreIndex

    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    index.storage_context.persist(persist_dir="sec_storage")

    llama_index_llm = get_llama_index_llm(llm_provider=llm_provider, llm_model=llm_model)

    query_engine = index.as_query_engine(llm=llama_index_llm)

    return LlamaIndexTool.from_query_engine(
        query_engine,
        name="SEC Filing Query Tool",
        description="Use this tool to search and analyze SEC filings",
    )
