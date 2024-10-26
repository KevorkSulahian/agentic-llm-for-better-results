from finmas.crews.model_provider import get_embedding_model, get_llama_index_llm
from finmas.data.sec.parser import parse_sec_filing_to_documents
from finmas.data.sec.tool import SECSemanticSearchTool


def get_sec_query_engine(
    ticker: str,
    llm_provider: str,
    llm_model: str,
    embedding_model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """
    Fetch SEC filings for a ticker, prepare the embedding model, and initialize the LLM for analysis.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
        llm_provider (str): LLM provider (e.g., 'huggingface', 'openai')
        llm_model (str): LLM model name

    Returns:
        LlamaIndexTool: A tool for querying SEC filings
    """
    sec_tool = SECSemanticSearchTool()

    # Try 10-Q, otherwise fall back to 10-K
    filing_type = "10-Q"
    filing_content = sec_tool.search_filing(ticker, filing_type=filing_type)
    if "couldn't find any 10-Q filings" in filing_content:
        filing_type = "10-K"
        filing_content = sec_tool.search_filing(ticker, filing_type=filing_type)

    # Parse the filing content into a list of Document objects
    documents = parse_sec_filing_to_documents(filing_content, filing_type, ticker)

    embed_model = get_embedding_model(embedding_model)

    from llama_index.core import VectorStoreIndex

    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    index.storage_context.persist(persist_dir="sec_storage")

    llama_index_llm = get_llama_index_llm(
        llm_provider=llm_provider,
        llm_model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    query_engine = index.as_query_engine(llm=llama_index_llm)

    return query_engine
