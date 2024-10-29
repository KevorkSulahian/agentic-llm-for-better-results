import time
from pathlib import Path

import html2text
from datamule import parse_textual_filing
from datamule.filing_viewer.filing_viewer import json_to_html
from edgar import Filing, set_identity

from finmas.constants import defaults
from finmas.crews.model_provider import get_embedding_model, get_llama_index_llm
from finmas.crews.utils import IndexCreationMetrics
from finmas.data.sec.tool import SECSemanticSearchTool

set_identity("John Doe john.doe@example.com")


def get_sec_filing_as_text_content(ticker: str, filing: Filing) -> str:
    """Fetch the SEC filing as text content. Either from accession number or the latest filing among
    the filing types specified.

    The following methodology is used:
    1. If accession number is not given, then fetch the latest filing according to filing_types.
    2. Parse the HTML content into JSON according to datamule package.
    3. Convert the JSON content to HTML using datamule package.
    4. Convert the HTML content to text using html2text package.
    5. Store the text content as Markdown for inspection by the user.

    Args:
        ticker: Ticker for company
        filing: The SEC filing object for parsing.
    """
    filing_type = filing.form
    filings_dir = Path(defaults["filings_dir"]) / ticker / filing_type
    filings_dir.mkdir(parents=True, exist_ok=True)

    filename = filing.document.document
    output_file = filings_dir / filename
    if not output_file.exists():
        filing.document.download(path=filings_dir)

    filing_url = filing.document.url

    json_content = parse_textual_filing(filing_url, return_type="json")
    html_content = json_to_html(json_content)

    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_tables = False

    text_content = h.handle(html_content)
    with open(output_file.with_suffix(".md"), "w") as f:
        f.write(text_content)

    return text_content


def get_sec_query_engine(
    ticker: str,
    llm_provider: str,
    llm_model: str,
    embedding_model: str,
    filing: Filing,
    compress_filing: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    similarity_top_k: int | None = None,
):
    """
    Create a llama-index query engine that uses a Vector Store Index that is created using the
    text content of the SEC filing.

    The following methodology is used:
    1. Fetch the SEC filing as text content. Tables and images are ignored.
    2. If compress_filing is True, compress the filing content using SECSemanticSearchTool.
       This tool compresses the filing to only extract relevant sections for a selection of keywords.
    3. Create a llama-index Vector Store Index using the filing content.
    4. Create a llama-index query engine using the Vector Store Index and the specified LLM.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        llm_provider: LLM provider (e.g., 'groq', 'openai')
        llm_model: LLM model name
        embedding_model: Embedding model name
        compress_filing: Whether to compress the filing content. Default is False.
        temperature: Temperature for the LLM
        max_tokens: Maximum number of tokens for the LLM
    """

    text_content = get_sec_filing_as_text_content(ticker=ticker, filing=filing)

    if compress_filing:
        sec_tool = SECSemanticSearchTool(model_name=embedding_model)
        text_content = sec_tool.extract_key_metrics(content=text_content)

    from llama_index.core import Document

    documents = [Document(text=text_content, metadata={"SEC Filing Form": filing.form})]

    embed_model = get_embedding_model(embedding_model)

    from llama_index.core import Settings, VectorStoreIndex

    start = time.time()
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    index.storage_context.persist(persist_dir=defaults["sec_filing_index_dir"])

    metrics = IndexCreationMetrics(
        time_spent=round(time.time() - start, 2),
        num_nodes=len(index.index_struct.nodes_dict.keys()),
        text_length=len(text_content),
        chunk_size=Settings.chunk_size,
        chunk_overlap=Settings.chunk_overlap,
    )

    print(f"Created Vector Store Index with {len(index.index_struct.nodes_dict.keys())} nodes")

    llama_index_llm = get_llama_index_llm(
        llm_provider=llm_provider,
        llm_model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    query_engine = index.as_query_engine(llm=llama_index_llm, similarity_top_k=similarity_top_k)

    return (query_engine, metrics)
