import datetime as dt
import time
from pathlib import Path

import html2text
from datamule import parse_textual_filing
from datamule.filing_viewer.filing_viewer import json_to_html
from edgar import Filing, set_identity

from finmas.constants import defaults
from finmas.crews.model_provider import get_embedding_model, get_llama_index_llm
from finmas.crews.utils import IndexCreationMetrics
from finmas.data.sec.sec_parser import SECTION_FILENAME_MAP, SECFilingParser
from finmas.data.sec.tool import SECSemanticSearchTool
from finmas.data.token_counting import token_counter
from finmas.logger import get_logger
from finmas.utils.common import get_text_content_file, get_vector_store_index_dir

set_identity("John Doe john.doe@example.com")

logger = get_logger(__name__)

SUPPORTED_METHODS = [
    "section:mda",
    "section:risk_factors",
    "semantic_search_keywords",
    "full_text",
]


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


def _get_sec_text_content(ticker: str, embedding_model: str, filing: Filing, method: str) -> str:
    """Handler function to get the text content of the SEC filing based on the method specified.

    We currently support 3 main methods:
    1. Extract a specific section from the filing (e.g., 'Management Discussion and Analysis')
    2. Extract sections relevant to predefined keywords using a semantic search
    3. Use the full text content of the SEC filing.

    See get_sec_query_engine for description of arguments.
    """

    if method.startswith("section"):
        parser = SECFilingParser(ticker=ticker, form_type=filing.form)
        parser.parse_filing_as_html(filing)
        toc = parser.extract_table_of_contents_from_html()

        section_abbr = method.split(":")[1]
        section_name = SECTION_FILENAME_MAP[section_abbr]
        for i, heading in enumerate(toc[:-1]):
            if section_name in heading:
                next_heading = toc[i + 1]
                text_content = parser.extract_section_from_html(heading, next_heading)
                break
    else:
        text_content = get_sec_filing_as_text_content(ticker=ticker, filing=filing)

        if method == "semantic_search_keywords":
            sec_tool = SECSemanticSearchTool(model_name=embedding_model)
            text_content = sec_tool.extract_key_metrics(content=text_content)

    # Prepend the text content with ticker, filing type and report date
    filing_date = (
        filing.filing_date.strftime("%Y-%m-%d")
        if isinstance(filing.filing_date, dt.date)
        else filing.filing_date
    )
    text_content = (
        f"Ticker: {ticker}\nSEC Filing Form: {filing.form}\n"
        + f"Filing Date: {filing_date}\n\n"
        + text_content
    )
    return text_content


def get_sec_query_engine(
    ticker: str,
    llm_provider: str,
    llm_model: str,
    embedding_model: str,
    filing: Filing,
    method: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    similarity_top_k: int | None = None,
) -> tuple:
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
        embedding_model: Embedding model name according to HuggingFace
        filing: The edgartools SEC filing object.
        method: Method to use for extracting the text content. Supported methods are:
            - 'section:mda': Extract the 'Management Discussion and Analysis' section
            - 'section:risk_factors': Extract the 'Risk Factors' section
            - 'semantic_search_keywords': Extract sections relevant to keywords using a semantic search
            - 'full_text': Use the full text content of the SEC filing.
        temperature: Temperature for the LLM
        max_tokens: Maximum number of tokens for the LLM
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Method {method} is not supported. Supported methods are: {SUPPORTED_METHODS}"
        )

    text_content = _get_sec_text_content(
        ticker=ticker, filing=filing, embedding_model=embedding_model, method=method
    )
    if defaults["save_text_content"]:
        file_path = get_text_content_file(
            ticker=ticker, data_type="sec", suffix=method.replace(":", "_")
        )
        file_path.write_text(text_content, encoding="utf-8")
        logger.info(f"Saved SEC filing text content to {file_path}")

    from llama_index.core import Document

    document = Document(text=text_content, metadata={"SEC Filing Form": filing.form})
    if method.startswith("section"):
        document.metadata["Section"] = SECTION_FILENAME_MAP[method.split(":")[1]]

    embed_model = get_embedding_model(llm_provider, embedding_model)

    from llama_index.core import Settings, VectorStoreIndex

    start = time.time()
    token_counter.reset_counts()
    index = VectorStoreIndex.from_documents([document], embed_model=embed_model)
    index.storage_context.persist(
        persist_dir=get_vector_store_index_dir(
            ticker=ticker, data_type="sec", subfolder=method.replace(":", "_")
        )
    )

    metrics = IndexCreationMetrics(
        embedding_model=embedding_model,
        time_spent=round(time.time() - start, 2),
        num_nodes=len(index.index_struct.nodes_dict.keys()),
        text_length=len(text_content),
        chunk_size=Settings.chunk_size,
        chunk_overlap=Settings.chunk_overlap,
        total_embedding_token_count=token_counter.total_embedding_token_count,
    )

    logger.info(
        f"Created Vector Store Index for SEC filing with {len(index.index_struct.nodes_dict.keys())} nodes"
    )

    llama_index_llm = get_llama_index_llm(
        llm_provider=llm_provider,
        llm_model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    from llama_index.core.query_engine import BaseQueryEngine

    query_engine: BaseQueryEngine = index.as_query_engine(
        llm=llama_index_llm, similarity_top_k=similarity_top_k
    )

    return (query_engine, metrics)
