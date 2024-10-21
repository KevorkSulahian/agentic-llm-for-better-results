from llama_index.core import Document
import datetime as dt


def parse_sec_filing_to_documents(
    filing_content: str, filing_type: str, ticker: str
) -> list[Document]:
    """
    Convert the SEC filing content into a list of Document objects for LlamaIndex.

    Args:
        filing_content (str): The raw SEC filing content.
        filing_type (str): The type of filing (e.g., '10-Q', '10-K').
        ticker (str): The stock ticker symbol (e.g., 'AAPL').

    Returns:
        list[Document]: A list of Document objects to be indexed by LlamaIndex.
    """
    current_date = dt.datetime.now().date().isoformat()

    doc_text = (
        f"Ticker: {ticker}\n"
        f"Filing Type: {filing_type}\n"
        f"Fetched Date: {current_date}\n\n"
        f"{filing_content}"
    )

    return [Document(text=doc_text)]
