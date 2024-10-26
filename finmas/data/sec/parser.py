from llama_index.core import Document
import datetime as dt


def parse_sec_filing_to_documents(
    filing_content: str, filing_type: str, ticker: str
) -> list[Document]:
    """
    Convert the SEC filing content into a list of Document objects for LlamaIndex.

    Args:
        filing_content: The raw SEC filing content.
        filing_type: The type of filing (e.g., '10-Q', '10-K').
        ticker: The stock ticker symbol (e.g., 'AAPL').
    """
    current_date = dt.datetime.now().date().isoformat()

    doc_text = (
        f"Ticker: {ticker}\n"
        f"Filing Type: {filing_type}\n"
        f"Fetched Date: {current_date}\n\n"
        f"{filing_content}"
    )

    return [Document(text=doc_text)]
