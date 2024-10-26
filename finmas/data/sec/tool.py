import re
from pathlib import Path
from typing import List, Optional

import html2text
import numpy as np
import torch
from datamule import parse_textual_filing
from datamule.filing_viewer.filing_viewer import json_to_html
from edgar import Company, set_identity
from pydantic.v1 import BaseModel, Field
from transformers import AutoModel, AutoTokenizer

from finmas.constants import defaults

# Set identity for SEC API access
set_identity("John Doe john.doe@example.com")


class SECFilingSearchSchema(BaseModel):
    """Input schema for SEC filings search."""

    ticker: str = Field(..., description="Mandatory valid stock ticker (e.g., 'NVDA').")


class SECSemanticSearchTool:
    """A tool to perform semantic searches in 10-K and 10-Q SEC filings for a specified company."""

    predefined_metrics = [
        "Total revenue",
        "Net income",
        "Operating expenses",
        "Risk factors",
        "Cash flow",
        "Management's discussion",
    ]

    def __init__(self, model_name: str = defaults["hf_embedding_model"]):
        """
        Initializes the SEC tools for retrieving and searching through SEC filings.

        Args:
            model_name: The HuggingFace model name for generating text embeddings.
        """
        # HF tokenizer and model for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # FAISS index to store
        # and retrieve embeddings
        from faiss import IndexFlatL2

        self.index = IndexFlatL2(768)

    def search_filing(self, stock: str, filing_type: str) -> str:
        """
        Automatically extract summary for key financial metrics from
        the latest filing for a given stock ticker.

        Args:
            stock: The stock ticker symbol (e.g., 'AAPL').
        """
        content = self.get_filing_content(stock, filing_type=filing_type)
        if not content:
            return f"Sorry, I couldn't find any {filing_type} filings for {stock}."

        return self.extract_key_metrics(content)

    def get_filing_content(self, ticker: str, filing_type: str) -> Optional[str]:
        """
        Fetches the latest 10-K or 10-Q filing content for the given stock ticker.
        The functions downloads the filing as HTML and converts it to plain text.

        Args:
            ticker: The stock ticker symbol.
            filing_type: The type of filing (e.g., '10-K' or '10-Q').
        """
        try:
            filings_dir = Path(defaults["filings_dir"]) / ticker / filing_type
            filings_dir.mkdir(parents=True, exist_ok=True)
            filing = Company(ticker).get_filings(form=filing_type).latest(1)

            filename = filing.document.document
            output_file = filings_dir / filename
            if not output_file.exists():
                filing.document.download(path=filings_dir)

            json_content = parse_textual_filing(filing.document.url, return_type="json")
            html_content = json_to_html(json_content)
            return self.convert_html_to_text(html_content)
        except Exception as e:
            print(f"Error fetching {filing_type} URL: {e}")

        return None

    def convert_html_to_text(self, html_content: str) -> str:
        """
        Converts the HTML content of the SEC filing into plain text.

        Args:
            file_path: The file path of the downloaded SEC filing.
        """
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_tables = False

        return h.handle(html_content)

    def extract_key_metrics(self, content: str) -> str:
        """
        Automatically extract a structured summary for each
        predefined key financial metrics from the filing content.

        Args:
            content: The plain text content of the filing.
        """
        results = []
        for metric in self.predefined_metrics:
            relevant_text = self.embedding_search(content, metric)
            results.append(f"### {metric}\n\n{relevant_text}\n\n")

        return "\n".join(results)

    def embedding_search(self, content: str, query: str) -> str:
        """
        Perform an embedding-based search using FAISS to retrieve most relevant sections of the content.

        Args:
            content: The plain text content of the filing.
            query: The search query.
        """
        chunks = self.chunk_text(content, chunk_size=1000, overlap=100)
        chunk_embeddings = [self.get_embedding(chunk) for chunk in chunks]

        chunk_embeddings_np = np.vstack(chunk_embeddings)
        # Dynamically adjusting embedding dimension
        # based on the model's embedding size
        embedding_dim = chunk_embeddings_np.shape[1]

        # Initialize FAISS index with the correct embedding size
        from faiss import IndexFlatL2

        self.index = IndexFlatL2(embedding_dim)

        # Eembeddings -> FAISS index
        self.index.add(chunk_embeddings_np)

        query_embedding = self.get_embedding(query)
        # Top 3 results
        D, indices = self.index.search(np.array([query_embedding]), k=3)

        relevant_chunks = [chunks[i] for i in indices[0]]
        return "\n\n".join(relevant_chunks)

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Splits the text into a list of chunks with a defined overlap for better context retention.

        Args:
            text: The input text to be chunked.
            chunk_size: The size of each chunk in characters. Defaults to 1000.
            overlap: The number of overlapping characters between chunks. Defaults to 100.
        """
        sentences = re.split(r"(?<=[.!?]) +", text)
        chunks, chunk = [], []
        char_count = 0

        for sentence in sentences:
            char_count += len(sentence)
            chunk.append(sentence)
            if char_count > chunk_size - overlap:
                chunks.append(" ".join(chunk))
                chunk, char_count = [], 0

        if chunk:
            chunks.append(" ".join(chunk))
        return chunks

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate the embedding vector for a given text using a pre-trained HuggingFace model.

        Args:
            text: The input text to generate embeddings for.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
