from typing import Optional, List
from pydantic.v1 import BaseModel, Field
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
from edgar import Company
import html2text
import re
from pathlib import Path
from constants import HF_EMBEDDING_MODEL, FILINGS_DIR


class SECFilingSearchSchema(BaseModel):
    """Input schema for SEC filings search."""

    ticker: str = Field(..., description="Mandatory valid stock ticker (e.g., 'NVDA').")
    query: str = Field(
        ...,
        description="Query to search within the SEC filing (e.g., 'What is last year's revenue?').",
    )


class SECTools:
    """A tool to perform semantic searches in 10-K and 10-Q SEC filings for a specified company."""

    def __init__(self, model_name: str = HF_EMBEDDING_MODEL):
        """
        Initializes the SEC tools for retrieving and searching through SEC filings.

        :param model_name: The HuggingFace model name for generating text embeddings.
        """
        # HF tokenizer and model for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # FAISS index to store
        # and retrieve embeddings
        self.index = faiss.IndexFlatL2(768)

    def search_10q(self, stock: str, query: str) -> str:
        """
        Search a query within the latest 10-Q filing for a given stock ticker.

        :param stock: The stock ticker symbol (e.g., 'AAPL').
        :param query: The query to search within the filing.
        :return: The most relevant sections of the filing.
        """
        content = self.get_filing_content(stock, "10-Q")
        if not content:
            return f"Sorry, I couldn't find any 10-Q filings for {stock}."

        return self.embedding_search(content, query)

    def search_10k(self, stock: str, query: str) -> str:
        """
        Search a query within the latest 10-K filing for a given stock ticker.

        :param stock: The stock ticker symbol (e.g., 'AAPL').
        :param query: The query to search within the filing.
        :return: The most relevant sections of the filing.
        """
        content = self.get_filing_content(stock, "10-K")
        if not content:
            return f"Sorry, I couldn't find any 10-K filings for {stock}."

        return self.embedding_search(content, query)

    def get_filing_content(self, ticker: str, filing_type: str) -> Optional[str]:
        """
        Fetches the latest 10-K or 10-Q filing content for the given stock ticker.

        :param ticker: The stock ticker symbol.
        :param filing_type: The type of filing (e.g., '10-K' or '10-Q').
        :return: The content of the filing as a plain text string.
        """
        try:
            filings_dir = Path(FILINGS_DIR) / ticker / filing_type
            filings_dir.mkdir(parents=True, exist_ok=True)
            filings = Company(ticker).get_filings(form=filing_type).latest(1)

            for filing in [filings]:
                filename = filing.document.model_dump().get("document")
                output_file = filings_dir / filename
                if not output_file.exists():
                    filing.document.download(path=filings_dir)
                return self.convert_html_to_text(output_file)
        except Exception as e:
            print(f"Error fetching {filing_type} URL: {e}")

        return None

    def convert_html_to_text(self, file_path: str) -> str:
        """
        Converts the HTML content of the SEC filing into plain text.

        :param file_path: The file path of the downloaded SEC filing.
        :return: The plain text representation of the filing.
        """
        """Convert the HTML content of the filing to plain text."""
        h = html2text.HTML2Text()
        h.ignore_links = False
        with open(file_path, "r", encoding="utf-8") as html_file:
            html_content = html_file.read()
        return h.handle(html_content)

    def embedding_search(self, content: str, query: str) -> str:
        """
        Perform an embedding-based search using FAISS to retrieve relevant sections of the content.

        :param content: The plain text content of the filing.
        :param query: The search query.
        :return: The most relevant sections of the content.
        """
        chunks = self.chunk_text(content, chunk_size=1000, overlap=100)
        chunk_embeddings = [self.get_embedding(chunk) for chunk in chunks]

        chunk_embeddings_np = np.vstack(chunk_embeddings)
        # Dynamically adjusting embedding dimension
        # based on the model's embedding size
        embedding_dim = chunk_embeddings_np.shape[1]

        # Initialize FAISS index with the correct embedding size
        self.index = faiss.IndexFlatL2(embedding_dim)

        # Eembeddings -> FAISS index
        self.index.add(chunk_embeddings_np)

        query_embedding = self.get_embedding(query)
        # Top 3 results
        D, indices = self.index.search(np.array([query_embedding]), k=3)

        relevant_chunks = [chunks[i] for i in indices[0]]
        return "\n\n".join(relevant_chunks)

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Splits the text into chunks with a defined overlap for better context retention.

        :param text: The input text to be chunked.
        :param chunk_size: The size of each chunk in characters.
        :param overlap: The number of overlapping characters between chunks.
        :return: A list of text chunks.
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
        Generate the embedding for a given text using a pre-trained HuggingFace model.

        :param text: The input text to generate embeddings for.
        :return: A NumPy array representing the text embedding.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
