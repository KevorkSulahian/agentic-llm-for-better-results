import re
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from finmas.constants import defaults

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 100
DEFAULT_TOP_K = 3


class SECSemanticSearchTool:
    """A tool to extract key ."""

    def __init__(
        self,
        model_name: str = defaults["hf_embedding_model_sec"],
        predefined_metrics: list[str] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        top_k: int = DEFAULT_TOP_K,
    ):
        """
        Initializes the SEC tools for retrieving and searching through SEC filings.

        Args:
            model_name: The HuggingFace model name for generating text embeddings.
        """
        self.predefined_metrics = (
            predefined_metrics or defaults["sec_filing_key_metrics_for_compression"]
        )
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k

        # HF tokenizer and model for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # FAISS index to store
        # and retrieve embeddings
        from faiss import IndexFlatL2

        self.index = IndexFlatL2(768)

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
        chunks = self.chunk_text(content, chunk_size=self.chunk_size, overlap=self.overlap)
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
        D, indices = self.index.search(np.array([query_embedding]), k=self.top_k)

        relevant_chunks = [chunks[i] for i in indices[0]]
        return "\n\n".join(relevant_chunks)

    def chunk_text(
        self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
    ) -> List[str]:
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
