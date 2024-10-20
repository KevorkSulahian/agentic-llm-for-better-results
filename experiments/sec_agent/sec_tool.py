from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
from edgar import Company
import html2text
import re
from pathlib import Path
from constants import HF_EMBEDDING_MODEL, FILINGS_DIR


class SECTools:
    def __init__(self):
        # HF tokenizer and model for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(HF_EMBEDDING_MODEL)
        self.model = AutoModel.from_pretrained(HF_EMBEDDING_MODEL)

        # FAISS index to store
        # and retrieve embeddings
        self.index = faiss.IndexFlatL2(768)

    def search_10q(self, stock, ask):
        content = self.get_filing_content(stock, "10-Q")
        if not content:
            return f"Sorry, I couldn't find any 10-Q filings for {stock}."

        return self.embedding_search(content, ask)

    def search_10k(self, stock, ask):
        content = self.get_filing_content(stock, "10-K")
        if not content:
            return f"Sorry, I couldn't find any 10-K filings for {stock}."

        return self.embedding_search(content, ask)

    def get_filing_content(self, ticker, filing_type):
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

    def convert_html_to_text(self, file_path):
        """Convert the HTML content of the filing to plain text."""
        h = html2text.HTML2Text()
        h.ignore_links = False
        with open(file_path, "r", encoding="utf-8") as html_file:
            html_content = html_file.read()
        return h.handle(html_content)

    def embedding_search(self, content, ask):
        """Perform an embedding-based search using FAISS."""
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

        ask_embedding = self.get_embedding(ask)
        # Top 3 results
        D, indices = self.index.search(np.array([ask_embedding]), k=3)

        relevant_chunks = [chunks[i] for i in indices[0]]
        return "\n\n".join(relevant_chunks)

    def chunk_text(self, text, chunk_size=1000, overlap=100):
        """Improved text chunking, chunking at sentence boundaries."""
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

    def get_embedding(self, text):
        """Generate embeddings for a given text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
