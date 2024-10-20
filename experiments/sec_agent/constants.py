TQDM_BAR_FORMAT = "{l_bar}{bar} | Iteration: {n_fmt}/{total_fmt} [Time - Remaining: {remaining} Elapsed: {elapsed}]"

DEFAULT_TICKER = "NVDA"
FILINGS_DIR = "filings"

# List of available models:
# https://huggingface.co/models?inference=warm&sort=trending&pipeline_tag=text-generation
# HF_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
