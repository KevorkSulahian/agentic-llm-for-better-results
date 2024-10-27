FILINGS_DIR = "filings"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1024

DEFAULT_TICKER = "COIN"
TICKER_OPTIONS = ["NVDA", "AAPL", "MSFT", "WMT", "MCD", "COIN"]

DEFAULT_QUERY_TEXT = f"What are the main risk factors for {DEFAULT_TICKER}"

LLM_PROVIDER = "groq"
LLM_PROVIDERS = ["groq", "openai", "google"]
LLM_MODELS = {
    "groq": [
        "llama3-8b-8192",
        "llama3-70b-8192",
        "llama3-groq-8b-8192-tool-use-preview",
        "llama3-groq-70b-8192-tool-use-preview",
        "llama-3.2-90b-text-preview",
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
    ],
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "google": ["gemini"],
}

HF_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

CLEAR_EMBEDDINGS_CACHE = True
CLEAR_INDEX = True
