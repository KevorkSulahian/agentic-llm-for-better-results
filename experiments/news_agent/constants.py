from enum import Enum

TQDM_BAR_FORMAT = "{l_bar}{bar} | Iteration: {n_fmt}/{total_fmt} [Time - Remaining: {remaining} Elapsed: {elapsed}]"

DEFAULT_TICKER = "TSLA"
FILINGS_DIR = "filings"

# List of available models:
# https://huggingface.co/models?inference=warm&sort=trending&pipeline_tag=text-generation
HF_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
HF_MODELS = {
    "llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.3",
}
HF_BASE_URL = "https://api-inference.huggingface.co/models"
DEFAULT_HF_MODEL = "llama-3.2-3B-Instruct"

DEFAULT_GROQ_MODEL = "llama3-8b-8192"

ALPACA_NEWS_START_DATE = "2020-01-01"

OUTPUT_DIR = "results"


class LlmProviderEnum(str, Enum):
    huggingface = "huggingface"
    groq = "groq"


class LlmNameEnum(str, Enum):
    llama3_2_1B_Instruct = "llama-3.2-1B-Instruct"
    llama3_2_3B_Instruct = "llama-3.2-3B-Instruct"
    llama3_1 = "llama3.1"
    # gemini1_5_pro = "gemini-1.5-pro"
    # gemini1_5_flash = "gemini-1.5-flash"
    mistral_7B_Instruct = "mistral-7B-Instruct"
    # mistral_nemo = "mistral-nemo"
    gemma2_9b_it = "gemma2-9b-it"


class NewsSourceEnum(str, Enum):
    yahoo_finance = "yahoo"
    benzinga = "benzinga"
