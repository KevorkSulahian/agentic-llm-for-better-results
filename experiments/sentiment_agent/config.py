from enum import Enum

TQDM_BAR_FORMAT = "{l_bar}{bar} | Iteration: {n_fmt}/{total_fmt} [Time - Remaining: {remaining} Elapsed: {elapsed}]"

FPB_CONFIGURATION = "sentences_50agree"
FPB_LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

BERT_MODELS = {
    "fin-distilroberta": "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
}

OLLAMA_MODELS = ["llama3.1", "mistral-nemo"]


class LlmModelEnum(str, Enum):
    llama3_1 = "llama3.1"
    mistral_nemo = "mistral-nemo"


class BertModulEnum(str, Enum):
    fin_distilroberta = "fin-distilroberta"


class DataSetEnum(str, Enum):
    fpb = "fpb"
