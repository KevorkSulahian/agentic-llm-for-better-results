import os
from pathlib import Path

from finmas.constants import defaults
from finmas.utils.common import get_environment_variable, get_valid_models


def validate_llm_info(llm_provider: str, llm_model: str) -> None:
    if llm_provider not in ["groq", "huggingface", "openai"]:
        raise ValueError(f"Invalid LLM provider: {llm_provider}")
    valid_models = get_valid_models(llm_provider)["id"].tolist()

    if llm_model not in valid_models:
        raise ValueError(f"Invalid LLM model: {llm_model}. Valid models are: {valid_models}")


def get_crewai_llm_model(
    llm_provider: str,
    llm_model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    validate_llm_info(llm_provider, llm_model)
    if llm_provider == "groq":
        config = {"api_key": get_environment_variable("GROQ_API_KEY")}
    elif llm_provider == "huggingface":
        config = {"token": get_environment_variable("HF_TOKEN")}
    elif llm_provider == "openai":
        config = {"api_key": get_environment_variable("OPENAI_API_KEY")}

    crewai_llm_model_name = f"{llm_provider}/{llm_model}" if llm_provider != "openai" else llm_model

    from crewai import LLM

    return LLM(
        model=crewai_llm_model_name,
        temperature=temperature or defaults["llm_temperature"],
        max_tokens=max_tokens or defaults["llm_max_tokens"],
        **config,
    )


def get_llama_index_llm(
    llm_provider: str,
    llm_model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """Get a llama-index compatible LLM model"""
    validate_llm_info(llm_provider, llm_model)
    config = dict(
        temperature=temperature or defaults["llm_temperature"],
        max_tokens=max_tokens or defaults["llm_max_tokens"],
    )
    if llm_provider == "groq":
        from llama_index.llms.groq import Groq

        return Groq(model=llm_model, api_key=get_environment_variable("GROQ_API_KEY"), **config)
    elif llm_provider == "huggingface":
        from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

        return HuggingFaceInferenceAPI(
            model_name=llm_model, token=get_environment_variable("HF_TOKEN"), **config
        )
    elif llm_provider == "openai":
        from llama_index.llms.openai import OpenAI

        return OpenAI(model=llm_model, **config)


def get_hf_embedding_model(model_name: str | None = None):
    """Get a HuggingFace embedding model"""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    cache_dir = Path(defaults["embedding_models_dir"]).absolute()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variable for Hugging Face to use our cache directory
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    return HuggingFaceEmbedding(
        model_name=model_name or defaults["hf_embedding_model"],
        device="cpu",
        cache_folder=str(cache_dir),
    )


def get_embedding_model(llm_provider: str, embedding_model: str):
    """Get the embedding model based on the LLM provider"""
    if llm_provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(model=embedding_model)
    else:
        # If openai is not used, then fetch a HuggingFace embedding model
        return get_hf_embedding_model(embedding_model)
