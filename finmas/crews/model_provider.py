from finmas.constants import defaults
from finmas.utils import get_environment_variable, get_valid_models


def validate_llm_info(llm_provider: str, llm_model: str) -> None:
    if llm_provider not in ["groq", "huggingface", "openai"]:
        raise ValueError(f"Invalid LLM provider: {llm_provider}")
    valid_models = get_valid_models(llm_provider)["id"].tolist()

    if llm_model not in valid_models:
        raise ValueError(f"Invalid LLM model: {llm_model}. Valid models are: {valid_models}")


def get_crewai_llm_model(llm_provider: str, llm_model: str):
    validate_llm_info(llm_provider, llm_model)
    if llm_provider == "groq":
        config = {"api_key": get_environment_variable("GROQ_API_KEY")}
    elif llm_provider == "huggingface":
        config = {"token": get_environment_variable("HF_TOKEN")}

    crewai_llm_model_name = f"{llm_provider}/{llm_model}" if llm_provider != "openai" else llm_model

    from crewai import LLM

    return LLM(
        model=crewai_llm_model_name,
        temperature=defaults["llm_temperature"],
        max_tokens=defaults["llm_max_tokens"],
        **config,
    )


def get_llama_index_llm(llm_provider: str, llm_model: str):
    """Get a llama-index compatible LLM model"""
    validate_llm_info(llm_provider, llm_model)
    config = dict(temperature=defaults["temperature"], max_tokens=defaults["max_tokens"])
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
