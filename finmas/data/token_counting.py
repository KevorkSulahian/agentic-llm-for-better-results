import tiktoken
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

from finmas.constants import defaults
from finmas.logger import get_logger

logger = get_logger(__name__)

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-4o").encode,
    verbose=True,
    logger=logger,
)
Settings.callback_manager = CallbackManager([token_counter])


def get_token_counter_as_string(llm_model: str | None = None) -> str:
    """Returns a string representation of the token counter."""
    output = (
        f"Total tokens: {token_counter.total_llm_token_count}  \n"
        f"Prompt tokens: {token_counter.prompt_llm_token_count}  \n"
        f"Completion tokens: {token_counter.completion_llm_token_count}  \n"
    )
    if llm_model and llm_model in defaults["llm_model_cost"]:
        cost = (
            defaults["llm_model_cost"][llm_model]["input_cost"]
            * token_counter.prompt_llm_token_count
            + defaults["llm_model_cost"][llm_model]["output_cost"]
            * token_counter.completion_llm_token_count
        )
        cost_str = f"{cost:.15f}".rstrip("0")
        output += f"Estimated LLM Model cost for total tokens: ${cost_str}  \n"
    return output
