import time

from finmas.constants import defaults
from finmas.crews.model_provider import get_embedding_model, get_llama_index_llm
from finmas.crews.utils import IndexCreationMetrics
from finmas.data.news.news_fetcher import parse_news_to_documents
from finmas.logger import get_logger
from finmas.utils.common import get_text_content_file, get_vector_store_index_dir

logger = get_logger(__name__)


def get_news_query_engine(
    ticker: str,
    records: list[dict],
    llm_provider: str,
    llm_model: str,
    embedding_model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    similarity_top_k: int | None = None,
):
    documents = parse_news_to_documents(records, field="markdown_content")

    if defaults["save_text_content"]:
        text_content = "\n\n".join(doc.text for doc in documents)
        file_path = get_text_content_file(ticker=ticker, data_type="news")
        file_path.write_text(text_content, encoding="utf-8")
        logger.info(f"Saved news text content to {file_path}")

    embed_model = get_embedding_model(llm_provider, embedding_model)

    from llama_index.core import Settings, VectorStoreIndex
    from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

    start = time.time()
    token_counter = TokenCountingHandler()
    index = VectorStoreIndex.from_documents(
        documents, embed_model=embed_model, callback_manager=CallbackManager([token_counter])
    )
    index.storage_context.persist(persist_dir=get_vector_store_index_dir(ticker, "news"))
    time_spent = round(time.time() - start, 2)
    logger.info(f"News index created in {time_spent}s with {len(documents)} documents")

    text_length = sum([len(doc.text) for doc in documents])

    metrics = IndexCreationMetrics(
        embedding_model=embedding_model,
        time_spent=time_spent,
        num_nodes=len(index.index_struct.nodes_dict.keys()),
        text_length=text_length,
        chunk_size=Settings.chunk_size,
        chunk_overlap=Settings.chunk_overlap,
        total_embedding_token_count=token_counter.total_embedding_token_count,
    )

    llama_index_llm = get_llama_index_llm(
        llm_provider=llm_provider,
        llm_model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    query_engine = index.as_query_engine(llm=llama_index_llm, similarity_top_k=similarity_top_k)

    return (query_engine, metrics)
