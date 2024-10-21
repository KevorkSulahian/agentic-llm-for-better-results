from crewai_tools import LlamaIndexTool

from finmas.crews.model_provider import get_embedding_model, get_llama_index_llm
from finmas.news.news_fetcher import parse_news_to_documents


def get_news_tool(
    records: list[dict],
    llm_provider: str,
    llm_model: str,
    embedding_model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    print("Loading embedding model, creating vector store index and loading LLM model")

    documents = parse_news_to_documents(records, field="content")

    embed_model = get_embedding_model(embedding_model)

    from llama_index.core import VectorStoreIndex

    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    index.storage_context.persist(persist_dir="storage")

    llama_index_llm = get_llama_index_llm(
        llm_provider=llm_provider,
        llm_model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    query_engine = index.as_query_engine(llm=llama_index_llm)

    return LlamaIndexTool.from_query_engine(
        query_engine, name="News Query Tool", description="Use this tool to lookup the latest news"
    )
