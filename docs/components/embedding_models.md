# Embedding Models

For the FinMAS system to be successful the relevant data needs to be sent to the LLM agents
together with the task that the LLM agents is set to perform. For the system to make a best
effort to find the most relevant data for the query, it uses embedding models to convert
textual data into dense numerical representations that we call embeddings.
The main concept is that by storing the data as numerical vectors the model would be able
to estimate which parts of the data are similar to each other and which part of the data
are very different from each other.

The user can choose from a pre-defined selection of embedding models that are retrieved
from HuggingFace. When an embedding model is retrieved from HuggingFace it will be downloaded
locally to the directory set in the `embedding_models_dir`. If an OpenAI model is used,
then the embedding model of OpenAI will be used.

The choice of embedding model can significantly affect the result from the analysis done by
the Multi-agent system, as the model is responsible for finding the relevant data to sent
to the LLM agent.

## HuggingFace embedding models

TODO: explain the main features of the most popular models

## OpenAI embedding model

When using the OpenAI embedding model ada-002 it will consume tokens during the embedding
process. For large datasets, the token consumption can be somewhat high and this should be
taken into consideration when using this model.
