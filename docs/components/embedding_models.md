# Embedding Models

For the FinMAS system to be successful, the relevant data needs to be sent to the LLM agents
together with the task that the LLM agents are set to perform. For the system to make a best
effort to find the most relevant data for the query, it uses embedding models to convert
textual data into dense numerical representations that we call embeddings.
The main concept is that by storing the data as numerical vectors, the model would be able
to estimate which parts of the data are similar to each other and which part of the data
are very different from each other.

The choice of embedding model can significantly affect the result from the analysis done by
the Multi-Agent system, as the model is responsible for finding the relevant data to sent
to the LLM agent.

## OpenAI embedding model

If an OpenAI model is used, then an embedding model of OpenAI will be used.
The default embedding model is set to [text-embedding-3-small](https://platform.openai.com/docs/guides/embeddings#embedding-models)
which is most cost-effective option from OpenAI. There is also an option to use
`text-embedding-3-large`, and these [embedding models were released early in 2024](https://openai.com/index/new-embedding-models-and-api-updates/).
We generally do not recommend to use the `text-embedding-ada-002` model as it is not as cost-effective.

## HuggingFace embedding models

The user can choose from a pre-defined selection of embedding models that are retrieved
from HuggingFace. When an embedding model is retrieved from HuggingFace it will be downloaded
locally to the directory set in the `embedding_models_dir`.

The default option is to use [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
This model is developed by Beijing Academy of Artificial Intelligence (BAAI) and is a small
English text embedding model. It has a maximum sequence length of 512 tokens and outputs a
vector representation with a dimension of 384.
