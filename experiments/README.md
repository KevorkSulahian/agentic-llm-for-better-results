# Experimentation section

The purpose of this section will be to collaborate and share results of different tests within the team

## Multi-agents frameworks

A selection of recent and active frameworks for LLM multi-agent systems:

- [crewAI](https://github.com/crewAIInc/crewAI)
- [AutoGen](https://microsoft.github.io/autogen/)
- [CAMEL](https://github.com/camel-ai/camel)
- [MetaGPT](https://github.com/geekan/MetaGPT)

### crewAI

crewAI uses GPT-4 as the default model. Through the [LiteLM](https://github.com/BerriAI/litellm) package, many other LLMs can be connected as well.
By using the [ollama](https://ollama.com/) it is possible to run a local LLM together with the crewAI framework.

An illustrative example for how crewAI can be used on a financial task:

https://github.com/crewAIInc/crewAI-examples/tree/main/stock_analysis

## LLM tools

The following tools can be used to work with LLMs locally:

- [ollama](https://ollama.com/)
- [LM studio](https://lmstudio.ai/)

This [LiteLLM page](https://models.litellm.ai/) gives a quick overview of the context length and model price for different LLMs.

This Hugging Face page lists all models that have a warm inference serverless deployment which means they are
accessible to use via the [InferenceClient](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client) from
[huggingface_hub](https://github.com/huggingface/huggingface_hub) Python package.

### ollama

[ollama CLI reference](https://github.com/ollama/ollama?tab=readme-ov-file#cli-reference)

Follow installation instructions for your platform.

Example for pulling and running llama3.1 8b model

```bash
ollama pull llama3.1
ollama run llama3.1
```

An LLM model will automatically stop after 5 minutes. If you want to pre-load it,
send it an empty query using curl

```bash
curl http://localhost:11434/api/generate -d '{"model": "llama3.1"}'
```

To check if an LLM is running:

```bash
ollama ps
```

#### models

A selection of models that can run on 16gb RAM laptop in CPU mode is listed below:

| model                                                     | parameters | size  |
| --------------------------------------------------------- | ---------- | ----- |
| [llama3.1](https://ollama.com/library/llama3.1)           | 8b         | 4.7gb |
| [mistral-nemo](https://ollama.com/library/mistral-nemo)   | 12b        | 7.1gb |
| [mistral-small](https://ollama.com/library/mistral-small) | 22b        | 13gb  |

#### webUIs

Using a webUI can be useful to check the LLM and prompt it before configuring an Agent.
The following UIs can be used:

- [hollama](https://github.com/fmaclen/hollama)
- [open-webui](https://github.com/open-webui/open-webui)

Make sure ollama is running on port 11434 before starting a UI.

##### hollama

Start a hollama server in a docker container:

```bash
docker run --rm -d -p 4173:4173 --name hollama ghcr.io/fmaclen/hollama:latest
```

Access the UI: [http://localhost:4173](http://localhost:4173)

##### open-webui

Start an open-webui server in a docker container:

```bash
docker run -d --network=host -v open-webui:/app/backend/data -e OLLAMA_BASE_URL=http://127.0.0.1:11434 --name open-webui --restart always ghcr.io/open-webui/open-webui:main
```

Access the UI: [http://localhost:3000/](http://localhost:3000/)
