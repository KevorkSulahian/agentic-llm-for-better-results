# Sentiment Analysis with LLM as an Agent

## sa_agent.py

This is a CLI script that shows how a single LLM agent can be used for sentiment analysis.

Currently, only the FPB dataset is supported.

1. Install [ollama](https://ollama.com/) and make sure it is running
2. Run [llama3.1](https://ollama.com/library/llama3.1) with

```bash
ollama run llama3.1
```

3. Activate the virtual environment for python
4. Run the CLI script with the following command for running SA on 10 entries in the FPB dataset:

```bash
python sa_agent.py llama3.1 fpb --limit 10
```
