# Sentiment Analysis with LLM as an Agent

For using these scripts the virtual environment for this project needs to be activated.
The scripts should be run from their own folder, not from the project folder.

## sa_agent.py

This is a CLI script that shows how a single LLM agent can be used for sentiment analysis.

Currently, only the FPB dataset is supported.

1. Install [ollama](https://ollama.com/) and make sure it is running
2. Run [llama3.1](https://ollama.com/library/llama3.1) with

```bash
ollama run llama3.1
```

3. Run the CLI script with the following command for running SA on 10 entries in the FPB dataset:

```bash
# Usage: python sa_agent.py <model> <dataset> --limit <entries>
python sa_agent.py llama3.1 fpb --limit 10
```

## roberta.py

Similar CLI script that shows how a BERT model that is fine tuned for financial sentiment analysis
can be used. This script is useful to use for comparison against LLM model for speed and efficiency
of a light and specialized BERT model.

```bash
# python roberta.py <model> <dataset> --limit <entries --output <filename> --show
python roberta.py fin-distilroberta fpb
```

The script will run the following HuggingFace hosted BERT model:

[mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis)
