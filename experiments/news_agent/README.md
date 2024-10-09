# News

## query_news.py

This is a simple CLI script that can setup a `llama-index` query engine from a news source,
and use this engine to query the news.
It uses the VectorStoreIndex from `llama-index` to construct a store from the news.

### Usage

```bash
python query_news.py <ticker> --source <benzinga,yahoo> --llm-name <llm-name> --llm-provider <huggingface>
```

Example:

```bash
python query_news.py AAPL --source benzinga --llm-name llama-3.2-3B-Instruct
```

For seeing a list of available values to options and description of the script:

```bash
python query_news.py --help
```

For a list of available HuggingFace models see:

https://huggingface.co/models?inference=warm&sort=trending&pipeline_tag=text-generation

## Data sources

### Yahoo Finance

The Yahoo Finance news are fetched from their RSS feed.

```
https://finance.yahoo.com/rss/headline?s={ticker}
```

### Benzinga / Alpaca API

For fetching news from Benzinga, the [alpaca-py](https://pypi.org/project/alpaca-py/) client is used.
The environment variables `ALPACA_API_KEY` and `ALPACA_API_SECRET` needs to be set.
