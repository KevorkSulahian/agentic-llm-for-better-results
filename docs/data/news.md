# News

## Benzinga News

This News source is accessed through the Alpaca API. It contains historical news articles from Benzinga.
The processing of the news articles is done in the [BenzingaNewsFecther](https://github.com/KevorkSulahian/agentic-llm-for-better-results/blob/main/finmas/data/news/benzinga_news.py) class.
The news articles are converted from HTML to Markdown using the [html-to-markdown](https://pypi.org/project/html-to-markdown/) package.
The news articles are filtered so that only news articles that are likely to be relevant
to the chosen ticker are used.

The following rules are used to filter out news articles:

- News articles about options
- News articles with more than 5 tickers tagged to the article.
