import datetime as dt
import os
import time

import feedparser
import pandas as pd
from alpaca.data.historical.news import NewsClient
from alpaca.data.models import NewsSet
from alpaca.data.requests import NewsRequest
from constants import ALPACA_NEWS_START_DATE, NewsSourceEnum


class NewsFetcher:
    def __init__(self, source: NewsSourceEnum):
        self.source = source

    def get_news(self, ticker: str) -> list[dict]:
        return self._get_news(ticker, self.source)

    def _get_news(self, ticker: str, source: NewsSourceEnum) -> list[dict]:
        match source:
            case NewsSourceEnum.yahoo_finance:
                print(f"Fetching Yahoo Finance news for {ticker}")
                return self.get_yahoo_finance_news(ticker)
            case NewsSourceEnum.benzinga:
                print(f"Fetching Benzinga news for {ticker}")
                return self.get_benzinga_news(ticker)
            case _:
                raise ValueError("Invalid news source")

    def get_yahoo_finance_news(self, ticker: str) -> list[dict]:
        rss_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
        feed = feedparser.parse(rss_url)

        records = []
        for entry in feed.entries:
            record = dict(
                title=entry.title,
                link=entry.link,
                id=entry.id,
                published=dt.datetime.fromtimestamp(time.mktime(entry.published_parsed)),
                summary=entry.summary,
            )
            records.append(record)

        return records

    def get_benzinga_news(self, ticker: str) -> list[dict]:
        assert os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_API_SECRET")

        client = NewsClient(
            api_key=os.getenv("ALPACA_API_KEY"), secret_key=os.getenv("ALPACA_API_SECRET")
        )

        request = NewsRequest(
            symbols=ticker,
            start=dt.datetime.fromisoformat(ALPACA_NEWS_START_DATE),
            end=dt.datetime.now(),
            include_content=True,
        )
        news_set = client.get_news(request_params=request)
        assert isinstance(news_set, NewsSet)

        records = []
        for news in news_set.data["news"]:
            record = dict(
                title=news.headline,
                link=news.url,
                id=news.id,
                published=news.updated_at,
                summary=news.summary,
                # TODO: Add content, symbols, author fields
            )
            records.append(record)

        return records


def parse_news_to_dataframe(news: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(news)
    return df


def parse_news_to_documents(news: list[dict]):
    from llama_index.core import Document

    documents = []
    for item in news:
        text = (
            f"Title: {item['title']}\n"
            f"Published: {item['published'].date().isoformat()}\n"
            f"Summary: {item['summary']}"
        )
        doc = Document(text=text)
        documents.append(doc)

    return documents
