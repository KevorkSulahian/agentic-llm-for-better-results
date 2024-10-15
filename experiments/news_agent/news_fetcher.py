import datetime as dt
import os
import time

import feedparser
from alpaca.data.historical.news import NewsClient
from alpaca.data.models import NewsSet
from alpaca.data.requests import NewsRequest
from benzinga_news import HEADLINE_IGNORE_LIST, get_benzinga_content_text
from constants import ALPACA_NEWS_START_DATE, NewsSourceEnum
import sys


class NewsFetcher:
    def get_news(self, ticker: str, source: NewsSourceEnum) -> list[dict]:
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

        # Filter out news items from a fixed headline ignore list
        news_items = [
            news
            for news in news_set.data["news"]
            if not any(
                ignore_headline.lower() in news.headline.lower()
                for ignore_headline in HEADLINE_IGNORE_LIST
            )
        ]

        for news in news_items:
            record = dict(
                title=news.headline,
                link=news.url,
                id=news.id,
                published=news.updated_at,
                summary=news.summary,
                content=get_benzinga_content_text(news.content),
                num_symbols=len(news.symbols),
                # TODO: Add author fields
            )
            records.append(record)

        return records

    def parse_news_to_documents(self, records: list[dict], field: str = "summary"):
        from llama_index.core import Document

        documents = []

        for item in records:
            text = f"Title: {item['title']}\nPublished: {item['published'].date().isoformat()}\n"
            try:
                text += f"{item[field]}"
            except KeyError as e:
                print(f"The news item does not contain a content field. Error: {e}")
                sys.exit(1)

            doc = Document(text=text)
            documents.append(doc)

        return documents
