import datetime as dt
import os
import re
import sys
import time

import feedparser
from alpaca.data.historical.news import NewsClient
from alpaca.data.models import NewsSet, News
from alpaca.data.requests import NewsRequest
from bs4 import BeautifulSoup
from html_to_markdown import convert_to_markdown

from finmas.news.benzinga_news import HEADLINE_IGNORE_LIST, get_benzinga_content_text

BENZINGA_NEWS_LIMIT = 50


def condense_newline(text):
    return "\n".join([p for p in re.split("\n|\r", text) if len(p) > 0])


class NewsFetcher:
    def get_news(
        self,
        ticker: str,
        source: str,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
    ) -> list[dict]:
        source = source.lower().replace(" ", "_")
        match source:
            case "yahoo_finance":
                print(f"Fetching Yahoo Finance news for {ticker}")
                return self.get_yahoo_finance_news(ticker)
            case "benzinga":
                print(f"Fetching Benzinga news for {ticker}")
                return self.get_benzinga_news(ticker, start, end)
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

    def get_benzinga_news(
        self, ticker: str, start: dt.datetime | None = None, end: dt.datetime | None = None
    ) -> list[dict]:
        """Getting Benzing News using the Alpaca Historical News articles API

        Ref: https://docs.alpaca.markets/reference/news-3
        """
        assert os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_API_SECRET")

        client = NewsClient(
            api_key=os.getenv("ALPACA_API_KEY"), secret_key=os.getenv("ALPACA_API_SECRET")
        )
        start = start or dt.datetime.now() - dt.timedelta(days=14)

        page_token = None
        news_list: list[News] = []
        while True:
            request = NewsRequest(
                symbols=ticker,
                start=start,
                end=end or dt.datetime.now(),
                include_content=True,
                exclude_contentless=True,
                limit=BENZINGA_NEWS_LIMIT,
                page_token=page_token,
            )
            news_set = client.get_news(request_params=request)
            assert isinstance(news_set, NewsSet)
            news_list.extend(news_set.data["news"])

            page_token = news_set.next_page_token
            if page_token is None or len(news_set.data["news"]) < BENZINGA_NEWS_LIMIT:
                break

        records = []

        # Filter out news items from a fixed headline ignore list
        news_items = [
            news
            for news in news_list
            if not any(
                ignore_headline.lower() in news.headline.lower()
                for ignore_headline in HEADLINE_IGNORE_LIST
            )
        ]

        for news in news_items:
            if news.content is None or len(news.content) == 0:
                continue
            soup = BeautifulSoup(news.content, "html.parser")
            record = dict(
                title=news.headline,
                published=news.updated_at,
                author=news.author,
                num_symbols=len(news.symbols),
                symbols=news.symbols,
                link=news.url,
                id=news.id,
                summary=news.summary,
                content=news.content,
                markdown_content=convert_to_markdown(soup),
                text=get_benzinga_content_text(news.content),
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
