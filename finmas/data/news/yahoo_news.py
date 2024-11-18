import datetime as dt
import time

import feedparser

from finmas.data.news.news_fetcher import NewsFetcherBase


class YahooFinanceNewsFetcher(NewsFetcherBase):
    """
    Fetch news from Yahoo Finance RSS feed.
    Since the RSS feed does not contain a cleaned text content field, it is not used
    in any crew or analysis tasks.
    """

    def get_news(
        self, ticker: str, start: dt.datetime | None = None, end: dt.datetime | None = None
    ) -> list[dict]:
        rss_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
        feed = feedparser.parse(rss_url)

        records = []
        for entry in feed.entries:
            published = dt.datetime.fromtimestamp(time.mktime(entry.published_parsed))
            if (start and published < start) or (end and published > end):
                continue
            record = dict(
                title=entry.title,
                link=entry.link,
                id=entry.id,
                published=published,
                summary=entry.summary,
            )
            records.append(record)

        return records
