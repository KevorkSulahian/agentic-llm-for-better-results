import datetime as dt
import os
import re
import unicodedata

from alpaca.data.historical.news import NewsClient
from alpaca.data.models import News, NewsSet
from alpaca.data.requests import NewsRequest
from bs4 import BeautifulSoup
from html_to_markdown import convert_to_markdown

from finmas.cache_config import cache
from finmas.data.news.news_fetcher import NewsFetcherBase

BENZINGA_NEWS_LIMIT = 50


SENTENCES_IGNORE_LIST = [
    "disclaimer",
    "benzinga",
    "photo by",
    "see also",
    "shutterstock",
    "read next",
    "click here",
]
HEADLINE_IGNORE_LIST = ["market clubhouse"]


def condense_newline(text):
    """Helper to reduce consecutive newlines into single newline"""
    return "\n".join([p for p in re.split("\n|\r", text) if len(p) > 0])


class BenzingaNewsFetcher(NewsFetcherBase):
    @cache.memoize(expire=dt.timedelta(days=1).total_seconds())
    def get_news(
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

            # Decompose all tables
            for table in soup.find_all("table"):
                table.decompose()

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
                markdown_content=convert_to_markdown(
                    source=soup,
                    autolinks=False,
                    escape_misc=False,
                    heading_style="atx",
                    strip=["a", "table", "thead", "tbody", "td", "tr", "th", "img"],
                    wrap=True,
                    wrap_width=100,
                ),
                text=self.get_benzinga_content_text(news.content),
            )
            records.append(record)

        return records

    @staticmethod
    def get_benzinga_content_text(html_content: str, exclude_tables: bool = True) -> str:
        """Parses the HTML from a news content from Benzinga news source and returns the text.

        This method extracts a clean text from the HTML content.

        Args:
            html_content: The HTML content of the news article.
            exclude_tables: Whether to exclude tables from the text. Defaults to True.
        """
        soup = BeautifulSoup(html_content, "html.parser")

        if exclude_tables:
            for table in soup.find_all("table"):
                table.decompose()

        TAGS = ["p", "ul"]
        filtered_text_list = []
        for tag in soup.find_all(TAGS):
            text = condense_newline(tag.text)
            text = unicodedata.normalize("NFKD", text)  # Replace \xa0 with space
            # text = re.sub(r"[\n\r]", "", text)  # Remove all newlines
            text = re.sub(r"\t", " ", text)  # Replace all tab characters with space
            text = re.sub(r"\s+", " ", text).strip()  # Condense whitespace

            if len(text) == 0:
                continue

            text = re.sub(
                r"\.\s*([A-Z])", r". \1", text
            )  # Ensure that there is a space after period.
            sentences = re.split(r"(?<=[.!?])\s+", text)
            filtered_sentences = [
                sentence
                for sentence in sentences
                if not any(
                    ignore_word.lower() in sentence.lower() for ignore_word in SENTENCES_IGNORE_LIST
                )
            ]
            if filtered_sentences and not filtered_sentences[-1].endswith("."):
                filtered_sentences[-1] += "."
            filtered_text_list.extend(filtered_sentences)

        return " ".join(filtered_text_list)
