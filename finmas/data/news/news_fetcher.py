import abc
import datetime as dt
import sys


class NewsFetcherBase(abc.ABC):
    """Base class for news fetchers"""

    @abc.abstractmethod
    def get_news(
        self, ticker: str, start: dt.datetime | None = None, end: dt.datetime | None = None
    ) -> list[dict]:
        """Get the news from the data source for the given ticker and according to start and end dates."""
        pass


def parse_news_to_documents(articles: list[dict], field: str = "summary") -> list:
    """
    Parses news records to LlamaIndex Documents.
    For each news article, a simple text string is created from the fields title, published and the content field specified by `field`.
    The LlamaIndex Document is created from the text string.
    """
    from llama_index.core import Document

    documents = []

    for item in articles:
        text = f"Title: {item['title']}\nPublished: {item['published'].date().isoformat()}\n"
        try:
            text += f"{item[field]}"
        except KeyError as e:
            print(f"The news item does not contain the {field} field. Error: {e}")
            sys.exit(1)

        doc = Document(text=text)
        documents.append(doc)

    return documents
