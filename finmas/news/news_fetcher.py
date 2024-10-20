import abc
import datetime as dt
import sys


class NewsFetcherBase(abc.ABC):
    @abc.abstractmethod
    def get_news(
        self, ticker: str, start: dt.datetime | None = None, end: dt.datetime | None = None
    ) -> list[dict]:
        pass


def parse_news_to_documents(records: list[dict], field: str = "summary"):
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
