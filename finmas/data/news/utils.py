from finmas.data.news.benzinga_news import BenzingaNewsFetcher
from finmas.data.news.news_fetcher import NewsFetcherBase
from finmas.data.news.yahoo_news import YahooFinanceNewsFetcher


def get_news_fetcher(source: str) -> NewsFetcherBase:
    """
    Factory function to provide a news fetcher based on the provider string.
    """
    source = source.lower().replace(" ", "_")
    if source == "yahoo_finance":
        return YahooFinanceNewsFetcher()
    elif source == "benzinga":
        return BenzingaNewsFetcher()
    else:
        raise ValueError(
            f"Invalid News source: {source}. Choose either 'yahoo_finance' or 'benzinga'."
        )
