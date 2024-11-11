from enum import StrEnum
from pathlib import Path
from pprint import pprint
from typing import Optional

import pandas as pd
import typer
from rich import print
from typing_extensions import Annotated

from finmas.constants import defaults
from finmas.data.market.alpha_vantage import get_fundamental_data
from finmas.data.news.benzinga_news import BenzingaNewsFetcher
from finmas.utils.common import to_datetime

app = typer.Typer()


class FrequencyEnum(StrEnum):
    ANNUAL = "Annual"
    QUARTERLY = "Quarterly"


@app.command()
def download_fundamentals(
    ticker: Annotated[str, typer.Argument(help="Stock ticker to load fundamentals for")],
    freq: Annotated[
        FrequencyEnum, typer.Option(help="Annual or Quarterly")
    ] = FrequencyEnum.QUARTERLY,
) -> None:
    """
    Download fundamental data for a given stock ticker with Alpha Vantage.

    The data is saved as a CSV file in the data directory with the ticker as subfolder.
    Both income statement and balance sheet data are downloaded.
    """
    fundamentals_dir = Path(defaults["data_dir"]) / "fundamentals" / ticker
    fundamentals_dir.mkdir(parents=True, exist_ok=True)
    for type in ["income", "balance"]:
        df = get_fundamental_data(ticker, type, freq.value)
        if df.empty:
            print(f"No data found for ticker '{ticker}'")
            return

        file_path = fundamentals_dir / f"{type}_{freq.value.lower()}.csv"
        if not file_path.exists():
            df.to_csv(file_path)
            print(f"Fundamental data for '{ticker}' stored in '{fundamentals_dir}'")


@app.command()
def download_news(
    ticker: Annotated[str, typer.Argument(help="Stock ticker to load news for")],
    start: Annotated[Optional[str], typer.Option(help="Start date for news")] = None,
    end: Annotated[Optional[str], typer.Option(help="End date for news")] = None,
) -> None:
    """
    Download news for a given stock ticker from Alpaca News API (Benzinga News Source).

    The data is saved as a CSV file in the data directory with the ticker as subfolder.
    """
    news_dir = Path(defaults["data_dir"]) / "news" / ticker
    news_dir.mkdir(parents=True, exist_ok=True)

    news_fetcher = BenzingaNewsFetcher()

    if start is None:
        start = defaults["news_start_date"]
    if end is None:
        end = defaults["news_end_date"]

    news_records = news_fetcher.get_news(ticker, start=to_datetime(start), end=to_datetime(end))
    if not news_records:
        print(f"No news found for ticker '{ticker}'")
        return

    df = pd.DataFrame(news_records)
    df["published"] = df["published"].dt.strftime("%Y-%m-%d")

    print(df)


@app.command()
def clean_up():
    """
    Clean up data folders for news
    """
    print("Cleaning news data...")
    # News
    if typer.confirm("Do you want to clean the news data?", default=False):
        news_dir = Path(defaults["data_dir"]) / "benzinga_news"
        for folder in news_dir.iterdir():
            if folder.is_dir():
                files = [str(f) for f in folder.iterdir()]
                pprint(files)
                if typer.confirm(f"Delete folder '{folder}'?", default=False):
                    for file in folder.iterdir():
                        file.unlink()
                    folder.rmdir()
                    print("Deleted folder:", folder)


if __name__ == "__main__":
    app()
