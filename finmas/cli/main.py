from enum import StrEnum
from pathlib import Path

import typer
from rich import print
from typing_extensions import Annotated

from finmas.constants import defaults
from finmas.data.market.alpha_vantage import get_fundamental_data

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

        filename = f"{type}_{freq.value.lower()}.csv"
        df.to_csv(fundamentals_dir / filename)

    print(f"Fundamental data for '{ticker}' stored in '{fundamentals_dir}'")


@app.command()
def download_news(
    ticker: Annotated[str, typer.Argument(help="Stock ticker to load news for")],
) -> None:
    """
    Download news for a given stock ticker from Alpaca News API (Benzinga News Source).

    The data is saved as a CSV file in the data directory with the ticker as subfolder.
    """
    news_dir = Path(defaults["data_dir"]) / "news" / ticker
    news_dir.mkdir(parents=True, exist_ok=True)
    return


if __name__ == "__main__":
    app()
