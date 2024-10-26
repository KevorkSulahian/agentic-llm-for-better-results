import datetime as dt
from pathlib import Path

import pandas as pd
from edgar import Company, set_identity
from edgar.entities import EntityFilings

from finmas.cache_config import cache
from finmas.constants import SEC_FILINGS_COLS, defaults

set_identity("John Doe john.doe@example.com")


@cache.memoize(expire=dt.timedelta(days=1).total_seconds())
def get_sec_filings(ticker: str, latest: int = 10) -> EntityFilings:
    """Use edgartools to get the latest SEC filings"""
    return Company(ticker).get_filings(form=["10-K", "10-Q"]).latest(latest)


def filings_to_df(filings: EntityFilings) -> pd.DataFrame:
    """Convert filings to a DataFrame.
    Adds a column with the URL.
    Converts dates to strings.
    """
    df = filings.to_pandas()
    df["link"] = [f.document.url for f in filings]
    df = df[SEC_FILINGS_COLS]
    for col in ["filing_date", "reportDate"]:
        df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
    return df


def download_filings(filings: EntityFilings) -> None:
    """Download the SEC filings to local folder

    Args:
        filings: Filings fetched from edgartools
    """

    filings_base_dir = Path(defaults["filings_dir"])

    for filing in filings:
        ticker = filing.get_entity().tickers[0]
        filing_dir = filings_base_dir / ticker / filing.form
        filing_dir.mkdir(parents=True, exist_ok=True)
        output_file = filing_dir / filing.document.document
        if not output_file.exists():
            filing.document.download(path=filing_dir)
