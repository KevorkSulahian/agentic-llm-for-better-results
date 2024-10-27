import os
from edgar import Filing
from pathlib import Path
from constants import FILINGS_DIR


def get_environment_variable(key: str) -> str:
    """Get the value from the environment variables"""
    try:
        return os.environ[key]
    except KeyError:
        raise ValueError(
            f"{key} not found in environment variables. Please set {key} in the .env file."
        )


def download_filing(ticker: str, filing: Filing) -> Path:
    """Downloads the filing and returns the path to the file"""
    filingsdir = Path(FILINGS_DIR) / ticker / filing.form
    filingsdir.mkdir(parents=True, exist_ok=True)
    file_path = filingsdir / filing.document.document
    if not file_path.exists():
        filing.document.download(path=filingsdir)
    return file_path
