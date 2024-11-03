# SEC Filing Parsing

This experiment focuses on parsing an SEC filing in a way such that useful
information can be extracted from the filing and fed to an LLM agent for analysis.
The script uses edgartools to find the latest filing, and downloads the HTML file.
The HTML file is processed and cleaned with BeautifulSoup to make it easier for
an LLM to focus on a specific section of the filing.

The script has mainly been tested with 10-K filings and the following sections:

- Management's Discussion and Analysis
- Risk Factors

## Usage

Activate your virtual environment and run the script like this:

```shell
python main.py COIN NVDA MSFT NFLX --form 10-K --clean
```

This command will parse the latest 10-K filings for these 4 tickers.

## Extracting Table of Contents and sections

There is a span, div or p tag that contains no links and exactly the text:

- INDEX (TSLA, MSFT, AMZN)
- TABLE OF CONTENTS (AAPL, NFLX)
- Table of Contents (NVDA)
- TABLE OF CONTENTS split across

This is used to extract the Table of Contents from the HTML file.
By using the headings from the TOC then the cleaned HTML file is searched
by using BeautifulSoup and extract all the text between two headings.
