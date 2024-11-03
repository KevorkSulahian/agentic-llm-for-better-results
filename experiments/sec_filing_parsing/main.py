from typing import List, Optional

import typer
from rich import print
from sec_parser import SECFilingParser
from typing_extensions import Annotated

DEFAULT_FORM = "10-K"
HEADINGS_LIST = ["Discussion and Analysis", "Risk Factors"]


def main(
    tickers: Annotated[List[str], typer.Argument(help="Tickers to fetch filing for")],
    form: Annotated[str, typer.Option(help="Form to fetch")] = "10-K",
    clean: Annotated[bool, typer.Option(help="Clean existing filings")] = False,
    heading_text: Annotated[Optional[str], typer.Option(help="Heading text to search for")] = None,
):
    for ticker in tickers:
        print(f"Fetching filing for {ticker} with form type {form}")
        parser = SECFilingParser(ticker=ticker, form_type=form, clean=clean)
        parser.parse_latest_filing()
        toc = parser.extract_table_of_contents_from_html()

        print(toc)

        for i, heading in enumerate(toc[:-1]):
            for heading_text in HEADINGS_LIST:
                if heading_text in heading:
                    next_heading = toc[i + 1]
                    print(f"Found heading: {heading} with next heading: {next_heading}")
                    parser.extract_section_from_html(heading, next_heading)
                    break


if __name__ == "__main__":
    typer.run(main)
