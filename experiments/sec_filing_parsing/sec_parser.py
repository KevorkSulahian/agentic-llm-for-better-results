import logging
import re
import time
import unicodedata
from pathlib import Path

import html2text
from bs4 import BeautifulSoup, Tag
from datamule import parse_textual_filing
from datamule.filing_viewer.filing_viewer import json_to_html
from edgar import Company, Filing, find, set_identity

FILINGS_DIR = "filings"

set_identity("John Doe john.doe@example.com")

logger = logging.getLogger(__name__)

SECTION_FILENAME_MAP = {
    "Discussion and Analysis": "mda",
    "Risk Factors": "risk_factors",
}


class SECFilingParser:
    """Class to parse SEC filings for a given company and form type

    The class provides methods to download the latest filing for a given company and form type,
    parse the filing as HTML and Markdown, and extract the table of contents and sections from the filing.
    """

    def __init__(self, ticker: str, form_type: str, clean: bool = False) -> None:
        """Initialize the SECFilingParser class

        Args:
            ticker: Ticker that exists in the EDGAR database
            form_type: Form type to fetch, e.g. 10-K, 10-Q
            clean: Whether to clean existing filings before parsing
        """
        self.ticker = ticker
        self.company = Company(self.ticker)
        if self.company is None:
            raise ValueError(f"Could not find company with ticker {self.ticker}")
        self.form_type = form_type
        self.filing_path = Path(FILINGS_DIR) / self.ticker / form_type
        self.filing_path.mkdir(parents=True, exist_ok=True)
        if clean:
            for file in self.filing_path.glob("*"):
                file.unlink()

    def parse_latest_filing(self) -> None:
        """Parse the latest filing for the given company and form type"""
        filing = self.company.get_filings(form=self.form_type).latest(1)
        self.parse_filing(filing=filing)

    def parse_filing(
        self, *, filing: Filing | None = None, accession_number: str | None = None
    ) -> None:
        """Parse the given filing as HTML and Markdown

        Args:
            filing: Filing object to parse
            accession_number: Accession number of the filing to parse
        """
        if bool(filing) == bool(accession_number):
            raise ValueError("Must provide either filing or accession number, but not both")
        if accession_number:
            filing = find(accession_number)
        if not filing:
            raise ValueError("Could not find filing")

        self.parse_filing_as_html(filing)
        self.parse_filing_as_markdown(filing)

    def parse_filing_as_html(self, filing: Filing) -> None:
        """Parse filing as HTML and clean the HTML content"""
        self.filing_html_path = self.filing_path / filing.document.document
        if not self.filing_html_path.exists():
            filing.document.download(path=self.filing_html_path)
        self._clean_html_filing()

    def parse_filing_as_markdown(self, filing: Filing) -> None:
        """Parse filing as Markdown using the datamule endpoint

        This method strips the HTML content of any tables and inline XBRL tags before converting it to Markdown.

        Methodology:
            1. parse_textual_filing function from datamule is used to parse the filing as JSON.
            2. json_to_html function from datamule is used to convert the JSON content to HTML.
            3. html2text is used to convert the HTML content to plain text.

        Args:
            filing: Filing object to parse
        """
        start = time.time()
        self.filing_markdown_path = (self.filing_path / filing.document.document).with_suffix(".md")
        if self.filing_markdown_path.exists():
            logger.info(f"SEC Filing already stored as Markdown at: {self.filing_markdown_path}")
            return

        start = time.time()
        json_content = parse_textual_filing(filing.document.url, return_type="json")
        html_content = json_to_html(json_content)

        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        h.ignore_emphasis = True

        text_content = h.handle(html_content)

        # The text content from the HTML conversion includes headers
        # so we remove the leading '#' characters to avoid uncessary headers
        text_content = re.sub(r"^\s*#+\s*", "", text_content, flags=re.MULTILINE)
        text_content = re.sub(r"\u2022\s?", "- ", text_content)

        self.filing_markdown_path.write_text(text_content, encoding="utf-8")

        logger.info(f"SEC Filing stored as Markdown at: {self.filing_markdown_path}")
        logger.info(f"Time spent to parse and download filing: {time.time() - start:.1f} seconds")

    def _clean_html_filing(self) -> None:
        """Clean the HTML content of the filing

        The cleaning process is specifically developed for SEC 10-K filings.
        It is developed based on a sample of 10-K filings and may not work for all filings.

        Methodology:
        1. Normalize unicode characters.
        2. XBRL tags are removed as they are not relevant for the text content.
        3. Inline styles are removed from all tags.
        4. Span tags are removed as they are not relevant for the text content.
        5. The first page of the filing is removed by removing all tags before the TABLE OF CONTENTS or INDEX.
        6. Empty div tags with an id attribute are removed.
        7. Horizontal line tags <hr> are removed along with the tags before and after them.
        8. A collection of regex substitutions are performed to clean up the HTML content.
        """

        start = time.time()
        with open(self.filing_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        html_content = unicodedata.normalize("NFKD", html_content)  # Normalize unicode characters

        soup = BeautifulSoup(html_content, "html.parser")

        # ix: tags represent inline XBRL tags, which we can remove
        for tag in soup.find_all(re.compile(r"^(ix|xbrli):")):
            tag.unwrap()

        for tag in soup.find_all(True):
            del tag["style"]

        # Fix for NVDA filings where there is a Table of Contents link in the text
        # for tag in soup.find_all(lambda tag: tag.name in ["div", "p"] and tag.find("a", string="Table of Contents")):
        #     a_tag = tag.find("a", string="Table of Contents")
        #     # Replace content with TOC text
        #     if a_tag:
        #         a_tag.string = "TOC"

        toc_div = soup.find(
            lambda tag: tag.name in ["div", "p", "span"]
            and tag.text.strip() in ["TABLE OF CONTENTS", "INDEX", "Table of Contents"]
            and not tag.find("a")
        )

        if toc_div:
            # Remove all tags before the previous <hr> tag as they represent the first page
            hr_tag = toc_div.find_previous("hr")
            for sibling in hr_tag.find_previous_siblings(True):
                sibling.decompose()

        for tag in soup.find_all("span"):
            tag.unwrap()
        # Remove any div tags that are empty and have an id attribute
        for div_tag in soup.find_all("div", id=True):
            if not div_tag.text.strip():
                div_tag.decompose()

        # Remove all hr (horizontal line) tags and the single tag before and after it
        for hr_tag in soup.find_all("hr"):
            prev_tag = hr_tag.find_previous_sibling()
            if prev_tag:
                prev_tag.decompose()
            next_tag = hr_tag.find_next_sibling()
            if next_tag:
                next_tag.decompose()
            hr_tag.decompose()

        cleaned_html = str(soup)
        # Perform a collection of regex substitutions to clean up the HTML
        cleaned_html = re.sub(r"\n{3,}", "\n\n", cleaned_html)  # Remove extra newlines

        cleaned_html = re.sub(r"[\u00a0\xa0]", " ", cleaned_html)  # Non-breaking space
        cleaned_html = re.sub(r"\u2022\s?", "- ", cleaned_html)  # Bullet point
        cleaned_html = re.sub(r"[\u2018\u2019]", "'", cleaned_html)  # Left and right single quotes
        cleaned_html = re.sub(r"[\u201C\u201D]", '"', cleaned_html)  # Left and right double quotes
        cleaned_html = re.sub(r"[\u2013\u2014]", "-", cleaned_html)  # En dash and em dash

        self.filing_cleaned_html_path = self.filing_html_path.with_suffix(".refined.html")
        self.filing_cleaned_html_path.write_text(cleaned_html, encoding="utf-8")

        logger.info(f"Refined HTML stored at: {self.filing_html_path}")
        logger.info(f"Time spent to clean HTML filing: {time.time() - start:.1f} seconds")

    def extract_table_of_contents_from_html(self) -> list[str]:
        """Extract the table of contents from the filing HTML

        This method searches for the first table tag in the cleaned HTML content,
        and returns a list of strings representing the headings.
        """

        if self.filing_html_path is None or not self.filing_cleaned_html_path.exists():
            raise ValueError("Must download filing before extracting table of contents")

        html_content = self.filing_cleaned_html_path.read_text(encoding="utf-8")

        soup = BeautifulSoup(html_content, "html.parser")
        toc_table_tag = soup.find("table", recursive=True)

        toc: list[str] = []
        if toc_table_tag:
            for row in toc_table_tag.find_all("tr"):
                row_text = " ".join(cell.get_text(strip=True) for cell in row.find_all("td"))
                row_text = re.sub(r"\d+\s*$", "", row_text).strip()  # Remove trailing numbers
                row_text = re.sub(r"\s+", " ", row_text)
                if row_text.strip() == "Page" or not row_text.strip():
                    continue
                toc.append(row_text)

        return toc

    def extract_section_from_html(self, heading: str, next_heading: str) -> None:
        """Extract a section from the filing HTML between two headings.
        The section is saved as a Markdown file with a suffix based on the heading,
        and also returned as a string.

        This method is meant to be used for:
        - Management's Discussion and Analysis
        - Risk Factors
        """
        if self.filing_html_path is None:
            raise ValueError("Must download filing before extracting table of contents")
        start = time.time()
        html_content = self.filing_cleaned_html_path.read_text(encoding="utf-8")

        soup = BeautifulSoup(html_content, "html.parser")

        heading_div = soup.find(
            lambda tag: tag.name in ["div", "p"]
            and heading.upper() == tag.text.strip().upper()
            and tag.parent.name == "body"
        )
        # If not found try with only the heading before the first period, including the period
        heading_item = (heading.partition(".")[0] + ".").upper()
        next_heading_item = (next_heading.partition(".")[0] + ".").upper()
        if not heading_div:
            heading_div = soup.find(
                lambda tag: tag.name in ["div", "p"]
                and tag.text.strip().upper().startswith(heading_item)
                and tag.parent.name == "body"
            )

        # Extract all text between heading and next_heading, and ignore tables
        section_text = heading_div.get_text(strip=True) + "\n"
        if heading_div:
            for tag in heading_div.find_next_siblings():
                table_tag = tag.find("table")
                if table_tag:
                    continue
                if next_heading.upper() == tag.text.strip().upper() or (
                    next_heading_item and tag.text.strip().upper().startswith(next_heading_item)
                ):
                    break
                tag_text = tag.get_text(strip=True) + "\n"
                # Strip out any leading whitespace directly after a newline
                tag_text = re.sub(r"\n\s+", "\n", tag_text)
                section_text += tag_text

        for key in SECTION_FILENAME_MAP:
            if key in heading:
                suffix = SECTION_FILENAME_MAP[key]
                break

        filename = self.filing_markdown_path.stem + f"_{suffix}.md"
        filing_markdown_section_path = self.filing_markdown_path.with_name(filename)
        filing_markdown_section_path.write_text(section_text, encoding="utf-8")
        logger.info(f"Section stored as Markdown at: {filing_markdown_section_path}")
        logger.info(f"Time spent to extract section: {time.time() - start:.1f} seconds")


def table_to_markdown(table_tag: Tag) -> str:
    """Returns a Markdown representation of an HTML table

    This function iterates over the rows and extracts the text from each cell
    to create a Markdown table.

    Args:
        table_tag: BeautifulSoup Tag object representing the table
    """
    rows = table_tag.find_all("tr")

    headers: list[str] = []
    data: list[list[str]] = []
    for row in rows:
        if all(cell.get_text(strip=True) == "" for cell in row.find_all(["th", "td"])):
            # Skip empty rows
            continue
        md_row = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
        if not headers:
            headers = md_row
        else:
            data.append(md_row)

    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_rows = "\n".join("| " + " | ".join(row) + " |" for row in data)
    markdown_table = f"{header_row}\n{separator_row}\n{data_rows}"
    return markdown_table
