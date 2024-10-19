from bs4 import BeautifulSoup
import re
import unicodedata

SENTENCES_IGNORE_LIST = [
    "disclaimer",
    "benzinga",
    "photo by",
    "see also",
    "shutterstock",
    "read next",
    "click here",
]
HEADLINE_IGNORE_LIST = ["market clubhouse"]


def condense_newline(text):
    """Helper to reduce consecutive newlines into single newline"""
    return "\n".join([p for p in re.split("\n|\r", text) if len(p) > 0])


def get_benzinga_content_text(html_content: str, exclude_tables: bool = True) -> str:
    """Parses the HTML from a news content from Benzinga news source and returns the text.

    Args:
        html_content: The HTML content of the news article.
        exclude_tables: Whether to exclude tables from the text. Defaults to True.
    """

    soup = BeautifulSoup(html_content, "html.parser")

    if exclude_tables:
        for table in soup.find_all("table"):
            table.decompose()

    TAGS = ["p", "ul"]
    filtered_text_list = []
    for tag in soup.findAll(TAGS):
        text = condense_newline(tag.text)
        text = unicodedata.normalize("NFKD", text)  # Replace \xa0 with space
        text = re.sub(r"[\n\r]", "", text)  # Remove all newlines
        text = re.sub(r"\t", " ", text)  # Replace all tab characters with space
        text = re.sub(r"\s+", " ", text).strip()  # Condense whitespace

        if len(text) == 0:
            continue

        text = re.sub(r"\.\s*([A-Z])", r". \1", text)  # Ensure that there is a space after period.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        filtered_sentences = [
            sentence
            for sentence in sentences
            if not any(
                ignore_word.lower() in sentence.lower() for ignore_word in SENTENCES_IGNORE_LIST
            )
        ]
        if filtered_sentences and not filtered_sentences[-1].endswith("."):
            filtered_sentences[-1] += "."
        filtered_text_list.extend(filtered_sentences)

    return " ".join(filtered_text_list)
