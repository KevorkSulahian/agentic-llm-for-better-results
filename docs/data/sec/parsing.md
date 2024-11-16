# SEC Filings Parsing

One of the aspects that make parsing an SEC filing challenging is that each company have
a slightly different format for the filing. Therefore we have implemented a custom
process to attempt to clean up the HTML content as much as is necessary to get a common
format for each filing. Then we extract the headings from the Table of Contents that is always present at the
start of the filing.
Then the headings can be used to extract the relevant sections from the filing.

The process is implemented in the [SECFilingParser](https://github.com/KevorkSulahian/agentic-llm-for-better-results/blob/main/finmas/data/sec/sec_parser.py) class and relies heavily
on the [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) package to parse the HTML content.

The following simplified diagram shows the overall steps for the parsing:

```mermaid
{% include 'diagrams/sec_filing_parsing.mmd' %}
```
