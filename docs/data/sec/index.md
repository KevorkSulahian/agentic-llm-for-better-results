# SEC Filings

In this project we have mainly focused on the 10-K (annual) and 10-Q (quarterly) SEC filings.
All of the major companies that are listed on NASDAQ or NYSE are required to file these reports.
These filings provide information about the historical performance of the company, and also
give information about what the future may hold for the company.

The filing information is fetched by using the [edgartools](https://github.com/dgunning/edgartools) package in Python, which provides
a lot of convenience functions to fetch the filings from the EDGAR database.
The SEC filing is downloaded locally, and subsequently [parsed to extract specific sections of the filing](parsing.md).
So that the LLM agents can focus in on the most relevant information.

## Which information are interesting to extract from the SEC Filings?

The financial statements like income statemen, balance sheet and cash flow statement are already
available from other data proivders that can provide such data in a structured format.

Therefore the focus of analysis of an SEC filing is more directed towards the unstructured text data
that exists in the filings. We have focused particularly on the following sections:

- Management's Discussion and Analysis (MD&A)
- Risk Factors

In these sections there is typically information regarding management's view of the company's performance
for that last period (year or quarter), and what are the key risks that the company is facing or will face
in the upcoming period.

## Forward-looking statements

A key aspect of SEC Filings is that there are a lot of what is called "forward-looking statements" from the management
in the report. These statements provide information about what are some of the expectations from management
about what the future holds for the company. This can indicate what growth trajectory is planned for the company
and that can be taken into consideration when analyzing the company.
