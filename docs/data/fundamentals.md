# Fundamentals

Fundamental data consists mainly of the income statement and balance sheet of a company.
We extract these data from [Alpha Vantage](https://www.alphavantage.co/) to get a sufficient number of quarterly filing data.
The main line items of interest are:

- Total Revenue
- Gross Profit
- Operating Expenses
- Net Income
- Total Shareholder Equity
- Total Debt
- Number of Shares Outstanding

From these line items, we can calculate the following metrics and ratios:

- Net Profit Margin
- Earnings per Share (EPS)
- Price to Earnings Ratio (P/E)
- Price to Sales Ratio (P/S)
- Debt to Equity Ratio (D/E)

The processing of the fundamental data and conversion into Markdown tables that can be fed to the LLM models are
shown in the [StockFundamentalsTool](https://github.com/KevorkSulahian/agentic-llm-for-better-results/blob/main/finmas/data/market/fundamentals.py).
Especially computation of YoY growth rates is important for evaluating the trend and growth of a company.
