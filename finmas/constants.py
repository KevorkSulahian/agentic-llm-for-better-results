import yaml


CONFIG_FILE = "config.yaml"
INCOME_STATEMENT_COLS = ["totalRevenue", "operatingExpenses", "grossProfit", "netIncome"]

# Tickers Table
TICKER_COLS = [
    "name",
    "market_cap",
    "sector",
    "industry_group",
    "industry",
    "market",
    "website",
]
MARKET_CAP_MAP = {
    "Mega Cap": 5,
    "Large Cap": 4,
    "Mid Cap": 3,
    "Small Cap": 2,
    "Micro Cap": 1,
    "Nano Cap": 0,
}

# SEC Filings Table
SEC_FILINGS_COLS = [
    "filing_date",
    "reportDate",
    "form",
    "link",
    "filing",
]

with open(CONFIG_FILE, "r") as c:
    config = yaml.safe_load(c)

defaults = config.get("defaults", {})
