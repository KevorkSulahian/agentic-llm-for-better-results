import yaml
from pathlib import Path

CONFIG_FILE = "config.yaml"
DOCS_URL = "https://kevorksulahian.github.io/agentic-llm-for-better-results/"
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
    "accession_number",
]

with open(CONFIG_FILE, "r") as c:
    config = yaml.safe_load(c)

defaults = config.get("defaults", {})

# CrewAI
agent_config = dict(
    max_iter=defaults["crewai"]["max_iterations"],
    max_rpm=defaults["crewai"]["max_requests_per_minute"],
)

Path(defaults["crew_logs_dir"]).mkdir(parents=True, exist_ok=True)
