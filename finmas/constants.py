import yaml


CONFIG_FILE = "config.yaml"
INCOME_STATEMENT_COLS = ["totalRevenue", "operatingExpenses", "grossProfit", "netIncome"]

with open(CONFIG_FILE, "r") as c:
    config = yaml.safe_load(c)

defaults = config.get("defaults", {})
