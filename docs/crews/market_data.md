# Market Data Analysis Crew

The market data crew focus on fundamental and technical analysis of the specified ticker.
The data is formatted into a Markdown table and subsequently fed to the LLM agents
for interpretation and analysis. The end goal is to produce a summary based on the
market data, and argue for a buy, sell, or hold recommendation.

Illustration of the Market Data Analysis crew:

```mermaid
{% include 'diagrams/crews/market_data.mmd' %}
```

```python exec="on"
from pathlib import Path
from finmas.crews.utils import get_yaml_config_as_markdown

config_path = Path.cwd() / "finmas/crews/market_data/config"
print("## Agents")
print(get_yaml_config_as_markdown(config_path, "agents"))
print("## Tasks")
print(get_yaml_config_as_markdown(config_path, "tasks"))
```
