# Combined Analysis Crew

The Combined Analysis crew is similar to the [Market Data crew](market_data.md), but it also includes
agents that can analyze news articles and the selected SEC filing. The main goal of this crew is to
provide a final stock recommendation based on the analysis of all the data sources. The crew
is restricted to 5 agents, and token usage on this crew can be quite high as it combines
multiple data sources. So it it's suggested to test this crew using a free alternative for
LLM model before using a paid LLM.

Illustration of the Combined Analysis crew:

```mermaid
{% include 'diagrams/crews/combined.mmd' %}
```

```python exec="on"
from pathlib import Path
from finmas.crews.utils import get_yaml_config_as_markdown

config_path = Path.cwd() / "finmas/crews/combined/config"
print("## Agents")
print(get_yaml_config_as_markdown(config_path, "agents"))
print("## Tasks")
print(get_yaml_config_as_markdown(config_path, "tasks"))
```
