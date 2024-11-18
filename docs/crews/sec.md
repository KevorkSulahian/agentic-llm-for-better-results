# SEC MD&A and Risk Factors Analysis

The SEC Filing Analysis crew consists of 3 agents and the main goal is to extract
the key insights from the sections Management's Discussion and Analysis (MD&A) and Risk Factors of the
target SEC Filing. An 10-K or 10-Q filing can be quite an extensive document and time consuming to read.
Therefore, the goal is to analyze information from key sections that are not available in the financial statements.
The crew does not support tables nor images in the filing, and only focus on the textual information
that can be extracted.

Illustration of the SEC Filing Analysis crew:

```mermaid
{% include 'diagrams/crews/sec.mmd' %}
```

```python exec="on"
from pathlib import Path
from finmas.crews.utils import get_yaml_config_as_markdown

config_path = Path.cwd() / "finmas/crews/sec_mda_risk_factors/config"
print("## Agents")
print(get_yaml_config_as_markdown(config_path, "agents"))
print("## Tasks")
print(get_yaml_config_as_markdown(config_path, "tasks"))
```
