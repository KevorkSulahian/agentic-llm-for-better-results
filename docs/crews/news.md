# News Analysis Crew

The News crew consists of 3 agents with accompanying tasks, and the main goal is to provide
a summary of the news articles that are defined by extracting key insights and overview
of the sentiment of the news.

Illustration of the News Analysis crew:

```mermaid
{% include 'diagrams/crews/news.mmd' %}
```

```python exec="on"
from pathlib import Path
from finmas.crews.utils import get_yaml_config_as_markdown

config_path = Path.cwd() / "finmas/crews/news/config"
print("## Agents")
print(get_yaml_config_as_markdown(config_path, "agents"))
print("## Tasks")
print(get_yaml_config_as_markdown(config_path, "tasks"))
```
